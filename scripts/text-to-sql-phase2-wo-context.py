import contextlib
import logging
import os
import re
from pathlib import Path

import hydra
import torch
import wandb
from datasets import Dataset, DatasetDict, load_dataset
from datasets.exceptions import DatasetNotFoundError
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainerCallback,
)
from trl import SFTConfig

from configs.fine_tune.conf_schema import register_phase2_wo_ctx
from utils.checkpoint import (
    CheckpointMetadataCallback,
    CheckpointResolution,
    mark_training_completed,
    resolve_checkpoint,
)
from utils.evaluation_artifacts import (
    attach_sample_ids,
    create_evaluation_run,
    evaluate_artifacts,
    export_predictions,
    generation_batch,
)
from utils.hub import create_model_repo, load_with_retry, retry_hub_io
from utils.logging import setup_logging
from utils.system import check_secrets, seed_everything
from utils.resources import SystemResourceTracker, resource_output_dir
from utils.torch_utils import (
    ddp_cleanup,
    ddp_setup,
    rank_zero_evaluation,
    resolve_torch_dtype,
)
from utils.training import record_trainer_config
from utils.sft_trainer import ResumableSFTTrainer as SFTTrainer
from utils.wandb import checkpoint_run_identity, training_required_secrets, wandb_mode

logger = logging.getLogger(__name__)


def process_spider(sample):
    """Format raw Spider rows using the database ID as the context field."""
    return {
        "text": (
            f"Context: {sample['db_id']}\n"
            f"Question: {sample['question']}\n"
            f"Answer: {sample['query']}"
        )
    }


@retry_hub_io()
def load_model_with_retry(model_id, **kwargs):
    return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)


@retry_hub_io(attempts=5, min_wait=4, max_wait=60)
def load_dataset_with_retry(dataset_id: str, split: str) -> Dataset:
    logger.info(
        f"Attempting to load '{dataset_id}' (split={split}) from Hugging Face..."
    )
    dataset = load_dataset(dataset_id, split=split)
    logger.info(f"  -> {len(dataset)} rows loaded.")
    return dataset


def build_model_and_tokenizer(
    cfg: DictConfig,
    *,
    local_rank: int,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load base model with 4-bit quantization, apply Phase 1 LoRA adapters,
    and prepare the tokenizer.

    Unlike Phase 1, Phase 2 does not train a fresh LoRA adapter -- it loads the
    previously saved Phase 1 adapter via ``PeftModel.from_pretrained`` so that
    fine-tuning continues from the already syntax-anchored weights.

    Each process pins the model to its own GPU via ``device_map={"":local_rank}``
    instead of ``device_map="auto"``, which is incompatible with DDP.

    Args:
        cfg: Hydra configuration containing model and quantization parameters.
        local_rank: GPU index for this DDP worker.

    Returns:
        tuple[AutoModelForCausalLM, AutoTokenizer]: Prepared PEFT model and its tokenizer.
    """

    compute_dtype = resolve_torch_dtype(cfg.model.compute_dtype)
    torch_dtype = resolve_torch_dtype(cfg.model.torch_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.model.quantization.load_in_4bit,
        bnb_4bit_quant_type=cfg.model.quantization.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg.model.quantization.bnb_4bit_use_double_quant,
    )

    # Pin each DDP worker to its own GPU.
    base_model = load_model_with_retry(
        cfg.model.base_id,
        quantization_config=bnb_config,
        dtype=torch_dtype,
        device_map={"": local_rank},
        use_cache=False,
        attn_implementation=cfg.model.attn_implementation,
    )

    base_model = prepare_model_for_kbit_training(base_model)

    logger.info(
        "Loading Phase 1 adapters from '%s'...", cfg.hub.phase1_adapters_repo_id
    )
    model = load_with_retry(
        PeftModel.from_pretrained,
        base_model,
        cfg.hub.phase1_adapters_repo_id,
        is_trainable=True,
    )

    logger.info("All adapter weights normalised to float16/float32.")

    tokenizer = load_with_retry(
        AutoTokenizer.from_pretrained, cfg.model.base_id, fix_mistral_regex=True
    )
    if tokenizer.pad_token is None and tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"

    return model, tokenizer


# ── Spider Evaluation Callback ─────────────────────────────────────────────────


class SpiderEvaluationCallback(TrainerCallback):
    """Evaluate on the full Spider validation set using exact-match and
    execution-accuracy metrics from the official Spider evaluation script.

    Predictions are retained as JSONL and exported through the shared adapter.
    Partial-match scores (per SQL clause) are also uploaded.

    All ranks enter the hook; only rank zero generates and logs predictions.
    Other ranks wait on a CPU control group before returning to training.
    """

    def __init__(
        self,
        run: "wandb.sdk.wandb_run.Run | None",
        eval_dataset: "Dataset",
        tokenizer: AutoTokenizer,
        model: AutoModelForCausalLM,
        db_path: str,
        table_path: str,
        output_dir: str,
        etype: str,
        dataset_id: str = "training-validation",
        is_main_process: bool = True,
        evaluation_timeout_seconds: float = 86400,
        wait_time: float = 60.0,
    ):
        from utils.sql_execution import QueryLimits

        self.query_limits = QueryLimits(wait_time=wait_time)
        self.run = run
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.model = model
        self.db_path = db_path
        self.table_path = table_path
        self.etype = etype
        self.is_main_process = is_main_process
        self.evaluation_timeout_seconds = evaluation_timeout_seconds

        self.evaluation_run = (
            create_evaluation_run(
                eval_dataset,
                output_dir=output_dir,
                dataset_id=dataset_id,
                split="validation",
            )
            if is_main_process
            else None
        )
        if self.evaluation_run is not None:
            self.eval_dataset = attach_sample_ids(eval_dataset, self.evaluation_run)

    # ── Extraction helpers ────────────────────────────────────────────────

    def extract_context(self, text: str) -> str:
        """Extract the database ID between 'Context:' and 'Question:'."""
        match = re.search(
            r"Context:\s*(.*?)\s*^Question:", text, re.MULTILINE | re.DOTALL
        )
        return match.group(1).strip() if match else "Context not found"

    def extract_question(self, text: str) -> str:
        """Extract the natural-language question between 'Question:' and 'Answer:'."""
        match = re.search(
            r"^Question:\s*(.*?)\s*^Answer:", text, re.MULTILINE | re.DOTALL
        )
        return match.group(1).strip() if match else "Question not found"

    def extract_answer(self, text: str) -> str:
        """Extract the SQL query that follows 'Answer:'."""
        match = re.search(r"Answer:\s*(.*)", text, re.DOTALL)
        return match.group(1).strip() if match else "No answer found"

    # ── Partial-match table builder ───────────────────────────────────────

    def build_partial_tables(self, scores: dict, metric: str) -> list:
        splits = ["easy", "medium", "hard", "extra", "all"]
        clauses: set = set()
        for s in splits:
            clauses.update(scores[s].get("partial", {}).keys())
        clauses = sorted(clauses)

        rows = []
        for clause in clauses:
            row = [clause]
            for s in splits:
                row.append(scores[s].get("partial", {}).get(clause, {}).get(metric, 0))
            rows.append(row)
        return rows

    # ── Trainer hook ─────────────────────────────────────────────────────

    def on_evaluate(self, args, state, control, **kwargs):
        with rank_zero_evaluation(
            is_main_process=self.is_main_process,
            timeout_seconds=self.evaluation_timeout_seconds,
        ):
            if self.is_main_process:
                self._evaluate(args, state, control, **kwargs)

    def _evaluate(self, args, state, control, **kwargs):
        from functools import partial

        from scripts.evaluation import build_foreign_key_map_from_json, evaluate

        epoch = state.epoch if state.epoch is not None else -1
        model_key = f"epoch-{epoch:g}-step-{state.global_step}"
        logger.info("Starting Spider evaluation: %s", model_key)
        train_loss = next(
            (log["loss"] for log in reversed(state.log_history or []) if "loss" in log),
            0.0,
        )
        columns = [
            "Epoch",
            "Train Loss",
            "Context",
            "Question",
            "True SQL",
            "Generated SQL",
            "Status",
        ]
        table_data = []
        records = []
        was_training = self.model.training
        self.model.eval()
        try:
            manifest_by_id = {
                sample["sample_id"]: sample for sample in self.evaluation_run.manifest
            }
            for sample in self.eval_dataset:
                manifest_sample = manifest_by_id[sample["_evaluation_sample_id"]]

                def generate(sample=sample):
                    inputs = self.tokenizer(
                        f"Context: {sample['db_id']}\nQuestion: {sample['question']}\nAnswer:",
                        return_tensors="pt",
                    ).to(self.model.device)
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=2048,
                            pad_token_id=self.tokenizer.pad_token_id,
                            synced_gpus=False,
                        )
                    input_len = inputs.input_ids.shape[-1]
                    return [
                        self.tokenizer.decode(
                            outputs[0, input_len:], skip_special_tokens=True
                        )
                    ]

                batch_records = generation_batch(
                    [
                        {
                            "sample_id": sample["_evaluation_sample_id"],
                            "db_id": sample["db_id"],
                        }
                    ],
                    generate,
                    oom_exception=torch.cuda.OutOfMemoryError,
                )
                records.extend(batch_records)
                record = batch_records[0]
                table_data.append(
                    [
                        epoch,
                        train_loss,
                        sample["db_id"],
                        sample["question"],
                        manifest_sample["gold_sql"],
                        record["raw_output"],
                        record["status"],
                    ]
                )
                if record["status"] not in {"ok", "empty"}:
                    logger.error(
                        "Generation failed for %s: %s",
                        record["sample_id"],
                        record["error"],
                    )
                if record["status"] == "oom":
                    torch.cuda.empty_cache()

            artifacts = export_predictions(
                self.evaluation_run,
                records,
                model_key=model_key,
                model_id=str(
                    getattr(self.model.config, "_name_or_path", "training-model")
                ),
            )
            kmaps = build_foreign_key_map_from_json(self.table_path)
            scores = evaluate_artifacts(
                artifacts,
                evaluate=partial(evaluate, limits=self.query_limits),
                db_dir=self.db_path,
                etype=self.etype,
                kmaps=kmaps,
            )

            metrics_to_log = {
                "train/global_step": state.global_step,
                "epoch": epoch,
                "eval/sample_count": artifacts.sample_count,
            }
            for status, count in artifacts.status_counts.items():
                metrics_to_log[f"eval/generation_status/{status}"] = count
            for diff, data in scores.items():
                metrics_to_log[f"eval/{diff}_exact_match"] = data["exact"]
                metrics_to_log[f"eval/{diff}_execution_accuracy"] = data["exec"]

            if self.run is not None:
                self.run.log(metrics_to_log)
                self.run.log(
                    {
                        "train/global_step": state.global_step,
                        f"predictions_{model_key}": wandb.Table(
                            columns=columns, data=table_data
                        ),
                    }
                )
                artifact = wandb.Artifact(
                    name=f"spider_predictions_{model_key}", type="predictions"
                )
                artifact.add_dir(str(artifacts.path))
                artifact.add_file(str(self.evaluation_run.path / "manifest.jsonl"))
                artifact.add_file(str(self.evaluation_run.path / "dataset.json"))
                artifact.add_file(str(artifacts.gold_path))
                self.run.log_artifact(artifact)
                partial_cols = ["clause", "easy", "medium", "hard", "extra", "all"]
                for name, metric in (
                    ("recall", "rec"),
                    ("accuracy", "acc"),
                    ("f1", "f1"),
                ):
                    self.run.log(
                        {
                            "train/global_step": state.global_step,
                            f"partial_{name}_{model_key}": wandb.Table(
                                columns=partial_cols,
                                data=self.build_partial_tables(scores, metric),
                            ),
                        }
                    )
        except Exception:
            logger.exception(
                "Spider evaluation failed; no scores published for %s", model_key
            )
            raise
        finally:
            self.model.train(was_training)


def init_wandb(
    cfg: DictConfig, checkpoint: CheckpointResolution
) -> wandb.sdk.wandb_run.Run:
    """Initialize Weights & Biases run with configuration and run ID tracking.

    Args:
        cfg: Hydra configuration containing wandb settings.

    Returns:
        wandb.sdk.wandb_run.Run: Initialized Weights & Biases run object.
    """
    mode = wandb_mode(cfg)
    if mode == "online":
        wandb.login()
    if wandb.run is not None:
        wandb.finish()

    config = OmegaConf.to_container(cfg, resolve=True)

    identity = checkpoint_run_identity(
        checkpoint.metadata, resuming=checkpoint.path is not None
    )
    if mode != "online":
        # Offline runs keep their identity, but W&B's server resume policy is inapplicable.
        identity.pop("resume")

    return wandb.init(
        mode=mode,
        project=cfg.wandb.project_name,
        name=cfg.wandb.run_name,
        entity=cfg.wandb.entity,
        **identity,
        tags=list(cfg.wandb.tags),
        notes=cfg.wandb.notes,
        config=config,
    )


def build_trainer(
    cfg: DictConfig,
    run: wandb.sdk.wandb_run.Run | None,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_dataset: Dataset | DatasetDict,
    eval_dataset: Dataset | DatasetDict,
    *,
    is_main_process: bool,
    checkpoint_metadata: dict | None = None,
) -> SFTTrainer:
    """Build SFTTrainer configured with Spider evaluation callback.

    Unlike Phase 1, no ``peft_config`` is passed because the model already
    has the Phase 1 LoRA adapters attached and set as trainable.

    Args:
        cfg: Hydra configuration.
        run: Active W&B run on rank-0; ``None`` on all other ranks.
        model: PEFT model with Phase 1 adapters loaded and trainable.
        tokenizer: Initialized tokenizer.
        train_dataset: Training dataset split.
        eval_dataset: Evaluation dataset split.
        is_main_process: Whether the current process is rank-0.

    Returns:
        SFTTrainer: Configured SFTTrainer instance.
    """

    if cfg.hub.push_to_hub and is_main_process:
        create_model_repo(cfg.hub.repo_id, private=cfg.hub.private)

    args = SFTConfig(
        output_dir=cfg.paths.adapter_output_path,
        seed=cfg.system.seed,
        data_seed=cfg.dataset.seed,
        train_sampling_strategy=cfg.training.train_sampling_strategy,
        num_train_epochs=cfg.hyperparameters.epochs,
        max_steps=cfg.hyperparameters.max_steps,
        per_device_train_batch_size=cfg.hyperparameters.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.hyperparameters.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.hyperparameters.gradient_accumulation_steps,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        learning_rate=cfg.hyperparameters.learning_rate,
        optim=cfg.hyperparameters.optim,
        warmup_steps=cfg.hyperparameters.warmup_steps,
        weight_decay=cfg.hyperparameters.weight_decay,
        lr_scheduler_type=cfg.hyperparameters.lr_scheduler_type,
        fp16=cfg.training.fp16,
        bf16=cfg.training.bf16,
        logging_steps=cfg.training.logging_steps,
        ddp_find_unused_parameters=cfg.training.ddp_find_unused_parameters,
        dataset_text_field=cfg.dataset.text_field,
        max_length=cfg.model.tokenizer_max_length,
        # Offline mode still exercises logging; disabled mode skips the integration.
        report_to="wandb"
        if is_main_process and cfg.wandb.mode != "disabled"
        else "none",
        load_best_model_at_end=cfg.training.load_best_model_at_end,
        metric_for_best_model=cfg.training.metric_for_best_model,
        greater_is_better=cfg.training.greater_is_better,
        save_strategy=cfg.training.save_strategy,
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        eval_strategy=cfg.training.eval_strategy,
        eval_steps=cfg.training.eval_steps,
        push_to_hub=cfg.hub.push_to_hub,
        hub_model_id=cfg.hub.repo_id,
        hub_private_repo=cfg.hub.private,
        hub_strategy=cfg.hub.strategy,
        save_only_model=False,
    )

    spider_eval_callback = SpiderEvaluationCallback(
        run=run,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        model=model,
        db_path=cfg.spider.db_path,
        table_path=cfg.spider.table_path,
        output_dir=cfg.spider.output_dir,
        dataset_id=cfg.dataset.id,
        etype=cfg.spider.etype,
        is_main_process=is_main_process,
        evaluation_timeout_seconds=cfg.spider.evaluation_timeout_seconds,
        wait_time=cfg.sql_execution.wait_time,
    )

    callbacks = [spider_eval_callback]
    if checkpoint_metadata is not None:
        callbacks.append(CheckpointMetadataCallback(checkpoint_metadata))
    es = cfg.early_stopping
    if es.enabled:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=es.patience,
                early_stopping_threshold=es.threshold,
            )
        )

    trainer = SFTTrainer(
        allow_world_size_change=cfg.resume.allow_world_size_change,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=args,
        # No peft_config: the Phase 1 adapter is already attached to the model.
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    if is_main_process:
        record_trainer_config(trainer.args, run=run)
    return trainer


def validate_early_stopping(cfg: DictConfig) -> None:
    if not cfg.early_stopping.enabled:
        return
    t = cfg.training
    if not t.load_best_model_at_end:
        raise ValueError("Early stopping requires load_best_model_at_end=True")
    if not t.eval_strategy:
        raise ValueError("Early stopping needs periodic evaluation")
    if t.eval_strategy != t.save_strategy:
        raise ValueError("eval_strategy and save_strategy must match")
    if t.save_strategy == "steps" and t.save_steps % t.eval_steps != 0:
        raise ValueError("save_steps must be a multiple of eval_steps")
    if t.save_total_limit is None or t.save_total_limit < 2:
        raise ValueError("Keep at least 2 checkpoints so best + latest both survive")


def run_training(
    cfg: DictConfig,
    train_dataset: Dataset | DatasetDict,
    eval_dataset: Dataset | DatasetDict,
    *,
    local_rank: int,
    is_main_process: bool,
) -> tuple[dict[str, float], dict[str, str]]:
    """Coordinate end-to-end Phase 2 model fine-tuning and return training metrics and saved paths.

    Args:
        cfg: Hydra configuration.
        train_dataset: Training dataset split.
        eval_dataset: Evaluation dataset split.
        local_rank: GPU index for this DDP worker.
        is_main_process: Whether the current process is rank-0.

    Returns:
        tuple[dict[str, float], dict[str, str]]: A tuple containing:
            - metrics: Dictionary of training metrics.
            - paths: Dictionary of saved artifact paths (e.g., output_dir).
    """
    # Resolve before model allocation and W&B initialization on every rank.
    checkpoint = resolve_checkpoint(
        cfg,
        phase="phase2_wo_ctx",
        is_main_process=is_main_process,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    if checkpoint.completed:
        logger.info(
            "Attempt %s is already complete at step %s; skipping model loading and training.",
            checkpoint.metadata["attempt_id"],
            checkpoint.global_step,
        )
        if checkpoint.completion is None:
            logger.warning(
                "The saved maximum training step count was reached, but final exports are not confirmed by a completion record."
            )
        metrics = (
            dict(checkpoint.completion["metrics"]) if checkpoint.completion else {}
        )
        metrics.update(already_completed=1, global_step=checkpoint.global_step)
        return metrics, {
            "output_dir": checkpoint.output_dir,
            "status": "already_completed",
        }
    model, tokenizer = build_model_and_tokenizer(cfg, local_rank=local_rank)

    # W&B login and run initialisation must only happen on rank-0 to avoid
    # creating N parallel runs for a single training job.
    run_ctx = (
        init_wandb(cfg, checkpoint) if is_main_process else contextlib.nullcontext(None)
    )

    metrics = {}

    with run_ctx as run:
        # Every rank samples itself; only global rank zero forwards shared files to W&B.
        tracker = SystemResourceTracker(
            run,
            interval=cfg.system.monitor_interval,
            output_dir=resource_output_dir(
                "training",
                "phase2-wo-context",
                f"attempt-{checkpoint.metadata['attempt_id']}",
                root=cfg.system.get("resource_output_root", "artifacts/resources"),
            ),
            session_id=checkpoint.monitoring_session_id or None,
            local_rank=local_rank,
            device_indices=[local_rank],
            buffer_size=cfg.system.get("monitor_buffer_size", 1024),
            shutdown_timeout=cfg.system.get("monitor_shutdown_timeout", 5.0),
        )
        try:
            tracker.start()
            trainer = build_trainer(
                cfg=cfg,
                run=run,
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                is_main_process=is_main_process,
                checkpoint_metadata=checkpoint.metadata,
            )
            trainer.add_callback(tracker.callback())
            if is_main_process:
                trainer.model.print_trainable_parameters()

            train_result = trainer.train(resume_from_checkpoint=checkpoint.path)

            # Trainer.save_model / save_metrics / Hub push are guarded by
            # Trainer.is_world_process_zero() internally, so they only run
            # once per job even in multi-process setups.
            trainer.save_model(cfg.paths.adapter_output_path)
            tokenizer.save_pretrained(cfg.paths.adapter_output_path)

            metrics = dict(train_result.metrics)
            trainer.save_metrics("train", metrics)
            if is_main_process:
                mark_training_completed(
                    checkpoint,
                    trainer.state,
                    metrics,
                    repo_id=cfg.hub.repo_id if cfg.hub.push_to_hub else None,
                )
        finally:
            tracker.stop()

    saved_paths = {
        "output_dir": cfg.paths.adapter_output_path,
    }

    if not metrics:
        raise RuntimeError("Training produced no metrics — check logs above.")

    return metrics, saved_paths


# ====================================================
#                       Main
# ====================================================

# Schema must be registered before @hydra.main resolves the config.
register_phase2_wo_ctx()


@hydra.main(
    version_base=None, config_path="pkg://configs", config_name="fine_tune/phase2_wo_ctx"
)
def main(cfg: DictConfig) -> None:

    # Initialise the process group and detect rank so we can gate logging and side effects.
    local_rank: int = ddp_setup()
    is_main_process: bool = int(os.environ["RANK"]) == 0

    # Configure logging: rank-0 emits INFO; other ranks emit WARNING only
    log_filename = f"{Path(__file__).stem}.log"
    setup_logging(
        level=logging.INFO if is_main_process else logging.WARNING,
        filename=log_filename,
        rank=int(os.environ.get("RANK", local_rank)),
    )

    # Forbid accidental typo'd keys anywhere in the resolved config.
    OmegaConf.set_struct(cfg, True)

    # Validate tracking mode on every rank; W&B needs no key offline or disabled.
    required_secrets = training_required_secrets(cfg)
    if is_main_process:
        check_secrets(required_secrets)

    validate_early_stopping(cfg)

    seed_everything(cfg.system.seed)

    # ── Dataset loading ────────────────────────────────────────────────────────
    # All ranks load the dataset independently; HuggingFace Datasets uses a
    # file-lock-based cache so concurrent reads are safe.  The Trainer's built-in
    # DistributedSampler automatically shards data across processes.
    try:
        ds_train = load_dataset_with_retry(cfg.dataset.id, split="train")
        ds_val = load_dataset_with_retry(cfg.dataset.id, split="validation")

        ds_train = ds_train.map(
            process_spider, remove_columns=ds_train.column_names
        )
        # Keep query/db_id/question for manifest-based Spider evaluation.
        ds_val = ds_val.map(process_spider)

        logger.info(
            f"Successfully loaded and processed {len(ds_train)} train samples "
            f"and {len(ds_val)} validation samples."
        )
    except DatasetNotFoundError:
        msg = f"Dataset '{cfg.dataset.id}' not found on Hugging Face Hub. Please check the name."
        logger.error(msg)
        raise DatasetNotFoundError(msg)
    except Exception as e:
        msg = f"{type(e).__name__} - Fatal Data Loading/Mapping Error : {e}"
        logger.exception(msg)
        raise RuntimeError(msg)

    # ── Smoke-test dataset slicing ────────────────────────────────────────────
    smoke = cfg.get("smoke_test", None)
    if smoke and smoke.enabled:
        n_train = cfg.smoke_test.max_train_samples
        n_eval = cfg.smoke_test.max_eval_samples
        if n_train is not None:
            ds_train = ds_train.select(range(min(n_train, len(ds_train))))
            logger.info("[smoke] Capped train dataset to %d samples.", len(ds_train))
        if n_eval is not None:
            ds_val = ds_val.select(range(min(n_eval, len(ds_val))))
            logger.info("[smoke] Capped eval dataset to %d samples.", len(ds_val))

    metrics, paths = run_training(
        cfg=cfg,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        local_rank=local_rank,
        is_main_process=is_main_process,
    )

    if is_main_process:
        logger.info(
            "Training status: %s. Output location: %s",
            paths.get("status", "finished"),
            paths["output_dir"],
        )
        logger.info("Final metrics: %s", metrics)

    ddp_cleanup()


if __name__ == "__main__":
    main()
