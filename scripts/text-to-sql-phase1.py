import contextlib
import logging
import os
from pathlib import Path

import hydra
import wandb
from datasets import Dataset, DatasetDict, load_dataset
from datasets.exceptions import DatasetNotFoundError
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from trl import SFTConfig

from configs.fine_tune.conf_schema import register_phase1
from utils.checkpoint import (
    CheckpointMetadataCallback,
    CheckpointResolution,
    mark_training_completed,
    resolve_checkpoint,
)
from utils.hub import create_model_repo, load_with_retry, retry_hub_io
from utils.logging import setup_logging
from utils.model import SQLSampleCallback
from utils.system import check_secrets, seed_everything
from utils.resources import SystemResourceTracker, resource_output_dir
from utils.torch_utils import ddp_cleanup, ddp_setup, resolve_torch_dtype
from utils.training import record_trainer_config
from utils.sft_trainer import ResumableSFTTrainer as SFTTrainer
from utils.wandb import checkpoint_run_identity, training_required_secrets, wandb_mode

logger = logging.getLogger(__name__)


def process_gretelai(sample):
    return {
        "text": (
            f"Context: {sample['sql_context']}\n"
            f"Question: {sample['sql_prompt']}\n"
            f"Answer: {sample['sql']}"
        )
    }


@retry_hub_io()
def load_model_with_retry(model_id, **kwargs):
    return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)


@retry_hub_io(attempts=5, min_wait=4, max_wait=60)
def load_dataset_with_retry(dataset_id: str):
    logger.info(f"Attempting to load {dataset_id} from Hugging Face...")
    dataset = load_dataset(dataset_id)
    logger.info(
        f"  → {len(dataset['train'])} train rows, {len(dataset['validation'])} validation rows loaded."
    )
    return dataset


def build_model_and_tokenizer(
    cfg: DictConfig,
    *,
    local_rank: int,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Build and prepare the 4-bit quantized model and tokenizer.

    Each process pins the model to its own GPU via ``device_map={"":local_rank}``
    instead of ``device_map="auto"``, which spreads layers across ALL visible
    GPUs and is incompatible with DDP (each process must own a single replica).

    Args:
        cfg: Hydra configuration containing model and quantization parameters.
        local_rank: GPU index for this DDP worker.

    Returns:
        tuple[AutoModelForCausalLM, AutoTokenizer]: Prepared base causal model and its tokenizer.
    """
    compute_dtype = resolve_torch_dtype(cfg.model.compute_dtype)
    torch_dtype = resolve_torch_dtype(cfg.model.torch_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.model.quantization.load_in_4bit,
        bnb_4bit_quant_type=cfg.model.quantization.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg.model.quantization.bnb_4bit_use_double_quant,
    )

    # Pin each DDP worker to its own GPU so the process group can wrap a
    # single contiguous model replica without cross-device tensor moves.
    model = load_model_with_retry(
        cfg.model.base_id,
        quantization_config=bnb_config,
        dtype=torch_dtype,
        device_map={"": local_rank},
        use_cache=False,
        attn_implementation=cfg.model.attn_implementation,
    )

    tokenizer = load_with_retry(
        AutoTokenizer.from_pretrained, cfg.model.base_id, fix_mistral_regex=True
    )
    if tokenizer.pad_token is None and tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"

    model = prepare_model_for_kbit_training(model)

    return model, tokenizer


def build_lora_config(cfg: DictConfig) -> LoraConfig:
    """Build LoRA configuration for parameter-efficient fine-tuning (PEFT).

    Args:
        cfg: Hydra configuration containing PEFT settings.

    Returns:
        LoraConfig: Configured LoRA specification for PEFT adapter training.
    """
    return LoraConfig(
        r=cfg.peft.r,
        lora_alpha=cfg.peft.lora_alpha,
        lora_dropout=cfg.peft.lora_dropout,
        target_modules=list(cfg.peft.target_modules),
        bias=cfg.peft.bias,
        task_type=cfg.peft.task_type,
    )


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
    peft_config: LoraConfig,
    train_dataset: Dataset | DatasetDict,
    eval_dataset: Dataset | DatasetDict,
    *,
    is_main_process: bool,
    checkpoint_metadata: dict | None = None,
) -> SFTTrainer:
    """Build SFTTrainer configured with training hyperparameters and evaluation callbacks.

    Args:
        cfg: Hydra configuration.
        run: Active W&B run on rank-0; ``None`` on all other ranks.
        model: Base causal model prepared for LoRA fine-tuning.
        tokenizer: Initialized tokenizer.
        peft_config: LoRA PEFT adapter configuration.
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

    # SQLSampleCallback calls model.generate() and logs a W&B Table — both
    # must only happen on rank-0.  Pass is_main_process so the callback can
    # guard those side effects without importing this module's globals.
    sql_sample_callback = SQLSampleCallback(
        run=run,
        model=model,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        samples_num=cfg.wandb.samples_num,
        seed=cfg.system.seed,
        is_main_process=is_main_process,
    )

    callbacks = [sql_sample_callback]
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
        peft_config=peft_config,
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
    """Coordinate end-to-end model fine-tuning and return training metrics and saved paths.

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
        phase="phase1",
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
    peft_config = build_lora_config(cfg)

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
                "phase1",
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
                peft_config=peft_config,
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
register_phase1()


@hydra.main(
    version_base=None, config_path="pkg://configs", config_name="fine_tune/phase1"
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
        dataset = load_dataset_with_retry(cfg.dataset.id)
        ds_train = dataset["train"]
        ds_val = dataset["validation"]
        ds_train = ds_train.map(process_gretelai, remove_columns=ds_train.column_names)
        ds_val = ds_val.map(process_gretelai, remove_columns=ds_val.column_names)

        logger.info(
            f"Successfully loaded and processed {len(ds_train)} train samples and {len(ds_val)} validation samples."
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
