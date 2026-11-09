import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import wandb
import gc
import logging
import requests
from tempfile import TemporaryDirectory
from utils.logging import setup_logging
from pathlib import Path
from tqdm import tqdm
from configs.final_eval.conf_schema import register_configs
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.evaluation import evaluate, build_foreign_key_map_from_json
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
from torch.utils.data import DataLoader
from contextlib import contextmanager

from utils.model import calculate_perplexity
from utils.hub import load_with_retry, retry_hub_io
from utils.system import check_secrets
from utils.resources import SystemResourceTracker, resource_output_dir
from utils.evaluation_artifacts import (
    attach_sample_ids,
    create_evaluation_run,
    evaluate_artifacts,
    export_predictions,
    generation_batch,
)

logger = logging.getLogger(__name__)

register_configs()

# ====================================================
#               Checking helping functions
# ====================================================


def check_paths(cfg: DictConfig) -> None:
    """Check that the paths specified in the config exist and are writable."""

    if cfg is None or cfg.eval_paths is None:
        raise ValueError("Configuration or eval_paths is None")

    test_db_path = cfg.eval_paths.get("test_db_path")
    test_table_path = cfg.eval_paths.get("test_table_path")
    paths_to_check = [test_db_path, test_table_path]

    for path in paths_to_check:
        if not path:
            raise ValueError(f"Path is not specified in the configuration: {path}")

    test_db_path_obj = Path(test_db_path)

    if not test_db_path_obj.is_dir():
        raise FileNotFoundError(
            f"The following evaluation directory is missing {test_db_path}"
        )

    test_table_path_obj = Path(test_table_path)

    if not test_table_path_obj.is_file():
        raise FileNotFoundError(
            f"The following evaluation file is missing {test_table_path}"
        )

    try:
        with test_table_path_obj.open("rb"):
            pass
    except PermissionError:
        raise PermissionError(f"Read permission denied for file: {test_table_path_obj}")

    logger.info("All evaluation paths verified.")


# ====================================================
#                   Data Processing
# ====================================================


def process_dataset(sample):
    return {
        "text": (
            f"Context: {sample['context']}\n"
            f"Question: {sample['question']}\n"
            f"Answer: {sample['query']}"
        )
    }


# ========================================================================================================
#               Helper fucntions for wandb logging, ressource cleaning and evaluation
# ========================================================================================================


def build_partial_tables(data, metric):

    splits = ["easy", "medium", "hard", "extra", "all"]

    # collect all clause names first
    clauses = set()
    for s in splits:
        clauses.update(data[s].get("partial", {}).keys())

    clauses = sorted(clauses)

    logger.info(f"clauses: {clauses}")

    # build table
    rows = []
    for clause in clauses:
        row = [clause]
        for s in splits:
            row.append(data[s].get("partial", {}).get(clause, {}).get(metric, 0))
        rows.append(row)

    return rows


def run_spider_eval_suite(artifacts, run_name, cfg: DictConfig):
    from functools import partial

    from utils.sql_execution import QueryLimits

    try:
        kmaps = build_foreign_key_map_from_json(cfg.eval_paths.test_table_path)
        scorer = partial(evaluate, limits=QueryLimits(wait_time=cfg.sql_execution.wait_time))
        if not getattr(cfg.eval_paths, "strict_parsing", True):
            scorer = partial(scorer, strict_parsing=False)

        scores = evaluate_artifacts(
            artifacts,
            evaluate=scorer,
            db_dir=cfg.eval_paths.test_db_path,
            etype=cfg.eval_paths.etype,
            kmaps=kmaps,
        )

        metrics_to_log = {}
        metrics_to_log["eval/sample_count"] = artifacts.sample_count
        for status, count in artifacts.status_counts.items():
            metrics_to_log[f"eval/generation_status/{status}"] = count

        for diff, data in scores.items():
            metrics_to_log[f"eval/{diff}_exact_match"] = data["exact"]
            metrics_to_log[f"eval/{diff}_execution_accuracy"] = data["exec"]

        wandb.log(metrics_to_log)

        partial_recall_tab = build_partial_tables(scores, "rec")
        partial_accuracy_tab = build_partial_tables(scores, "acc")
        partial_f1_tab = build_partial_tables(scores, "f1")

        wandb.log(
            {
                f"partial_recall_{run_name}": wandb.Table(
                    columns=["clause", "easy", "medium", "hard", "extra", "all"],
                    data=partial_recall_tab,
                )
            }
        )
        wandb.log(
            {
                f"partial_accuracy_{run_name}": wandb.Table(
                    columns=["clause", "easy", "medium", "hard", "extra", "all"],
                    data=partial_accuracy_tab,
                )
            }
        )
        wandb.log(
            {
                f"partial_f1_{run_name}": wandb.Table(
                    columns=["clause", "easy", "medium", "hard", "extra", "all"],
                    data=partial_f1_tab,
                )
            }
        )

    except Exception as e:
        logger.exception(f"{type(e).__name__} - Spider Eval Suite Error")
        raise


def purge_vram():
    """Completely clears the GPU memory between"""

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


# ====================================================
#                Loading functions
# ====================================================


@retry_hub_io()
def load_model_with_retry(model_id, **kwargs):
    return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)


@retry_hub_io()
def load_dataset_with_retry(
    dataset_id, dry_run_samples: int = 2, dry_run: bool = False, split="test"
):
    logger.info(f"Loading dataset: {dataset_id}")
    eval_dataset = load_dataset(dataset_id, split=split)

    if dry_run and dry_run_samples is not None:
        n = min(dry_run_samples, len(eval_dataset))
        logger.warning(f"DRY RUN: truncating dataset to {n} samples")
        eval_dataset = eval_dataset.select(range(n))

    logger.info(f"Prepared {len(eval_dataset)} samples")
    return eval_dataset


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
def log_artifact_with_retry(run, artifact):
    return run.log_artifact(artifact)


def _artifact_is_missing(error, artifact_path):
    """Recognize only the pinned SDKs' missing-artifact results, not failed requests.

    W&B 0.24/0.26/0.28 raise these ValueErrors after receiving an existing
    project with no matching artifact/membership. The public API may wrap the
    ValueError in CommError.exc. Unknown formats deliberately stop the upload.
    """
    parts = artifact_path.split("/")
    if len(parts) != 3 or not all(parts):
        return False
    entity, project, name = parts
    project_path = f"{entity}/{project}"
    cause = getattr(error, "exc", None) or error
    return type(cause) is ValueError and str(cause) in {
        f"artifact {name!r} not found in {project_path!r}",
        f"artifact membership {name!r} not found in {project_path!r}",
    }


def artifact_exists_strict(api, artifact_path):
    """Return an existence boolean without hiding failed or ambiguous lookups."""
    try:
        api.artifact(artifact_path)
    except (wandb.errors.CommError, ValueError) as error:
        if _artifact_is_missing(error, artifact_path):
            return False
        raise
    return True


def _retryable_artifact_error(error):
    """Retry known transport failures and HTTP 408/429/5xx, never access errors."""
    cause = getattr(error, "exc", None) or error
    response = getattr(cause, "response", None)
    # requests uses status_code; W&B's service API uses http_status.
    status = getattr(response, "status_code", None) or getattr(
        response, "http_status", None
    )
    if isinstance(status, int):
        return status in (408, 429) or 500 <= status < 600
    # An SSL configuration/certificate failure is not a transient connection error.
    return isinstance(
        cause, (requests.Timeout, requests.ConnectionError)
    ) and not isinstance(cause, requests.exceptions.SSLError)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=2, max=30),
    retry=retry_if_exception(_retryable_artifact_error),
    reraise=True,
)
def upload_dataset_with_retry(cfg, dataset, evaluation_run):
    """Upload a missing dataset using staging owned solely by this attempt."""
    try:
        api = wandb.Api()
        logger.info("Checking for artifact: %s ...", cfg.wandb.ds_artifact_path)

        if artifact_exists_strict(api, cfg.wandb.ds_artifact_path):
            logger.info(
                "Artifact '%s' already exists. Skipping upload.",
                cfg.wandb.ds_artifact_name,
            )
            return

        # TemporaryDirectory allocates a fresh child and cleans up only that child
        run_path = evaluation_run.path.resolve(strict=True)
        with TemporaryDirectory(
            prefix=".dataset-upload-", dir=run_path
        ) as temporary_dir:
            save_path = Path(temporary_dir)
            logger.info("Saving processed dataset in %s", save_path)
            dataset.save_to_disk(str(save_path / "test"))

            # Keep the staging directory alive through upload AND run shutdown.
            with wandb.init(
                entity=cfg.wandb.entity,
                project=cfg.wandb.project_name,
                name="data-upload",
                job_type="data_prep",
            ) as run:
                new_artifact = wandb.Artifact(
                    name=cfg.wandb.ds_artifact_name,
                    type="dataset",
                    description=f"Spider dataset with 'Context' {cfg.dataset.id}",
                )
                new_artifact.add_dir(str(save_path))
                logger.info("Uploading artifact to W&B...")
                # The outer retry owns the complete attempt, including wait().
                logged = run.log_artifact(
                    new_artifact, aliases=[cfg.wandb.ds_artifact_alias]
                )
                logged.wait()
            logger.info("Upload complete!")
    except Exception:
        logger.exception("Dataset artifact lookup/upload failed")
        raise


# ====================================================
#              Custom context managers
# ====================================================


@contextmanager
def managed_model(model_id, **kwargs):
    model = tokenizer = None
    try:
        model = load_model_with_retry(model_id, **kwargs)

        tokenizer = load_with_retry(
            AutoTokenizer.from_pretrained, model_id, fix_mistral_regex=True
        )
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.unk_token

        yield model, tokenizer
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        purge_vram()


# ====================================================
#                Main evaluation function
# ====================================================


def evaluate_model(model_id, run_name, eval_dataset, cfg, evaluation_run, model_key):

    table_columns = [
        "Context",
        "Question",
        "True SQL",
        "Generated SQL",
    ]

    logger.info(f"Starting Evaluation for: {model_id}")

    with wandb.init(
        project=cfg.wandb.project_name,
        name=run_name,
        config={"model": model_id},
    ) as run:
        tracker = SystemResourceTracker(
            run=run, interval=1.0,
            output_dir=resource_output_dir(
                "evaluation", evaluation_run.path.name, model_key,
                root=cfg.system.get("resource_output_root", "artifacts/resources"),
            ),
        )
        try:
            tracker.start()
            with managed_model(
                model_id,
                device_map=cfg.model.device_map,
                dtype="auto",
                low_cpu_mem_usage=True,
            ) as (model, tokenizer):
                model.eval()

                ppl_sample = eval_dataset.select(range(min(200, len(eval_dataset))))
                perplexity = calculate_perplexity(model, tokenizer, ppl_sample)
                wandb.log({"eval/perplexity": perplexity})
                logger.info(f"Perplexity for {model_id}: {perplexity:.2f}")

                records = []
                table_data = []

                # Use DataLoader for batching
                dataloader = DataLoader(
                    eval_dataset,
                    batch_size=cfg.training.batch_size,
                    shuffle=False,
                    collate_fn=lambda samples: samples,
                )

                # Generate Predictions
                for batch in tqdm(dataloader, desc="Generating SQL"):
                    batch_manifest = [
                        {
                            "sample_id": sample["_evaluation_sample_id"],
                            "db_id": sample["db_id"],
                        }
                        for sample in batch
                    ]

                    def generate(batch=batch):
                        prompts = [
                            f"Context: {sample['context']}\nQuestion: {sample['question']}\nAnswer:"
                            for sample in batch
                        ]
                        inputs = tokenizer(
                            prompts, return_tensors="pt", padding=True
                        ).to("cuda")
                        with torch.no_grad():
                            generated_ids = model.generate(
                                **inputs,
                                max_new_tokens=cfg.training.max_new_tokens,
                                pad_token_id=tokenizer.pad_token_id,
                                use_cache=True,
                                do_sample=False,
                            )
                        input_len = inputs.input_ids.shape[-1]
                        return tokenizer.batch_decode(
                            generated_ids[:, input_len:], skip_special_tokens=True
                        )

                    batch_records = generation_batch(
                        batch_manifest,
                        generate,
                        oom_exception=torch.cuda.OutOfMemoryError,
                    )
                    records.extend(batch_records)
                    table_data.extend(
                        [
                            [
                                sample["context"],
                                sample["question"],
                                sample["query"],
                                record["raw_output"],
                            ]
                            for sample, record in zip(batch, batch_records)
                        ]
                    )
                    for record in batch_records:
                        if record["status"] not in {"ok", "empty"}:
                            logger.error(
                                "Generation failed for %s: %s",
                                record["sample_id"],
                                record["error"],
                            )
                    if any(record["status"] == "oom" for record in batch_records):
                        torch.cuda.empty_cache()

                wandb.log(
                    {
                        f"predictions_{run_name}": wandb.Table(
                            columns=table_columns, data=table_data
                        )
                    }
                )

                local_artifacts = export_predictions(
                    evaluation_run,
                    records,
                    model_key=model_key,
                    model_id=model_id,
                )

                artifact = wandb.Artifact(
                    name=f"predictions_{run_name}", type="predictions"
                )

                artifact.add_dir(str(local_artifacts.path))
                artifact.add_file(str(evaluation_run.path / "manifest.jsonl"))
                artifact.add_file(str(evaluation_run.path / "dataset.json"))
                artifact.add_file(str(local_artifacts.gold_path))

                log_artifact_with_retry(run, artifact)
                run_spider_eval_suite(local_artifacts, run_name, cfg)

        except torch.cuda.OutOfMemoryError as e:
            logger.exception(f"{type(e).__name__} - OOM evaluating model {model_id}")
            raise

        except Exception as e:
            logger.exception(
                f"{type(e).__name__} - Unexpected error evaluating model {model_id}"
            )
            raise
        finally:
            try:
                try:
                    purge_vram()
                finally:
                    tracker.stop()
            except Exception as e:
                logger.exception(
                    f"{type(e).__name__} - Error occurred while purging VRAM or stopping tracker"
                )


# ====================================================
#                       Main
# ====================================================


@hydra.main(
    version_base=None, config_path="pkg://configs", config_name="final_eval/config"
)
def main(cfg: DictConfig) -> None:
    """Main entry point for the evaluation script."""

    log_filename = f"{Path(__file__).stem}.log"
    setup_logging(filename=log_filename)

    # Forbid accidental typo'd keys anywhere in the resolved config.
    OmegaConf.set_struct(cfg, True)

    # Check for env vars and paths
    check_secrets(cfg.system.required_env_vars)
    check_paths(cfg)

    wandb.login()

    # Load dataset and process it
    try:
        eval_dataset = load_dataset_with_retry(
            cfg.dataset.id,
            split="test",
            dry_run=cfg.dry_run,
            dry_run_samples=cfg.dry_run_samples,
        )

        logger.info(f"Successfully loaded {len(eval_dataset)} test samples.")

    except Exception as e:
        logger.critical(
            f"Fatal Data Loading/Mapping Error ({type(e).__name__}): {e}", exc_info=True
        )
        raise

    evaluation_run = create_evaluation_run(
        eval_dataset,
        output_dir=cfg.eval_paths.artifacts_output_path,
        dataset_id=cfg.dataset.id,
        split="test",
    )
    # Allocate the run before upload so staging is isolated under its directory.
    upload_dataset_with_retry(cfg, eval_dataset, evaluation_run)
    eval_dataset = attach_sample_ids(eval_dataset, evaluation_run)
    logger.info("Evaluation artifacts in: %s", evaluation_run.path)

    for model_key, model_run in cfg.runs.items():
        evaluate_model(
            model_id=model_run.model_id,
            run_name=model_run.run_name,
            eval_dataset=eval_dataset,
            cfg=cfg,
            evaluation_run=evaluation_run,
            model_key=model_key,
        )

    logger.info("Evaluations completed successfully.")


if __name__ == "__main__":
    sys.exit(main())
