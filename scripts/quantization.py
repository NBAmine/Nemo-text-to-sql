import gc
import os

from utils.hub import create_model_repo, load_with_retry

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import logging
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from uuid import uuid4

import hydra
import torch
import torch.distributed as dist
import wandb
from compressed_tensors.offload import disable_onloading, init_dist
from datasets import load_dataset
from huggingface_hub import HfApi
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQMapping
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier
from llmcompressor.utils import load_context
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from configs.quantization.conf_schema import register_quantization_config
from utils.logging import setup_logging
from utils.resources import SystemResourceTracker, resource_output_dir
from utils.system import check_secrets

logger = logging.getLogger(__name__)

register_quantization_config()


def build_quantization_recipe(method: str) -> list:
    """Build the AWQ, GPTQ or FP8 recipe with CPU-backed AWQ caches."""
    method_upper = method.upper()

    if method_upper == "AWQ":
        logger.info("Configuring AWQ (W4A16_ASYM) Recipe...")
        mistral_nemo_mappings = [
            # 1. Smooth the Input Norm -> Balance Q, K, V
            AWQMapping(
                smooth_layer="re:.*input_layernorm$",
                balance_layers=["re:.*q_proj$", "re:.*k_proj$", "re:.*v_proj$"],
            ),
            # 2. Smooth the MLP Gate/Up -> Balance Down
            # We combine gate and up because they are parallel
            AWQMapping(
                smooth_layer="re:.*post_attention_layernorm$",
                balance_layers=["re:.*gate_proj$", "re:.*up_proj$"],
            ),
            # 3. Handle the Up -> Down projection specifically
            AWQMapping(
                smooth_layer="re:.*up_proj$",
                balance_layers=["re:.*down_proj$"],
            ),
        ]

        return [
            AWQModifier(
                duo_scaling="both",
                mappings=mistral_nemo_mappings,
                offload_device=torch.device("cpu"),
            ),
            QuantizationModifier(
                ignore=["lm_head"],
                scheme="W4A16_ASYM",
                targets=["Linear"],
            ),
        ]

    if method_upper == "GPTQ":
        logger.info("Configuring GPTQ (W4A16) Recipe...")
        return [GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])]

    elif method_upper == "FP8":
        logger.info("Configuring FP8 (W8A8 Dynamic/Static) Recipe...")
        return [
            QuantizationModifier(
                targets="Linear",
                scheme="FP8",
                ignore=["lm_head"],
            )
        ]

    else:
        raise ValueError(
            f"Unsupported quantization method: {method}. Choose 'AWQ', 'GPTQ' or 'FP8'."
        )


def broadcast_from_main(value):
    """Share small run metadata without involving monitoring threads."""
    if dist.is_initialized():
        values = [value]
        dist.broadcast_object_list(values, src=0)
        return values[0]
    return value


def partition_calibration_data(dataset, rank: int, world_size: int):
    """Assign every selected sample to exactly one worker, including remainders."""
    if len(dataset) < world_size:
        raise ValueError("Calibration needs at least one sample per GPU")
    if world_size == 1:
        return dataset
    start = len(dataset) * rank // world_size
    end = len(dataset) * (rank + 1) // world_size
    # Distinct select() fingerprints tell LLM Compressor these are already
    # partitioned datasets, so its identical-dataset sampler does not shard again.
    return dataset.select(range(start, end))


def run_quantization(cfg: DictConfig) -> list:
    distributed = dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    is_main = rank == 0

    log_filename = f"{Path(__file__).stem}.log"
    setup_logging(filename=log_filename, rank=rank if distributed else None)
    logger.info("Quantization worker %d/%d", rank, world_size)

    # Forbid accidental typo'd keys anywhere in the resolved config.
    OmegaConf.set_struct(cfg, True)

    # Check for env vars
    check_secrets(cfg.system.required_env_vars)

    # ===============================================================
    #                   --- Data Preparation ---
    # ===============================================================

    requested_samples = cfg.quantization.num_calibration_samples
    if requested_samples != "all" and (
        type(requested_samples) is not int or requested_samples <= 0
    ):
        raise ValueError(
            "quantization.num_calibration_samples must be a positive integer or 'all'"
        )

    try:
        logger.info(f"Loading dataset: {cfg.dataset.repo_id} for calibration...")
        ds_test = load_with_retry(
            load_dataset, cfg.dataset.repo_id, split=cfg.dataset.split
        )
        available_samples = len(ds_test)
        if not available_samples:
            raise ValueError("The calibration split is empty")
        sample_count = (
            available_samples
            if requested_samples == "all"
            else min(requested_samples, available_samples)
        )
        if sample_count < world_size:
            raise ValueError("Calibration needs at least one sample per GPU")
        logger.info(
            "Calibration selection: requested=%s, available=%d, selected=%d, seed=%d",
            requested_samples,
            available_samples,
            sample_count,
            cfg.system.seed,
        )
        ds_test = ds_test.shuffle(seed=cfg.system.seed).select(range(sample_count))
        ds_test = partition_calibration_data(ds_test, rank, world_size)

        def process_spider_test_set(sample):
            return {
                "text": (
                    f"Context: {sample['context']}\n"
                    f"Question: {sample['question']}\n"
                    f"Answer: {sample['query']}"
                )
            }

        calib_data = ds_test.map(
            process_spider_test_set, remove_columns=ds_test.column_names
        )

        logger.info(
            "Prepared %d calibration samples on rank %d (%d total across GPUs)",
            len(calib_data),
            rank,
            sample_count,
        )
        logger.info("-" * 30)
        logger.info("Example Sample:")
        logger.info(calib_data[0])
        logger.info("-" * 30)

    except Exception as e:
        logger.exception(f"{type(e).__name__} - Dataset processing failed")
        raise

    # ===============================================================
    #                   --- Quantization Process ---
    # ===============================================================

    summary_rows = []

    for method in cfg.quantization.methods:
        logger.info("=" * 60)
        logger.info(f"STARTING PIPELINE FOR: {method}")
        logger.info("=" * 60)

        # Unique paths & repo IDs per quantization variant
        method_lower = method.lower()
        method_quant_path = f"{cfg.model.quant_path}/{method_lower}"
        hub = getattr(cfg, "hub")
        run_name = method_lower

        try:
            if is_main:
                wandb.login()

            run_context = (
                wandb.init(
                    project=cfg.wandb.project_name,
                    name=run_name,
                    config={
                        "model": hub.finetuned_model_id,
                        "quantization": method,
                        "max_seq_length": cfg.quantization.max_seq_length,
                        "num_calibration_samples": sample_count,
                        "world_size": world_size,
                    },
                )
                if is_main
                else nullcontext(None)
            )
            with run_context as run:
                run_id = broadcast_from_main(run.id if is_main else None)
                session_id = broadcast_from_main(uuid4().hex if is_main else None)
                logger.info(f"Loading {hub.finetuned_model_id} from Hugging Face...")

                # The distributed load context shares CPU-backed weights and
                # dispatches each worker's computation onto its own local GPU.
                with load_context() if distributed else nullcontext():
                    model = load_with_retry(
                        AutoModelForCausalLM.from_pretrained,
                        f"{hub.user}/{hub.finetuned_model_id}",
                        device_map="auto_offload"
                        if distributed
                        else cfg.model.device_map,
                        # A100 supports BF16. Avoid inheriting FP32 from a checkpoint.
                        dtype=torch.bfloat16,
                        attn_implementation="sdpa",
                    )
                model.eval()
                original_use_cache = model.config.use_cache
                model.config.use_cache = False

                tokenizer = load_with_retry(
                    AutoTokenizer.from_pretrained,
                    f"{hub.user}/{hub.finetuned_model_id}",
                    fix_mistral_regex=True,
                )

                # size before quantization
                with disable_onloading():
                    original_size_gb = (
                        sum(p.numel() * p.element_size() for p in model.parameters())
                        / 1e9
                    )

                logger.info("Starting Quantization...")

                recipe = build_quantization_recipe(method)

                resource_tracker = SystemResourceTracker(
                    run=run,
                    interval=cfg.system.monitor_interval,
                    output_dir=resource_output_dir(
                        "quantization",
                        run_id,
                        method_lower,
                        root=cfg.system.get(
                            "resource_output_root", "artifacts/resources"
                        ),
                    ),
                    session_id=session_id,
                    device_indices=[torch.cuda.current_device()]
                    if distributed
                    else None,
                )
                try:
                    resource_tracker.start()
                    t0 = time.time()

                    with torch.no_grad():
                        oneshot(
                            model=model,
                            dataset=calib_data,
                            recipe=recipe,
                            max_seq_length=cfg.quantization.max_seq_length,
                            num_calibration_samples=len(calib_data),
                            batch_size=1,
                            # Preserve complete decoder blocks for AWQ mappings.
                            sequential_targets=["MistralDecoderLayer"],
                            sequential_offload_device="cpu",
                        )

                    quant_duration_s = time.time() - t0
                finally:
                    # Stop before leaving wandb.init(), including on CUDA OOM.
                    resource_tracker.stop()
                    model.config.use_cache = original_use_cache

                logger.info(f"Saving {method} model to {method_quant_path}...")

                model.save_pretrained(method_quant_path, save_compressed=True)
                # Model saving is collective: every rank must call it. Only
                # rank 0 writes model files; guard the ordinary tokenizer save.
                if is_main:
                    tokenizer.save_pretrained(method_quant_path)
                if distributed:
                    dist.barrier()
                if not is_main:
                    continue

                quantized_size_gb = (
                    sum(
                        f.stat().st_size
                        for f in Path(method_quant_path).rglob("*")
                        if f.is_file()
                    )
                    / 1e9
                )

                compression_ratio = original_size_gb / quantized_size_gb

                summary_rows.append(
                    [
                        method,
                        quant_duration_s,
                        original_size_gb,
                        quantized_size_gb,
                        compression_ratio,
                    ]
                )

        except torch.cuda.OutOfMemoryError as e:
            logger.exception(
                f"{type(e).__name__} - CUDA OOM during {method} quantization. "
                "Try reducing quantization.num_calibration_samples or "
                "quantization.max_seq_length."
            )
            raise
        except Exception as e:
            logger.exception(f"{type(e).__name__} - Failed during {method} execution")
            raise

        finally:
            # Clean VRAM & RAM before next method iteration
            logger.info("Releasing memory and clearing CUDA cache...")

            if "model" in locals():
                del model
            if "tokenizer" in locals():
                del tokenizer

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return summary_rows


def upload_quantized_model(method, method_quant_path, method_hf_repo, hub):
    """Publish once, after workers have finished all collective operations."""
    try:
        logger.info(f"Uploading {method} artifact to Hugging Face: {method_hf_repo}")
        api = HfApi()
        create_model_repo(method_hf_repo, private=hub.private, api=api)
        api.upload_folder(
            folder_path=method_quant_path,
            repo_id=method_hf_repo,
            repo_type="model",
            commit_message=f"Upload {method} quantized model",
        )
        logger.info(f"Successfully uploaded {method} to {method_hf_repo}")
    except Exception:
        logger.exception("Failed to upload %s", method)
        raise


@hydra.main(
    version_base=None, config_path="pkg://configs", config_name="quantization/config"
)
def main(cfg: DictConfig) -> None:
    try:
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            init_dist()
        is_main = not dist.is_initialized() or dist.get_rank() == 0
        summary_rows = run_quantization(cfg)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

    if not is_main:
        return
    # Uploading can take longer than the process-group timeout. Finish DDP
    # first so other workers do not wait in NCCL while rank 0 uses the network.
    for method in cfg.quantization.methods:
        upload_quantized_model(
            method,
            f"{cfg.model.quant_path}/{method.lower()}",
            f"{cfg.hub.user}/{cfg.hub.finetuned_model_id}-{method.lower()}",
            cfg.hub,
        )

    with wandb.init(
        project=cfg.wandb.project_name, name="quantization_summary"
    ) as summary_run:
        table = wandb.Table(
            columns=[
                "method",
                "duration_seconds",
                "original_size_gb",
                "quantized_size_gb",
                "compression_ratio",
            ],
            data=summary_rows,
        )
        summary_run.log({"quantization_comparison": table})


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"{type(e).__name__} - Script execution failed")
        sys.exit(1)
