"""Merge LoRA adapters and publish the model using Hydra configuration."""

import logging
import sys
from pathlib import Path

import hydra
from huggingface_hub.utils import validate_repo_id
from omegaconf import DictConfig, OmegaConf

from configs.merge_model.conf_schema import register_configs
from utils.logging import setup_logging

logger = logging.getLogger(__name__)

register_configs()


def validate_config(cfg: DictConfig) -> None:
    """Resolve required settings and reject invalid values before model loading."""

    OmegaConf.set_struct(cfg, True)
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    for repo_id in (cfg.model.base_id, cfg.hub.repo_id):
        validate_repo_id(repo_id)
    adapter_path = Path(cfg.hub.adapters_repo_id)
    if adapter_path.is_dir():
        if not (adapter_path / "adapter_config.json").is_file():
            raise ValueError(
                f"Local adapter has no adapter_config.json: {adapter_path}"
            )
    else:
        validate_repo_id(cfg.hub.adapters_repo_id)
    if cfg.model.dtype not in {"auto", "float16", "bfloat16", "float32"}:
        raise ValueError("model.dtype must be auto, float16, bfloat16, or float32")
    if not cfg.model.device_map.strip():
        raise ValueError("model.device_map must not be empty")
    if not cfg.paths.backup_output_path.strip():
        raise ValueError("paths.backup_output_path must not be empty")


def merge_and_publish(cfg: DictConfig) -> None:

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from utils.hub import create_model_repo, load_with_retry, upload_folder_with_retry
    from utils.system import check_secrets

    check_secrets(cfg.system.required_env_vars)

    try:
        logger.info("Loading tokenizer from %s...", cfg.hub.adapters_repo_id)
        tokenizer = load_with_retry(
            AutoTokenizer.from_pretrained,
            cfg.hub.adapters_repo_id,
            # revision="<adapter-commit-sha>",  # Match the adapter revision below.
            trust_remote_code=False,
            fix_mistral_regex=True,
        )

        logger.info("Loading base model ...")
        base_model = load_with_retry(
            AutoModelForCausalLM.from_pretrained,
            cfg.model.base_id,
            # revision="<base-model-commit-sha>",  # Optional: pin a reviewed full commit SHA.
            dtype=cfg.model.dtype,
            device_map=cfg.model.device_map,
            trust_remote_code=False,
        )

        logger.info("Loading adapters from %s...", cfg.hub.adapters_repo_id)
        model = load_with_retry(
            PeftModel.from_pretrained,
            base_model,
            cfg.hub.adapters_repo_id,
        )

        logger.info("Merging weights...")
        merged_model = model.merge_and_unload()
    except torch.cuda.OutOfMemoryError:
        logger.error("Error: GPU Out of Memory during merge.")
        sys.exit(1)
    except Exception as e:
        logger.error("Failed to load or merge model: %s", e)
        sys.exit(1)

    output_path = cfg.paths.backup_output_path
    try:
        logger.info("Saving merged model and tokenizer to %s...", output_path)
        merged_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
    except Exception as e:
        logger.error(
            "Failed to save merged model and tokenizer to %s: %s", output_path, e
        )
        sys.exit(1)

    try:
        logger.info(
            "Uploading saved model from %s to %s...", output_path, cfg.hub.repo_id
        )
        create_model_repo(cfg.hub.repo_id, private=cfg.hub.private)
        upload_folder_with_retry(output_path, cfg.hub.repo_id)

        logger.info("Success! The merged model is now live on Hugging Face.")
    except Exception as e:
        logger.error(
            "Failed to push to Hugging Face: %s. Local model and tokenizer remain at %s.",
            e,
            output_path,
        )
        sys.exit(1)


@hydra.main(
    version_base=None, config_path="pkg://configs", config_name="merge_model/config"
)
def main(cfg: DictConfig) -> None:

    # Configure logging
    log_filename = f"{Path(__file__).stem}.log"
    setup_logging(level=logging.INFO, filename=log_filename)

    validate_config(cfg)

    merge_and_publish(cfg)


if __name__ == "__main__":
    main()
