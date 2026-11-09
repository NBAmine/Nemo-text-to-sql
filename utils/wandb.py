import hashlib

from omegaconf import DictConfig, OmegaConf


def wandb_mode(cfg: DictConfig) -> str:
    """Validate the configured tracking mode before training or login."""
    mode = cfg.wandb.get("mode", "online")
    if mode not in ("online", "offline", "disabled"):
        raise ValueError(
            f"Unknown wandb.mode={mode!r}; expected online, offline or disabled."
        )
    return mode


def training_required_secrets(cfg: DictConfig) -> list[str]:
    """Keep other secret requirements, excluding W&B's key when tracking locally."""
    mode = wandb_mode(cfg)
    return [
        name
        for name in cfg.system.required_env_vars
        if name != "WANDB_API_KEY" or mode == "online"
    ]


def checkpoint_run_identity(metadata: dict, *, resuming: bool) -> dict:
    """Reuse only a checkpoint's tracking identity; reject fresh-run collisions."""
    return {"id": metadata["wandb_run_id"], "resume": "allow" if resuming else "never"}


def make_run_id(run_name: str, cfg: DictConfig) -> str:
    cfg_str = OmegaConf.to_yaml(cfg, resolve=True)
    cfg_hash = hashlib.sha1(cfg_str.encode()).hexdigest()[:8]
    return f"{run_name}-{cfg_hash}"
