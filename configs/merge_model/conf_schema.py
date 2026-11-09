"""Structured configuration for the merge command."""

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from configs.common_schema import CommonConfig


@dataclass
class ModelConfig:
    base_id: str = MISSING
    dtype: str = "auto"
    device_map: str = "cuda:0"


@dataclass
class HubConfig:
    adapters_repo_id: str = MISSING
    repo_id: str = MISSING
    finetuned_model_id: str = MISSING
    user: str = MISSING
    private: bool = True


@dataclass
class PathsConfig:
    # Retain the configuration key for existing CLI and pipeline overrides.
    backup_output_path: str = "artifacts/merged_model"


@dataclass
class SystemConfig:
    required_env_vars: list[str] = field(default_factory=lambda: ["HF_TOKEN"])


@dataclass
class MergeConfig(CommonConfig):
    model: ModelConfig = field(default_factory=ModelConfig)
    hub: HubConfig = field(default_factory=HubConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    system: SystemConfig = field(default_factory=SystemConfig)


def register_configs() -> None:
    ConfigStore.instance().store(name="merge_schema", node=MergeConfig)
