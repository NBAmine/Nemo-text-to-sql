from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from configs.common_schema import CommonConfig


@dataclass
class ModelConfig:
    device_map: str = "cuda:0"


@dataclass
class DatasetConfig:
    id: str = MISSING


@dataclass
class WandbConfig:
    project_name: str = MISSING
    entity: str = MISSING
    ds_artifact_name: str = MISSING
    ds_artifact_alias: str = MISSING
    ds_artifact_path: str = MISSING


@dataclass
class TrainingConfig:
    batch_size: int = MISSING
    max_new_tokens: int = MISSING

    #   ConfigError(f"max_new_tokens must be >= 1, got {self.max_new_tokens}")


@dataclass
class EvalPathsConfig:
    test_db_path: str = MISSING
    test_table_path: str = MISSING
    artifacts_output_path: str = "artifacts/evaluation"
    etype: str = MISSING
    strict_parsing: bool = True


@dataclass
class RunConfig:
    model_id: str = MISSING
    run_name: str = MISSING


@dataclass
class HubConfig:
    user: str = MISSING
    finetuned_model_id: str = MISSING


@dataclass
class Config(CommonConfig):
    model: ModelConfig = field(default_factory=ModelConfig)
    hub: HubConfig = field(default_factory=HubConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval_paths: EvalPathsConfig = field(default_factory=EvalPathsConfig)
    runs: dict[str, RunConfig] = field(default_factory=dict)
    hydra: Any = field(default_factory=dict)
    dry_run: bool = False
    dry_run_samples: int = 2


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=Config)


register_configs()
