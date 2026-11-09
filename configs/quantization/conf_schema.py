from dataclasses import dataclass, field
from typing import Any
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore

from configs.common_schema import CommonConfig


@dataclass
class ModelConfig:
    quant_path: str = MISSING
    device_map: str = "cuda:0"


@dataclass
class DatasetConfig:
    repo_id: str = MISSING
    split: str = MISSING


@dataclass
class HubConfig:
    user: str = MISSING
    finetuned_model_id: str = MISSING
    private: bool = True


@dataclass
class QuantizationConfig:
    max_seq_length: int = MISSING
    num_calibration_samples: int | str = 256  # Positive integer or "all".
    methods: list[str] = field(default_factory=list)


@dataclass
class WandbConfig:
    entity: str = MISSING
    project_name: str = MISSING


@dataclass
class SystemConfig:
    seed: int = MISSING
    required_env_vars: list[str] = field(default_factory=list)
    monitor_interval: int = 1
    resource_output_root: str = "artifacts/resources"


@dataclass
class Config(CommonConfig):
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    hub: HubConfig = field(default_factory=HubConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    hydra: Any = field(default_factory=dict)


def register_quantization_config() -> None:
    cs = ConfigStore.instance()
    cs.store(group="quantization", name="quantization_schema", node=Config)
