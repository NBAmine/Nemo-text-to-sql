from dataclasses import dataclass, field
from enum import Enum

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from configs.common_schema import CommonConfig

# ---------------------------------------------------------------------------
# Leaf / nested schemas
# ---------------------------------------------------------------------------


@dataclass
class QuantizationConfig:
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class ModelConfig:
    base_id: str = MISSING
    tokenizer_max_length: int = 2048
    attn_implementation: str = "sdpa"
    torch_dtype: str = "bf16"
    compute_dtype: str = "bf16"
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)


@dataclass
class PeftConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass
class DatasetFineTuneConfig:
    """Dataset config used by the fine-tuning scripts (phase 1)."""

    id: str = MISSING  # resolved HF Hub repo ID for the prepared subset
    source_id: str = "gretelai/synthetic_text_to_sql"
    seed: int = 42  # Trainer data_seed: training example order, independent of system.seed
    text_field: str = "text"


@dataclass
class HyperparametersConfig:
    epochs: int = MISSING
    learning_rate: float = MISSING
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 16
    optim: str = "paged_adamw_8bit"
    warmup_steps: int = 100
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    max_steps: int = -1  # -1 = disabled; overridden to small int by smoke_test=enabled


@dataclass
class EarlyStoppingConfig:
    enabled: bool = True
    patience: int = 3
    threshold: float = 0.0


class TrainSamplingStrategy(str, Enum):
    group_by_length = "group_by_length"
    random = "random"


@dataclass
class TrainingConfig:
    fp16: bool = False
    bf16: bool = True
    logging_steps: int = 10
    eval_strategy: str = "steps"
    eval_steps: int = 300
    save_strategy: str = "steps"
    save_steps: int = 300
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    gradient_checkpointing: bool = True
    train_sampling_strategy: TrainSamplingStrategy = TrainSamplingStrategy.group_by_length
    ddp_find_unused_parameters: bool = False


@dataclass
class SpiderConfig:
    """Paths required by SpiderEvaluationCallback."""

    db_path: str = MISSING
    table_path: str = MISSING
    output_dir: str = "artifacts/evaluation"
    etype: str = "all"
    evaluation_timeout_seconds: float = 86400  # CPU wait, independent of NCCL


@dataclass
class PathsConfig:
    adapter_output_path: str = MISSING
    smoke_adapter_output_path: str = MISSING


@dataclass
class ResumeConfig:
    mode: str = "auto"  # auto | required | fresh
    source: str = "auto"  # auto compares local + Hub within one attempt | local | hub
    allow_world_size_change: bool = False  # DDP up/down; preserve effective batch size


@dataclass
class SystemConfig:
    """Cross-cutting operational settings (seeding, monitoring, secrets)."""

    seed: int = MISSING
    monitor_interval: float = MISSING  # seconds between SystemResourceTracker polls
    resource_output_root: str = "artifacts/resources"
    monitor_buffer_size: int = 1024  # Recent samples in RAM; complete history stays on disk.
    monitor_shutdown_timeout: float = 5.0
    required_env_vars: list[str] = field(default_factory=list)


@dataclass
class Phase1HubNamespace:
    """Nested hub.phase1.* keys defined in config.yaml."""

    adapters_repo_id: str = MISSING
    dataset_repo_id: str = MISSING


@dataclass
class Phase1HubConfig:
    """hub.* keys for phase 1 training."""

    user: str = MISSING
    finetuned_model_id: str | None = None
    phase1: Phase1HubNamespace = field(default_factory=Phase1HubNamespace)
    # Per-run overrides (set in phase1.yaml body)
    repo_id: str = MISSING
    strategy: str = "checkpoint"
    private: bool = True
    push_to_hub: bool = True


@dataclass
class Phase2HubConfig:
    """hub.* keys for phase 2 training.

    Extends Phase1HubConfig with ``phase1_adapters_repo_id`` so the script
    knows where to load the pre-trained Phase 1 LoRA adapter from.
    """

    user: str = MISSING
    finetuned_model_id: str | None = None
    phase1: Phase1HubNamespace = field(default_factory=Phase1HubNamespace)
    repo_id: str = MISSING
    # Repo ID of the Phase 1 adapter to load before Phase 2 fine-tuning.
    phase1_adapters_repo_id: str = MISSING
    strategy: str = "checkpoint"
    private: bool = True
    push_to_hub: bool = True


@dataclass
class WandbFineTuneConfig:
    mode: str = "online"  # online | offline | disabled
    entity: str = MISSING
    project_name: str = MISSING
    run_name: str = MISSING
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    # Used by phase-2 / prepare-dataset paths; optional here
    artifact_name: str = ""
    artifact_alias: str = ""
    samples_num: int = MISSING


# ---------------------------------------------------------------------------
# Smoke-Test schema
# ---------------------------------------------------------------------------


@dataclass
class SmokeTestConfig:
    """Controls fast sanity-check mode (activated with ``smoke_test=enabled``)."""

    enabled: bool = False
    max_train_samples: int | None = None  # None -> use full dataset
    max_eval_samples: int | None = None  # None -> use full dataset
    max_steps: int = -1  # -1 -> run full epochs


# ---------------------------------------------------------------------------
# Top-level Phase configs
# ---------------------------------------------------------------------------


@dataclass
class Phase1Config(CommonConfig):
    """Complete schema for ``configs/fine_tune/phase1.yaml``.

    Inherits ``hydra: Any`` from CommonConfig so Hydra own runtime keys
    never fail struct-mode validation.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    peft: PeftConfig = field(default_factory=PeftConfig)
    dataset: DatasetFineTuneConfig = field(default_factory=DatasetFineTuneConfig)
    hyperparameters: HyperparametersConfig = field(
        default_factory=HyperparametersConfig
    )
    training: TrainingConfig = field(default_factory=TrainingConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    resume: ResumeConfig = field(default_factory=ResumeConfig)
    hub: Phase1HubConfig = field(default_factory=Phase1HubConfig)
    wandb: WandbFineTuneConfig = field(default_factory=WandbFineTuneConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    smoke_test: SmokeTestConfig = field(default_factory=SmokeTestConfig)


@dataclass
class Phase2Config(CommonConfig):
    """Shared training schema for Phase 2 with or without schema context.

    Phase 2 reuses the Phase 1 adapter and adds Spider evaluation paths.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetFineTuneConfig = field(default_factory=DatasetFineTuneConfig)
    spider: SpiderConfig = field(default_factory=SpiderConfig)
    hyperparameters: HyperparametersConfig = field(
        default_factory=HyperparametersConfig
    )
    training: TrainingConfig = field(default_factory=TrainingConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    resume: ResumeConfig = field(default_factory=ResumeConfig)
    hub: Phase2HubConfig = field(default_factory=Phase2HubConfig)
    wandb: WandbFineTuneConfig = field(default_factory=WandbFineTuneConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    smoke_test: SmokeTestConfig = field(default_factory=SmokeTestConfig)


@dataclass
class Phase2WCtxConfig(Phase2Config):
    """Complete schema for ``configs/fine_tune/phase2_w_ctx.yaml``.

    Mirrors Phase1Config but replaces Phase1HubConfig with Phase2HubConfig
    (which adds ``phase1_adapters_repo_id``) and adds SpiderConfig for the
    evaluation callback paths. No ``peft`` block: Phase 2 reuses the adapter
    already trained in Phase 1.
    """


@dataclass
class DatasetPrepConfig:
    """Dataset fields for ``configs/fine_tune/prepare_dataset.yaml``."""

    source_id: str = MISSING
    output_id: str = MISSING
    stratify_column: str = MISSING
    complexity_sampling_rates: dict[str, float] = field(default_factory=dict)


@dataclass
class PrepareDatasetConfig(CommonConfig):
    """Complete schema for ``scripts/prepare_dataset.py``."""

    dataset: DatasetPrepConfig = field(default_factory=DatasetPrepConfig)
    hub: Phase1HubConfig = field(default_factory=Phase1HubConfig)
    wandb: WandbFineTuneConfig = field(default_factory=WandbFineTuneConfig)
    system: SystemConfig = field(default_factory=SystemConfig)


# ---------------------------------------------------------------------------
# Hydra ConfigStore Registration  (call exactly one of these per entry-point)
# ---------------------------------------------------------------------------


def register_phase1() -> None:
    """Register schemas needed by ``scripts/text-to-sql-phase1.py``."""
    cs = ConfigStore.instance()
    cs.store(group="fine_tune", name="phase1_schema", node=Phase1Config)


def register_phase2_w_ctx() -> None:
    """Register schemas needed by ``scripts/text-to-sql-phase2-w-context.py``."""
    cs = ConfigStore.instance()
    cs.store(group="fine_tune", name="phase2_w_ctx_schema", node=Phase2WCtxConfig)


def register_phase2_wo_ctx() -> None:
    """Register schemas needed by ``scripts/text-to-sql-phase2-wo-context.py``."""
    cs = ConfigStore.instance()
    cs.store(group="fine_tune", name="phase2_wo_ctx_schema", node=Phase2Config)


def register_prepare_dataset() -> None:
    """Register schemas needed by ``scripts/prepare_dataset.py``."""
    cs = ConfigStore.instance()
    cs.store(
        group="fine_tune", name="prepare_dataset_schema", node=PrepareDatasetConfig
    )
