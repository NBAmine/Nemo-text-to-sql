from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING


@dataclass
class SqlExecutionConfig:
    # Supplied by configs/base.yaml; a schema default would overwrite that
    # setting when final evaluation composes its schema after the base values.
    wait_time: float = MISSING


@dataclass
class CommonConfig:
    """Minimal base that every Hydra config inherits.

    ``hydra`` absorbs Hydra's own job/runtime keys so they don't trip
    OmegaConf's struct-mode checks.
    """

    hydra: Any = field(default_factory=dict)
    sql_execution: SqlExecutionConfig = field(default_factory=SqlExecutionConfig)
