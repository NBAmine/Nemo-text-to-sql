"""Persist the effective training arguments after Trainer initialization."""

import json
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile


logger = logging.getLogger(__name__)


def record_trainer_config(args, *, run=None) -> Path:
    """Save resolved Trainer arguments locally and in the active W&B config.

    Call only on rank zero, using ``trainer.args`` after construction so library
    defaults and normalizations are included. ``to_dict`` uses Transformers'
    serialization, including its masking of token credential fields.
    """
    effective = args.to_dict()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "trainer_args.json"
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output,
            prefix=".trainer_args-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(effective, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    if run is not None:
        run.config.update({"effective_trainer_args": effective}, allow_val_change=True)
    logger.info(
        "Saved effective Trainer arguments to %s (seed=%s, data_seed=%s, train_sampling_strategy=%s).",
        destination,
        effective["seed"],
        effective["data_seed"],
        effective.get("train_sampling_strategy"),
    )
    return destination
