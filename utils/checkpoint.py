"""Shared resume policy, local validation and distributed coordination."""

from __future__ import annotations

import json
import logging
import re
import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from uuid import uuid4

import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from transformers.trainer_callback import ExportableState, TrainerCallback

from utils.hub import download_hub_checkpoint, publish_hub_completion

logger = logging.getLogger(__name__)
METADATA_CALLBACK = "CheckpointMetadataCallback"
COMPLETION_FILE = "training_completed.json"


class CheckpointError(RuntimeError):
    """Training cannot safely continue with the requested resume policy."""


@dataclass
class CheckpointResolution:
    path: str | None  # Existing checkpoint to load, or None for fresh training
    metadata: dict  # Attempt ID, W&B run ID, compatibility information
    output_dir: str  # Where this attempt should save its outputs
    global_step: int = 0
    completed: bool = False
    completion: dict | None = None
    monitoring_session_id: str = ""  # Fresh per launch; broadcast with existing coordination.


class CheckpointMetadataCallback(TrainerCallback, ExportableState):
    """Persist identity and compatibility data before Trainer uploads a checkpoint.

    Trainer serializes ExportableState callbacks into trainer_state.json during
    _save_checkpoint, before _push_from_checkpoint (and before on_save).
    """

    def __init__(self, metadata: dict):
        """Store metadata for Trainer checkpoint serialization.

        Args:
            metadata: Attempt identity and compatibility contract to persist.

        Returns:
            None.
        """
        self.metadata = metadata

    def state(self):
        """Provide the callback state that Trainer serializes.

        Args:
            None.

        Returns:
            dict: Constructor arguments and attributes used to restore the callback.
        """
        return {"args": {"metadata": self.metadata}, "attributes": {}}


def checkpoint_contract(
    cfg: DictConfig, *, phase: str, world_size: int, train_dataset, eval_dataset
) -> dict:
    """Build compatibility settings, excluding output paths and resume policy.

    Args:
        cfg: Training configuration.
        phase: Training phase identifier.
        world_size: Number of distributed training workers.
        train_dataset: Processed training dataset with size and optional fingerprint.
        eval_dataset: Processed evaluation dataset with size and optional fingerprint.

    Returns:
        dict: Configuration, dataset and tracking properties required to match on resume.
    """
    sections = (
        "model",
        "peft",
        "dataset",
        "hyperparameters",
        "training",
        "early_stopping",
    )
    settings = {
        key: OmegaConf.to_container(cfg[key], resolve=True)
        for key in sections
        if key in cfg
    }
    return {
        "phase": phase,
        "world_size": world_size,
        "settings": settings,
        "seed": cfg.system.seed,
        "phase1_adapters_repo_id": cfg.hub.get("phase1_adapters_repo_id"),
        "wandb_project": cfg.wandb.project_name,
        "wandb_entity": cfg.wandb.entity,
        "datasets": {
            name: {
                "rows": len(dataset),
                "fingerprint": getattr(dataset, "_fingerprint", None),
            }
            for name, dataset in (("train", train_dataset), ("eval", eval_dataset))
        },
    }


def _latest_checkpoint(directory: Path) -> str | None:
    """Select a checkpoint entry by its highest numeric suffix without validating it.

    Args:
        directory: Directory whose immediate entries are inspected.

    Returns:
        str | None: Highest-numbered checkpoint path, or None if no name matches.
    """
    candidates = [
        (int(match.group(1)), path)
        for path in directory.iterdir()
        if (match := re.fullmatch(r"checkpoint-(\d+)", path.name))
    ]
    return str(max(candidates, key=lambda item: item[0])[1]) if candidates else None


def _local_attempt_dir(output_dir: str) -> Path | None:
    """Find one local attempt containing checkpoints or a completion record.

    Args:
        output_dir: Output root or an explicitly selected attempt directory.

    Returns:
        Path | None: Unique matching attempt directory, or None if none exists.
    """
    root = Path(output_dir)
    if not root.exists():
        return None
    if not root.is_dir():
        raise CheckpointError(f"Output path is not a directory: {root}")

    candidates = [
        directory
        for directory in [root, *sorted(root.glob("attempt-*"))]
        if directory.is_dir()
        and (
            _latest_checkpoint(directory) is not None
            or (directory / COMPLETION_FILE).is_file()
        )
    ]
    if len(candidates) > 1:
        raise CheckpointError(
            f"Multiple training attempts in '{root}'; set paths.adapter_output_path to the intended attempt directory."
        )
    return candidates[0] if candidates else None


def get_local_checkpoint(output_dir: str) -> str | None:
    """Find the latest numeric checkpoint without skipping a broken latest entry.

    Args:
        output_dir: Output root or an explicitly selected attempt directory.

    Returns:
        str | None: Latest checkpoint path, or None if no numeric checkpoint exists.
    """
    directory = _local_attempt_dir(output_dir)
    return _latest_checkpoint(directory) if directory is not None else None


def _read_json(path: Path) -> dict:
    """Read a JSON object and report file or parsing failures as checkpoint errors.

    Args:
        path: UTF-8 JSON file to read.

    Returns:
        dict: Parsed JSON object.
    """
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value
    except (OSError, ValueError) as error:
        raise CheckpointError(
            f"Cannot read checkpoint JSON '{path}': {error}"
        ) from error


def _validate_metadata(
    metadata: dict, contract: dict, path: str, *,
    allow_world_size_change: bool = False,
) -> dict:
    """Validate identity and optionally normalize a compatible DDP resize.

    Args:
        metadata: Saved metadata to validate.
        contract: Compatibility properties expected by the current run.
        path: Source path included in validation errors.
        allow_world_size_change: Accept more or fewer DDP workers with matching effective batch.

    Returns:
        dict: Validated metadata with the current contract after an accepted resize.
    """
    try:
        if metadata["version"] != 1:
            raise ValueError("unsupported metadata version")
        if not all(
            isinstance(metadata[key], str)
            and re.fullmatch(r"[A-Za-z0-9_-]+", metadata[key])
            for key in ("attempt_id", "wandb_run_id")
        ):
            raise ValueError("missing attempt identity")
        saved_contract = metadata["contract"]
        if not isinstance(saved_contract, dict):
            raise ValueError("invalid compatibility contract")
    except (KeyError, TypeError, ValueError) as error:
        raise CheckpointError(
            f"Checkpoint '{path}' has no supported attempt metadata. Legacy checkpoints "
            "cannot be verified automatically; use resume.mode=fresh for a new attempt."
        ) from error
    if saved_contract != contract and allow_world_size_change:
        saved_world = saved_contract.get("world_size")
        current_world = contract.get("world_size")
        # Single-process RNG files use a different format/name; this policy
        # handles changes between ordinary DDP process groups only.
        if (
            type(saved_world) is int and type(current_world) is int
            and min(saved_world, current_world) > 1
        ):
            comparable = deepcopy(saved_contract)
            try:
                saved_hp = comparable["settings"]["hyperparameters"]
                current_hp = contract["settings"]["hyperparameters"]
                batch_keys = ("per_device_train_batch_size", "gradient_accumulation_steps")
                values = [hp[key] for hp in (saved_hp, current_hp) for key in batch_keys]
                valid = all(type(value) is int and value > 0 for value in values)
                saved_batch, saved_accum, current_batch, current_accum = values
                if not valid or (
                    saved_world * saved_batch * saved_accum
                    != current_world * current_batch * current_accum
                ):
                    raise CheckpointError(
                        "Changing world_size requires the same effective training batch size "
                        "(world_size * per_device_train_batch_size * gradient_accumulation_steps). "
                        "Set new gradient_accumulation_steps = old accumulation * old world_size / "
                        "new world_size; the result must be a positive integer."
                    )
                # Only accumulation may compensate for a worker-count change. All other
                # training settings, including the per-device batch, still match.
                comparable["world_size"] = current_world
                saved_hp["gradient_accumulation_steps"] = current_accum
            except (KeyError, TypeError, ValueError) as error:
                raise CheckpointError("Cannot verify the effective batch size for world_size change.") from error
            if comparable == contract:
                # Normalize returned metadata for future saves and local/Hub
                # comparisons, leaving the checkpoint on disk untouched.
                return {**metadata, "contract": deepcopy(contract)}

    if saved_contract != contract:
        differences = [
            key for key in contract if saved_contract.get(key) != contract[key]
        ]
        raise CheckpointError(
            f"Incompatible checkpoint '{path}': {', '.join(differences)} changed."
            + (" To resize DDP, set resume.allow_world_size_change=true "
               "and preserve the effective training batch size (at least 2 workers)."
               if "world_size" in differences else "")
        )
    return metadata


def validate_checkpoint(
    path: str, contract: dict, *,
    allow_world_size_change: bool = False,
) -> dict:
    """Validate checkpoint files and identity without loading binary training state.

    Args:
        path: Checkpoint directory to inspect.
        contract: Expected configuration, dataset and worker compatibility properties.

    Returns:
        dict: Validated attempt metadata extracted from trainer_state.json.
    """
    root = Path(path)
    required = ["trainer_state.json", "optimizer.pt", "scheduler.pt"]
    # The supported scripts use LoRA and ordinary DDP, not DeepSpeed/FSDP.
    weight_names = (
        "adapter_model.safetensors",
        "adapter_model.bin",
        "model.safetensors",
        "pytorch_model.bin",
    )
    missing = [
        name
        for name in required
        if not (root / name).is_file() or (root / name).stat().st_size == 0
    ]

    weights = next(
        (
            name
            for name in weight_names
            if (root / name).is_file() and (root / name).stat().st_size > 0
        ),
        None,
    )
    if weights is None:
        missing.append("model/adapter weights")
    if missing:
        raise CheckpointError(
            f"Incomplete checkpoint '{path}': missing {', '.join(missing)}"
        )
    if weights.startswith("adapter_"):
        _read_json(root / "adapter_config.json")
    state = _read_json(root / "trainer_state.json")
    step = state.get("global_step")
    if type(step) is not int or step < 0:
        raise CheckpointError(f"Invalid global_step in '{path}'.")
    try:
        metadata = state["stateful_callbacks"][METADATA_CALLBACK]["args"]["metadata"]
    except (KeyError, TypeError):
        metadata = None
    validated_metadata = _validate_metadata(
        metadata, contract, path,
        allow_world_size_change=allow_world_size_change,
    )
    # Validate the complete original checkpoint, including RNG state for ranks
    # retired by a reduction, rather than accepting a partially written save.
    world_size = metadata["contract"]["world_size"]
    if type(world_size) is not int or world_size < 1:
        raise CheckpointError(f"Invalid world_size in '{path}'.")
    rng = ([f"rng_state_{rank}.pth" for rank in range(world_size)]
           if world_size > 1 else ["rng_state.pth"])
    missing = [name for name in rng
               if not (root / name).is_file() or (root / name).stat().st_size == 0]
    if missing:
        raise CheckpointError(f"Incomplete checkpoint '{path}': missing {', '.join(missing)}")
    # FP16 uses GradScaler; BF16 does not.
    if (
        contract["settings"]["training"].get("fp16")
        and not (root / "scaler.pt").is_file()
    ):
        raise CheckpointError(f"Incomplete FP16 checkpoint '{path}': missing scaler.pt")
    logger.info("Validated checkpoint %s at global step %s", path, step)
    return validated_metadata


def _read_completion(
    path: Path, contract: dict, *,
    allow_world_size_change: bool = False,
) -> dict:
    """Read and validate a completed attempt's success record.

    Args:
        path: Completion JSON file to read.
        contract: Compatibility properties expected by the current run.

    Returns:
        dict: Validated completion record, including metadata, final step and metrics.
    """
    record = _read_json(path)
    if (
        record.get("version") != 1
        or record.get("status") != "completed"
        or type(record.get("global_step")) is not int
        or record["global_step"] < 0
        or not isinstance(record.get("metrics"), dict)
    ):
        raise CheckpointError(f"Invalid completion record '{path}'.")
    record["metadata"] = _validate_metadata(
        record.get("metadata"), contract, str(path),
        allow_world_size_change=allow_world_size_change,
    )
    return record


def _validate_export(directory: Path) -> None:
    """Check final weights, optional adapter configuration and saved training metrics.

    Args:
        directory: Attempt directory containing the final exported artifacts.

    Returns:
        None. Invalid or missing required artifacts raise CheckpointError.
    """
    names = (
        "adapter_model.safetensors",
        "adapter_model.bin",
        "model.safetensors",
        "pytorch_model.bin",
    )
    weights = next(
        (
            name
            for name in names
            if (directory / name).is_file() and (directory / name).stat().st_size > 0
        ),
        None,
    )
    if weights is None:
        raise CheckpointError(
            f"Completed attempt is missing exported weights: {directory}"
        )
    if weights.startswith("adapter_"):
        _read_json(directory / "adapter_config.json")
    _read_json(directory / "train_results.json")


def mark_training_completed(
    resolution: CheckpointResolution,
    state,
    metrics: dict,
    *,
    repo_id: str | None = None,
) -> None:
    """Record completion after successful training and final artifact saves.

    The local record is written atomically before optional Hub publication.
    It remains present if that publication fails.

    Args:
        resolution: Selected attempt identity and output directory.
        state: Trainer state providing global_step and max_steps.
        metrics: Final training metrics to include in the success record.
        repo_id: Optional Hub repository receiving the attempt-specific record.

    Returns:
        None.
    """
    directory = Path(resolution.output_dir)

    _validate_export(directory)

    record = {
        "version": 1,
        "status": "completed",
        "metadata": resolution.metadata,
        "global_step": state.global_step,
        "stop_reason": "max_steps_reached"
        if state.global_step >= state.max_steps
        else "stopped_early",
        "metrics": metrics,
    }

    step = record["global_step"]
    if type(step) is not int or step < 0:
        raise CheckpointError("Cannot record completion with an invalid training step.")

    path = directory / COMPLETION_FILE
    temporary = None

    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=directory,
            prefix=f".{COMPLETION_FILE}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary = Path(file.name)
            json.dump(record, file, indent=2)

        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    if repo_id:
        publish_hub_completion(
            repo_id, str(path), attempt_id=resolution.metadata["attempt_id"]
        )


def _candidate(
    path: str | None,
    contract: dict,
    output_dir: str,
    *,
    completion_path: Path | None = None,
    local: bool,
    allow_world_size_change: bool = False,
) -> CheckpointResolution:
    """Combine checkpoint validation and completion evidence into one candidate.

    Args:
        path: Checkpoint directory, or None for a completion-only local attempt.
        contract: Compatibility properties expected by the current run.
        output_dir: Attempt output directory; the caller adjusts Hub destinations.
        completion_path: Optional local success-record path; derived for Hub candidates.
        local: Whether the candidate comes from local attempt storage.

    Returns:
        CheckpointResolution: Validated identity, step and completion status.
    """
    if path is not None:
        metadata = validate_checkpoint(
            path, contract,
            allow_world_size_change=allow_world_size_change,
        )
        state = _read_json(Path(path) / "trainer_state.json")
        step = state["global_step"]
        maximum = state.get("max_steps", 0)
        max_steps_reached = type(maximum) is int and maximum > 0 and step >= maximum
        if not local:
            completion_path = (
                Path(path).parent
                / "attempt-completions"
                / f"{metadata['attempt_id']}.json"
            )
    else:
        metadata, step, max_steps_reached = None, 0, False

    record = None

    if completion_path is not None and completion_path.is_file():
        record = _read_completion(
            completion_path, contract,
            allow_world_size_change=allow_world_size_change,
        )

        if metadata is not None and record["metadata"] != metadata:
            raise CheckpointError(
                f"Checkpoint and completion record identities differ: {completion_path}"
            )

        if record["global_step"] < step:
            raise CheckpointError(
                f"Completion record predates the selected checkpoint: {completion_path}"
            )

        metadata, step = record["metadata"], record["global_step"]
        if local:
            _validate_export(Path(output_dir))

    if metadata is None:
        raise CheckpointError("No checkpoint or completion record was found.")

    return CheckpointResolution(
        path,
        metadata,
        output_dir,
        step,
        completed=record is not None or max_steps_reached,
        completion=record,
    )


def _checkpoint_files_match(source: Path, destination: Path) -> bool:
    """Compare checkpoint contents without accepting destination symlinks.

    Args:
        source: Original checkpoint directory; cached files may be symlinks.
        destination: Existing local checkpoint directory to compare.

    Returns:
        bool: Whether both directories contain the same regular-file contents.
    """
    if destination.is_symlink() or not destination.is_dir():
        return False
    source_files = {
        path.relative_to(source) for path in source.rglob("*") if path.is_file()
    }
    entries = list(destination.rglob("*"))
    if any(path.is_symlink() for path in entries):
        return False
    destination_files = {
        path.relative_to(destination) for path in entries if path.is_file()
    }
    if source_files != destination_files:
        return False
    for relative in source_files:
        first, second = source / relative, destination / relative
        if first.samefile(second) or first.stat().st_size != second.stat().st_size:
            return False
        with first.open("rb") as original, second.open("rb") as existing:
            while True:
                chunk = original.read(1024 * 1024)
                if chunk != existing.read(1024 * 1024):
                    return False
                if not chunk:
                    break
    return True


def _materialize_hub_checkpoint(
    resolution: CheckpointResolution, contract: dict, *,
    allow_world_size_change: bool = False,
) -> CheckpointResolution:
    """Publish a validated independent local copy of the selected Hub checkpoint.

    Args:
        resolution: Selected incomplete Hub candidate and its local output directory.
        contract: Compatibility properties used to validate the staged checkpoint.

    Returns:
        CheckpointResolution: Same candidate with its path set to the local copy.
    """
    if resolution.path is None or resolution.completed:
        raise CheckpointError(
            "Only an incomplete Hub checkpoint can be copied for resume."
        )
    source = Path(resolution.path).resolve(strict=True)
    output = Path(resolution.output_dir).resolve()
    if output.is_relative_to(source.parent):
        raise CheckpointError(
            "Checkpoint output must be outside the Hub snapshot cache."
        )
    destination = output / f"checkpoint-{resolution.global_step}"
    output.mkdir(parents=True, exist_ok=True)

    if destination.exists() or destination.is_symlink():
        if not _checkpoint_files_match(source, destination):
            raise CheckpointError(
                f"Refusing to overwrite conflicting checkpoint: {destination}"
            )
        validate_checkpoint(
            str(destination), contract,
            allow_world_size_change=allow_world_size_change,
        )
    else:
        # The staging directory is on the destination filesystem and is never
        # discoverable as checkpoint-N while the files are still being copied.
        with TemporaryDirectory(
            dir=output, prefix=f".checkpoint-{resolution.global_step}."
        ) as temporary:
            staging = Path(temporary) / "checkpoint"
            shutil.copytree(source, staging, symlinks=False)
            metadata = validate_checkpoint(
                str(staging), contract,
                allow_world_size_change=allow_world_size_change,
            )
            step = _read_json(staging / "trainer_state.json")["global_step"]
            if metadata != resolution.metadata or step != resolution.global_step:
                raise CheckpointError(
                    "Staged checkpoint identity or step changed during copying."
                )
            if destination.exists() or destination.is_symlink():
                raise CheckpointError(
                    f"Checkpoint destination appeared during copying: {destination}"
                )
            staging.rename(destination)

    resolution.path = str(destination)
    logger.info("Hub checkpoint is ready for local resume: %s", destination)
    return resolution


def _resolve(cfg: DictConfig, contract: dict) -> CheckpointResolution:
    """Apply mode/source policy and compare candidates only within the same attempt.

    Args:
        cfg: Configuration with resume policy, Hub repository and adapter output path.
        contract: Compatibility properties expected by the current run.

    Returns:
        CheckpointResolution: Selected existing state or a fresh attempt description.
    """

    mode = cfg.resume.mode
    if mode not in ("auto", "required", "fresh"):
        raise CheckpointError(
            f"Unknown resume.mode={mode!r}; expected auto, required or fresh."
        )

    source = cfg.resume.get("source", "auto")
    allow_change = cfg.resume.get("allow_world_size_change", False)
    if type(allow_change) is not bool:
        raise CheckpointError("resume.allow_world_size_change must be a boolean.")
    if source not in ("auto", "local", "hub"):
        raise CheckpointError(
            f"Unknown resume.source={source!r}; expected auto, local or hub."
        )

    if mode != "fresh":
        local_candidate = hub_candidate = None

        if source in ("auto", "local"):
            directory = _local_attempt_dir(cfg.paths.adapter_output_path)
            if directory is not None:
                local_candidate = _candidate(
                    _latest_checkpoint(directory),
                    contract,
                    str(directory),
                    completion_path=directory / COMPLETION_FILE,
                    local=True,
                    allow_world_size_change=allow_change,
                )

        if source in ("auto", "hub"):
            if source == "hub" and not cfg.hub.get("repo_id"):
                raise CheckpointError("resume.source=hub requires hub.repo_id.")
            if cfg.hub.get("repo_id"):
                path = download_hub_checkpoint(cfg.hub.repo_id)
                if path is not None:
                    hub_candidate = _candidate(
                        path, contract, cfg.paths.adapter_output_path, local=False,
                        allow_world_size_change=allow_change,
                    )
                    # Keep the existing local attempt directory when Hub state is newer.
                    same_local = (
                        local_candidate is not None
                        and local_candidate.metadata == hub_candidate.metadata
                    )
                    hub_candidate.output_dir = (
                        local_candidate.output_dir
                        if same_local
                        else str(
                            Path(cfg.paths.adapter_output_path)
                            / f"attempt-{hub_candidate.metadata['attempt_id']}"
                        )
                    )

        selected = None
        if local_candidate is not None and hub_candidate is not None:
            if local_candidate.metadata != hub_candidate.metadata:
                raise CheckpointError(
                    "Local and Hub checkpoints belong to different attempts or tracking identities. "
                    "Select resume.source=local or resume.source=hub explicitly."
                )
            # A completion record wins a tie; otherwise prefer local on equal steps.
            candidates = [local_candidate, hub_candidate]
            selected = max(
                candidates,
                key=lambda item: (item.global_step, item.completion is not None),
            )

        elif local_candidate is not None or hub_candidate is not None:
            selected = local_candidate or hub_candidate

        if selected is not None:
            if selected is hub_candidate and not selected.completed:
                selected = _materialize_hub_checkpoint(
                    selected, contract,
                    allow_world_size_change=allow_change,
                )
            if not selected.completed and selected.path is not None:
                saved = _read_json(Path(selected.path) / "trainer_state.json")
                saved_metadata = saved["stateful_callbacks"][METADATA_CALLBACK]["args"]["metadata"]
                saved_world = saved_metadata["contract"]["world_size"]
                if saved_world != contract["world_size"]:
                    logger.warning(
                        "Resuming DDP with %s -> %s workers at step %s; effective batch size "
                        "is preserved. Data grouping and random streams change; this is not "
                        "a bit-for-bit continuation.",
                        saved_world, contract["world_size"], selected.global_step,
                    )
            return selected

        if mode == "required":
            raise CheckpointError("resume.mode=required but no checkpoint was found.")

    # Isolate every new trajectory; never overwrite old checkpoints or model files.
    attempt_id = uuid4().hex
    output_dir = str(Path(cfg.paths.adapter_output_path) / f"attempt-{attempt_id}")
    metadata = {
        "version": 1,
        "attempt_id": attempt_id,
        "wandb_run_id": attempt_id,
        "contract": contract,
    }
    return CheckpointResolution(None, metadata, output_dir)


def resolve_checkpoint(
    cfg: DictConfig, *, phase: str, is_main_process: bool, train_dataset, eval_dataset
) -> CheckpointResolution:
    """Resolve on rank zero and broadcast success or failure to every worker.

    Output paths must be shared across ranks; only rank zero needs the Hub cache. Updates
    cfg.paths.adapter_output_path to the selected attempt directory.

    Args:
        cfg: Training configuration updated with the resolved output directory.
        phase: Training phase identifier used in compatibility checks.
        is_main_process: Whether this worker is global rank zero.
        train_dataset: Processed training dataset used to build the contract.
        eval_dataset: Processed evaluation dataset used to build the contract.

    Returns:
        CheckpointResolution: Shared checkpoint/fresh-attempt result and completion status.
    """

    distributed = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if distributed else 1
    payload = [None]

    if is_main_process:
        try:
            contract = checkpoint_contract(
                cfg,
                phase=phase,
                world_size=world_size,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
            )
            result = _resolve(cfg, contract)
            result.monitoring_session_id = uuid4().hex
            payload[0] = {"result": result, "error": None}
        except Exception as error:
            # Do not raise until workers have received the same failure.
            logger.exception("Checkpoint resolution failed")
            payload[0] = {"result": None, "error": f"{type(error).__name__}: {error}"}

    if distributed:
        dist.broadcast_object_list(payload, src=0)

    if payload[0] is None:
        raise CheckpointError("Checkpoint resolution requires the main process.")

    if payload[0]["error"]:
        raise CheckpointError(payload[0]["error"])

    result = payload[0]["result"]

    cfg.paths.adapter_output_path = result.output_dir

    logger.info(
        "Training attempt %s: checkpoint=%s, output_dir=%s, step=%s, completed=%s",
        result.metadata["attempt_id"],
        result.path,
        cfg.paths.adapter_output_path,
        result.global_step,
        result.completed,
    )

    return result
