import os
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import timedelta
from math import isfinite

import torch
import torch.distributed as dist

DTYPE_MAP: Mapping[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def resolve_torch_dtype(dtype_str: str) -> torch.dtype:
    """Converts a string representation into a validated torch.dtype."""
    normalized = dtype_str.strip().lower()
    if normalized not in DTYPE_MAP:
        raise ValueError(
            f"Unsupported dtype '{dtype_str}'. "
            f"Allowed values are: {list(DTYPE_MAP.keys())}"
        )
    return DTYPE_MAP[normalized]


def ddp_setup() -> int:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl", device_id=torch.device(f"cuda:{local_rank}")
    )
    return local_rank


def ddp_cleanup() -> None:
    dist.destroy_process_group()


def ddp_is_dist_available() -> bool:
    """Return True when torch.distributed has been properly initialised."""
    return dist.is_available() and dist.is_initialized()


def ddp_barrier() -> None:
    """Call a distributed barrier only when the process group is active."""
    if ddp_is_dist_available():
        dist.barrier()


@contextmanager
def rank_zero_evaluation(*, is_main_process: bool, timeout_seconds: float = 86400):
    """Keep every rank in a rank-zero evaluation hook until its work completes.

    All ranks must enter in the same order. Only rank zero should run the body.
    Peers wait for its outcome on a separate CPU/Gloo group, leaving no NCCL
    collective pending during slow generation, scoring, or artifact uploads.
    The evaluation timeout is independent of the training collective timeout.
    """
    if not isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("Evaluation timeout_seconds must be finite and positive")
    if not ddp_is_dist_available() or dist.get_world_size() == 1:
        yield
        return
    if is_main_process != (dist.get_rank() == 0):
        raise ValueError("is_main_process must identify global rank zero")
    if not dist.is_gloo_available():
        raise RuntimeError("Distributed rank-zero evaluation requires the Gloo backend")

    group = dist.new_group(
        backend="gloo", timeout=timedelta(seconds=timeout_seconds)
    )
    outcome = [None]
    try:
        try:
            yield
        except BaseException as exc:
            outcome[0] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            dist.broadcast_object_list(
                outcome, src=0, group=group, device=torch.device("cpu")
            )
        if outcome[0] is not None:
            raise RuntimeError(f"Rank-zero evaluation failed: {outcome[0]}")
    finally:
        dist.destroy_process_group(group)
