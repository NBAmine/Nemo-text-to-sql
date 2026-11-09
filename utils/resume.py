"""Trainer hooks for ordinary DDP resumes with newly added workers."""

import hashlib
import logging
from pathlib import Path

import torch
from transformers import set_seed

from utils.checkpoint import CheckpointError, METADATA_CALLBACK, _read_json

logger = logging.getLogger(__name__)


def added_rank_seed(*, seed: int, attempt_id: str, global_step: int, rank: int) -> int:
    """Derive stable, distinct 32-bit seeds for ranks added at a checkpoint."""
    identity = f"ddp-resume-v1:{seed}:{attempt_id}:{global_step}".encode("utf-8")
    base = int.from_bytes(hashlib.sha256(identity).digest()[:4], "big")
    return (base + rank) % (2**32)


class WorldSizeResumeMixin:
    """Restore existing ranks normally and deterministically seed added DDP ranks.

    Keep this before Trainer/SFTTrainer in the method resolution order. The
    checkpoint resolver validates opt-in, topology and batch compatibility first.
    """

    def __init__(self, *args, allow_world_size_change=False, **kwargs):
        self.allow_world_size_change = allow_world_size_change
        super().__init__(*args, **kwargs)

    def _get_train_sampler(self, *args, **kwargs):
        sampler = super()._get_train_sampler(*args, **kwargs)
        if self.allow_world_size_change and hasattr(sampler, "generator") and sampler.generator is None:
            # Sampler order must stay shared across ranks even after model RNGs
            # diverge. Accelerate retains or wraps this dedicated generator.
            seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
            sampler.generator = torch.Generator().manual_seed(seed)
        return sampler

    def _load_rng_state(self, checkpoint):
        if checkpoint is None or not self.allow_world_size_change:
            return super()._load_rng_state(checkpoint)
        state = _read_json(Path(checkpoint) / "trainer_state.json")
        try:
            metadata = state["stateful_callbacks"][METADATA_CALLBACK]["args"]["metadata"]
            saved_world = metadata["contract"]["world_size"]
            seed = metadata["contract"]["seed"]
        except (KeyError, TypeError) as error:
            raise CheckpointError("Cannot initialize resume RNG without checkpoint metadata.") from error
        rank = self.args.process_index
        if 1 < saved_world < self.args.world_size and rank >= saved_world:
            rank_seed = added_rank_seed(seed=seed, attempt_id=metadata["attempt_id"],
                                        global_step=state["global_step"], rank=rank)
            set_seed(rank_seed)
            logger.info("Initialized added DDP rank %s RNG at step %s with seed %s.",
                        rank, state["global_step"], rank_seed)
            return
        rng_file = f"rng_state_{rank}.pth" if self.args.world_size > 1 else "rng_state.pth"
        if not (Path(checkpoint) / rng_file).is_file():
            raise CheckpointError(f"Missing saved RNG state for existing rank {rank}: {rng_file}")
        return super()._load_rng_state(checkpoint)
