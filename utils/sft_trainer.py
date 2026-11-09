"""SFTTrainer with the shared ordinary-DDP resize resume policy."""

from trl import SFTTrainer

from utils.resume import WorldSizeResumeMixin


class ResumableSFTTrainer(WorldSizeResumeMixin, SFTTrainer):
    """Use Trainer's standard loading, with explicit RNG handling for added ranks."""
