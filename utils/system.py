import logging
import os
import random
from collections.abc import Iterable

import numpy as np
import torch

logger = logging.getLogger(__name__)


def check_secrets(required: Iterable[str]) -> None:
    """Verify required secrets are present before expensive work starts.

    Args:
        required: names of required secrets (e.g. env var names).
    """
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise OSError(
            f"Missing required secret(s): {', '.join(missing)}. "
            f"Set them as environment variables before running."
        )

    logger.info("All secrets are present.")


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed every RNG used across the data/training pipeline.

    Call this once, as the first thing in each script's main(), before any
    model/dataset/dataloader construction.

    Args:
        seed: Seed value applied to all libraries.
        deterministic: If True, force fully deterministic (but slower) CUDA
            kernels. Default False for normal training throughput; note that
            4-bit bitsandbytes quantization and SDPA/flash-attention are not
            bit-exact deterministic regardless of this flag.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op if no GPU visible

    try:
        from transformers import set_seed as hf_set_seed

        hf_set_seed(seed)  # re-seeds random/np/torch; harmless redundancy
    except ImportError:
        pass

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True

    logger.info("Seeded all RNGs with seed=%d (deterministic=%s).", seed, deterministic)

