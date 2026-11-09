import hashlib
from pathlib import Path
import collections
import logging
from utils.logging import setup_logging
import math
import random
import sys

import hydra
from datasets import Dataset, DatasetDict, load_dataset
from datasets.exceptions import DatasetNotFoundError
from omegaconf import DictConfig, OmegaConf

from configs.fine_tune.conf_schema import register_prepare_dataset
from utils.hub import retry_hub_io
from utils.system import check_secrets, seed_everything

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

register_prepare_dataset()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retry-decorated I/O helpers
# ---------------------------------------------------------------------------


@retry_hub_io(attempts=5, min_wait=4, max_wait=60)
def _load_split_with_retry(source_id: str, split: str) -> Dataset:
    """Load a single HF dataset split, retrying on transient network errors."""
    logger.info("Loading '%s' (split='%s') from Hub …", source_id, split)
    ds = load_dataset(source_id, split=split)
    logger.info("  → %d rows loaded.", len(ds))
    return ds


@retry_hub_io(attempts=5, min_wait=4, max_wait=60)
def _push_to_hub_with_retry(
    dataset_dict: DatasetDict,
    output_id: str,
    private: bool,
) -> None:
    """Push a DatasetDict to the Hub, retrying on transient network errors."""
    logger.info("Pushing to Hub → '%s' (private=%s) …", output_id, private)
    dataset_dict.push_to_hub(output_id, private=private)
    logger.info("Successfully pushed to '%s'.", output_id)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_sampling_rates(rates: dict[str, float]) -> None:
    """
    Ensure every sampling rate is within [0.0, 1.0].

    Raises
    ------
    ValueError
        On any out-of-range rate.
    """
    bad = {k: v for k, v in rates.items() if not (0.0 <= v <= 1.0)}
    if bad:
        msg = (
            f"Sampling rates must be in [0.0, 1.0]. Invalid entries: {bad}. "
            "Fix 'dataset.complexity_sampling_rates' in prepare_dataset.yaml."
        )
        logger.error(
            f"Sampling rates must be in [0.0, 1.0]. Invalid entries: {bad}. "
            "Fix 'dataset.complexity_sampling_rates' in prepare_dataset.yaml."
        )
        raise ValueError(msg)


def _validate_categories_present(
    split_name: str,
    available: dict[str, int],
    requested: dict[str, float],
) -> None:
    """
    Warn (not hard-fail) if a configured category is absent from a split.
    That can legitimately happen for the small test split.
    """
    missing = set(requested) - set(available)
    if missing:
        logger.warning(
            "[%s split] The following configured categories are absent from this "
            "split and will be skipped: %s",
            split_name,
            sorted(missing),
        )


# ---------------------------------------------------------------------------
# Core sampling logic
# ---------------------------------------------------------------------------


def _compute_per_category_counts(
    category_sizes: dict[str, int],
    sampling_rates: dict[str, float],
) -> dict[str, int]:
    """
    Compute integer sample counts from per-category rates.

    For each category present in *category_sizes* **and** in *sampling_rates*:
        n = max(1, floor(size × rate))   if rate > 0
        n = 0                             if rate == 0

    Categories in *category_sizes* that are NOT listed in *sampling_rates*
    are silently excluded (rate implicitly 0).

    Parameters
    ----------
    category_sizes:
        Mapping ``{category: total_available_rows}`` for the current split.
    sampling_rates:
        Mapping ``{category: rate}`` from the YAML config (0.0 – 1.0).

    Returns
    -------
    Mapping ``{category: n_rows_to_sample}``.
    """
    counts: dict[str, int] = {}
    for cat, rate in sampling_rates.items():
        if cat not in category_sizes:
            continue  # absent from this split – already warned
        available = category_sizes[cat]
        if rate == 0.0:
            counts[cat] = 0
        else:
            n = math.floor(available * rate)
            counts[cat] = max(1, n)  # guarantee at least 1 row when rate > 0
    return counts


def stratified_sample_split(
    dataset: Dataset,
    stratify_column: str,
    sampling_rates: dict[str, float],
    seed: int,
    split_name: str,
) -> Dataset:
    """
    Draw a stratified sample from a single HF ``Dataset`` split.

    For each ``sql_complexity`` category, a random ``rate × count`` rows are
    selected using a deterministic, per-category-seeded shuffle so results
    are fully reproducible and independent across categories.

    Parameters
    ----------
    dataset:
        Source HF ``Dataset`` (one split).
    stratify_column:
        Column whose distinct values define the strata.
    sampling_rates:
        Mapping ``{category: fraction_of_category_to_keep}`` from YAML.
        Each rate is in [0.0, 1.0].
    seed:
        Base random seed; each category gets ``seed + hash(category)`` to
        guarantee independent, reproducible shuffles.
    split_name:
        Human-readable name used only for log messages.

    Returns
    -------
    A new ``Dataset`` containing the sampled rows (all original columns kept),
    shuffled so categories are interleaved.
    """
    logger.info(
        "[%s split] Building per-category index (N=%d rows) …", split_name, len(dataset)
    )

    # ── O(N) single-pass partition into per-category index lists ─────────────
    cat_to_indices: dict[str, list[int]] = {}
    for idx, value in enumerate(dataset[stratify_column]):
        cat_to_indices.setdefault(value, []).append(idx)

    available_counts = {cat: len(idxs) for cat, idxs in cat_to_indices.items()}

    # Log source distribution for this split
    logger.info("[%s split] Source category distribution:", split_name)
    for cat, cnt in sorted(available_counts.items()):
        logger.info("  %-22s  %6d rows", cat, cnt)

    # Validate configured categories exist in this split
    _validate_categories_present(split_name, available_counts, sampling_rates)

    # Resolve integer counts per category
    per_cat_counts = _compute_per_category_counts(available_counts, sampling_rates)

    # ── Per-category deterministic sampling ───────────────────────────────────
    selected_indices: list[int] = []
    for cat, n in per_cat_counts.items():
        if n == 0:
            logger.info("[%s split]   Skipping '%s' (rate=0).", split_name, cat)
            continue

        cat_indices = cat_to_indices[cat]
        # Use a category-specific seed so each stratum shuffles independently.
        category_hash = int.from_bytes(
            hashlib.sha256(cat.encode("utf-8")).digest(), "big"
        )
        cat_seed = seed ^ category_hash

        rng = random.Random(cat_seed)
        shuffled = cat_indices[:]

        rng.shuffle(shuffled)

        selected = shuffled[:n]

        selected_indices.extend(selected)
        rate_used = sampling_rates.get(cat, 0.0)

        logger.info(
            "[%s split]   %-22s  sampled %d / %d  (rate=%.4f)",
            split_name,
            cat,
            n,
            len(cat_indices),
            rate_used,
        )

    # Shuffle the final index pool so categories are interleaved
    random.Random(seed).shuffle(selected_indices)

    subset = dataset.select(selected_indices)
    logger.info(
        "[%s split] Stratified subset ready: %d rows total.", split_name, len(subset)
    )
    return subset


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def log_final_distribution(dd: DatasetDict, stratify_column: str) -> None:
    """Log per-category counts and percentages for every split in *dd*."""
    for split_name, split_ds in dd.items():
        counter = collections.Counter(split_ds[stratify_column])
        total = len(split_ds)
        logger.info(
            "─── Final distribution | split='%s' | %d rows ───", split_name, total
        )
        for cat, count in sorted(counter.items()):
            pct = 100.0 * count / total if total else 0.0
            logger.info("  %-22s  %5d  (%.2f %%)", cat, count, pct)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def prepare_and_upload(cfg: DictConfig) -> None:
    """
    End-to-end pipeline:
      load train + test  →  stratified sample per split  →  push to Hub.

    Parameters
    ----------
    cfg:
        Fully resolved Hydra ``DictConfig`` from prepare_dataset.yaml.
    """
    # ── 0. Config summary ────────────────────────────────────────────────────
    logger.info("Effective configuration:\n%s", OmegaConf.to_yaml(cfg))

    dataset_cfg = cfg.dataset
    hub_cfg = cfg.hub
    seed = cfg.system.seed

    # ── 1. Parse and validate sampling rates ─────────────────────────────────
    sampling_rates: dict[str, float] = OmegaConf.to_container(
        dataset_cfg.complexity_sampling_rates, resolve=True
    )
    _validate_sampling_rates(sampling_rates)

    logger.info("Configured sampling rates:")
    for cat, rate in sorted(sampling_rates.items()):
        logger.info("  %s  %.4f  (%.1f %%)", cat, rate, rate * 100)

    # ── 2. Load source splits ─────────────────────────────────────────────────
    source_splits: dict[str, Dataset] = {}
    for split_name in ("train", "test"):
        try:
            source_splits[split_name] = _load_split_with_retry(
                dataset_cfg.source_id, split=split_name
            )
        except DatasetNotFoundError:
            logger.critical(
                "Source dataset '%s' not found on Hugging Face Hub. "
                "Verify 'dataset.source_id' in prepare_dataset.yaml.",
                dataset_cfg.source_id,
            )
            raise

        # Sanity-check stratification column exists
        ds = source_splits[split_name]
        if dataset_cfg.stratify_column not in ds.column_names:
            msg = (
                f"Stratification column '{dataset_cfg.stratify_column}' not "
                f"found in '{split_name}' split columns: {ds.column_names}. "
                "Check 'dataset.stratify_column' in prepare_dataset.yaml."
            )
            logger.error(msg)
            raise ValueError(msg)

    # ── 3. Stratified sampling — applied independently per split ──────────────
    sampled_splits: dict[str, Dataset] = {}
    for split_name, source_ds in source_splits.items():
        sampled_splits[split_name] = stratified_sample_split(
            dataset=source_ds,
            stratify_column=dataset_cfg.stratify_column,
            sampling_rates=sampling_rates,
            seed=seed,
            split_name=split_name,
        )

    # ── 4. Assemble DatasetDict ───────────────────────────────────────────────
    dd = DatasetDict(sampled_splits)  # keys: "train", "test"

    # ── 5. Sanity-check final distribution and renaming "test to "validation" ───────────────────────────────────
    log_final_distribution(dd, dataset_cfg.stratify_column)

    total_rows = sum(len(v) for v in dd.values())
    logger.info(
        "Grand total across all splits: %d rows  (train=%d, test=%d)",
        total_rows,
        len(dd["train"]),
        len(dd["test"]),
    )
    dd["validation"] = dd.pop("test")

    # ── 6. Push to Hub ────────────────────────────────────────────────────────
    try:
        _push_to_hub_with_retry(
            dataset_dict=dd,
            output_id=dataset_cfg.output_id,
            private=hub_cfg.private,
        )
    except Exception:
        logger.error("Failed to push dataset to Hub repo '%s'", dataset_cfg.output_id)
        raise

    logger.info(
        "Done. Dataset available at: https://huggingface.co/datasets/%s",
        dataset_cfg.output_id,
    )


# ---------------------------------------------------------------------------
# Hydra entry-point
# ---------------------------------------------------------------------------


@hydra.main(
    version_base=None,
    config_path="pkg://configs",
    config_name="fine_tune/prepare_dataset",
)
def main(cfg: DictConfig) -> None:
    """Hydra entry-point: validate env, then run the pipeline."""

    log_filename = f"{Path(__file__).stem}.log"
    setup_logging(filename=log_filename)

    # Lock the config to catch typo'd keys early
    OmegaConf.set_struct(cfg, True)

    ENV_VAR_LIST = ["HF_TOKEN"]
    check_secrets(ENV_VAR_LIST)

    seed_everything(cfg.system.seed)

    prepare_and_upload(cfg)


if __name__ == "__main__":
    main()
