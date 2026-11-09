"""Sample resources, save full histories, and forward metrics from global rank 0.

See docs/resources-code-walkthrough.md for the detailed implementation guide.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import socket
import threading
import time
from collections import Counter, deque
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

import psutil
import torch

logger = logging.getLogger(__name__)
GIB = 1024**3
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resource_output_dir(*parts, root="artifacts/resources"):
    """Resolve a resource directory without creating it.

    Args:
        *parts: Workload/run labels to encode as directory components.
        root: Absolute root or path relative to the project directory.

    Returns:
        Path: Absolute resource directory with sanitized labels.
    """
    base = Path(root)
    if not base.is_absolute():
        base = PROJECT_ROOT / base
    components = []
    reserved = {"CON", "PRN", "AUX", "NUL"} | {
        f"{prefix}{index}" for prefix in ("COM", "LPT") for index in range(1, 10)
    }
    for part in parts:
        label = str(part)
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", label).strip(". ")[:80]
        if not safe or safe != label or safe.split(".")[0].upper() in reserved:
            digest = hashlib.sha256(label.encode("utf-8")).hexdigest()[:12]
            safe = f"{safe or 'id'}-{digest}"
        components.append(safe)
    return base.resolve().joinpath(*components)


def _atomic_json(path, value):
    """Atomically write a JSON file in the same directory as the target file.
    This is used to avoid having partially written JSON files for monitoring.

    Args:
        path: Destination Path whose parent directory already exists.
        value: JSON-serializable content to write.

    Returns:
        None.
    """
    temporary = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".resources-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, allow_nan=False)
            handle.write("\n")

        for attempt in range(5):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                # Windows readers/antivirus can briefly deny replacement of an open file.
                if attempt == 4:
                    raise
                time.sleep(0.01 * (attempt + 1))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _add_stats(stats, metrics):
    """Update per-metric counts, sums, and maxima in place.

    Args:
        stats: Mutable mapping of accumulated metric statistics.
        metrics: Metric names mapped to readings; nonfinite values are skipped.

    Returns:
        None.
    """
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and math.isfinite(value):
            item = stats.setdefault(key, {"count": 0, "sum": 0.0, "max": value})
            item["count"] += 1
            item["sum"] += value
            item["max"] = max(item["max"], value)


def _uuid_text(value):
    """Normalize a CUDA UUID for NVML lookup.

    Args:
        value: GPU identifier as bytes or a string-convertible value.

    Returns:
        str: Identifier with a GPU- or MIG- prefix.
    """
    if isinstance(value, bytes):
        value = value.decode("ascii")
    value = str(value)
    return value if value.startswith(("GPU-", "MIG-")) else f"GPU-{value}"


class SystemResourceTracker:
    """Collect per-rank resources with bounded RAM and complete disk histories."""

    def __init__(
        self,
        run,
        interval=1.0,
        *,
        output_dir=None,
        session_id=None,
        rank=None,
        local_rank=None,
        world_size=None,
        device_indices=None,
        buffer_size=1024,
        shutdown_timeout=5.0,
        warning_interval=60.0,
    ):
        """Prepare monitoring state without starting background threads.

        Args:
            run: W&B run or None; only global rank 0 retains the run.
            interval: Seconds to wait between sampling iterations.
            output_dir: Resource directory; must be shared across DDP ranks.
            session_id: Fresh launch ID; must match across DDP ranks.
            rank: Global rank, defaulting to RANK or 0.
            local_rank: Node-local rank, defaulting to LOCAL_RANK or 0.
            world_size: Rank count, defaulting to WORLD_SIZE or 1.
            device_indices: CUDA indices for allocator readings, or None for all.
            buffer_size: Maximum number of recent records retained in RAM.
            shutdown_timeout: Seconds budgeted for background-thread shutdown.
            warning_interval: Seconds between repeated warnings and NVML retries.

        Returns:
            None.
        """

        if not math.isfinite(interval) or interval <= 0 or buffer_size < 1:
            raise ValueError("Monitoring interval and buffer size must be positive")

        self.rank = int(os.environ.get("RANK", 0) if rank is None else rank)
        self.local_rank = int(
            os.environ.get("LOCAL_RANK", 0) if local_rank is None else local_rank
        )
        self.world_size = int(
            os.environ.get("WORLD_SIZE", 1) if world_size is None else world_size
        )
        self.run = run if self.rank == 0 else None
        self.interval = interval
        self.shutdown_timeout = shutdown_timeout
        self.warning_interval = warning_interval
        self.session_id = session_id or uuid4().hex

        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.session_id):
            raise ValueError("Invalid monitoring session ID")

        if self.world_size > 1 and (output_dir is None or session_id is None):
            raise ValueError(
                "Distributed monitoring needs a shared output_dir and session_id"
            )

        base = (
            output_dir
            if output_dir is not None
            else resource_output_dir(
                "standalone", getattr(run, "id", None) or self.session_id
            )
        )
        self.output_dir = Path(base)
        self.session_dir = self.output_dir / f"session-{self.session_id}"
        self.history_path = self.session_dir / f"rank-{self.rank}.jsonl"
        self.summary_path = self.session_dir / f"rank-{self.rank}.summary.json"
        self.identity_path = self.session_dir / f"rank-{self.rank}.identity.json"
        self.node = socket.gethostname()
        # Preserve the exact hostname in raw data; use a safe metric path segment.
        self.node_key = re.sub(r"[^A-Za-z0-9_.-]", "_", self.node)
        self.device_indices = device_indices
        self.records = deque(maxlen=buffer_size)
        self.stats = {}
        self.errors = Counter()
        self._warnings = {}
        self._warning_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._writer_stop = threading.Event()
        self._thread = None
        self._writer = None
        self._started = False
        self._stopped = False
        self._start_time = None
        self._stop_time = None
        self._sequence = 0
        self._train_step = None
        self._devices = []
        self._visible_devices = []
        self._nvml = None
        self._nvml_retry_at = 0.0
        self._handles = {}
        self._offsets = {}
        self._latest = {}
        self._aggregate_stats = {}
        self._axis_offset = None
        self._last_axis = 0.0
        self._unpersisted = 0
        self._metrics_configured = False

    @property
    def elapsed_seconds(self):
        """Read this session's monotonic monitoring duration.

        Returns:
            float: Elapsed seconds, frozen once sampling stops; zero before start.
        """
        if self._start_time is None:
            return 0.0
        end = self._stop_time if self._stop_time is not None else time.monotonic()
        return max(0.0, end - self._start_time)

    def _warn(self, key, error):
        """Count a failure and issue a rate-limited warning.

        Args:
            key: Error category used for counting and rate limiting.
            error: Exception or message describing the failure.

        Returns:
            None.
        """
        with self._warning_lock:
            self.errors[key] += 1
            now = time.monotonic()
            if now - self._warnings.get(key, -math.inf) >= self.warning_interval:
                logger.warning(
                    "Resource monitoring rank %s: %s: %s", self.rank, key, error
                )
                self._warnings[key] = now

    def _read(self, errors, key, operation):
        """Attempt a reading and record any failure without propagating it.

        Args:
            errors: Current sample's error mapping, updated on failure.
            key: Reading identifier used for errors and warnings.
            operation: Callable taking no arguments that performs the reading.

        Returns:
            The operation's result, or None if it raises an exception.
        """
        try:
            return operation()
        except Exception as error:
            errors[key] = f"{type(error).__name__}: {error}"
            self._warn(key, error)
            return None

    def _discover_devices(self):
        """Discover CUDA-visible devices and their available UUIDs.

        Returns:
            list[dict]: CUDA indices and UUIDs (possibly None); empty without CUDA.
        """
        try:
            if not torch.cuda.is_available():
                return []
            indices = range(torch.cuda.device_count())
        except Exception as error:
            self._warn("cuda_discovery", error)
            return []

        devices = []
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")

        for index in indices:
            uuid = None
            try:
                properties = torch.cuda.get_device_properties(index)
                value = getattr(properties, "uuid", None)
                if value:
                    uuid = _uuid_text(value)
                elif index < len(visible) and visible[index].startswith(
                    ("GPU-", "MIG-")
                ):
                    uuid = visible[index]
                else:
                    self._warn(
                        "cuda_identity",
                        f"CUDA device {index} has no UUID; NVML readings omitted",
                    )
            except Exception as error:
                self._warn("cuda_identity", error)
            devices.append({"cuda_index": index, "uuid": uuid})
        return devices

    def _identities(self):
        """Read available rank identity files for the current session.

        Returns:
            list[dict]: Successfully decoded identities; unreadable files are skipped.
        """
        result = []
        for path in self.session_dir.glob("rank-*.identity.json"):
            try:
                result.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception as error:
                self._warn("read_identity", error)
        return result

    def _device_metrics(self, metrics, errors):
        """Collect unique node GPU readings on local rank 0.

        Args:
            metrics: Mapping updated with valid device-wide NVML readings.
            errors: Mapping updated with device collection failures.

        Returns:
            None.
        """
        if self.local_rank != 0:
            return

        uuids = {
            device["uuid"]
            for identity in self._identities()
            if identity["node"] == self.node
            for device in identity["devices"]
            if device["uuid"] is not None
        }
        if not uuids:
            return

        if self._nvml is None:
            if time.monotonic() < self._nvml_retry_at:
                errors["nvml_init"] = "NVML unavailable; waiting before retry"
                return
            try:
                import pynvml

                pynvml.nvmlInit()
                self._nvml = pynvml
            except Exception as error:
                errors["nvml_init"] = f"{type(error).__name__}: {error}"
                self._warn("nvml_init", error)
                self._nvml_retry_at = time.monotonic() + self.warning_interval
                return

        for uuid in sorted(uuids):
            prefix = f"device/{self.node_key}/{uuid}"
            try:
                if uuid not in self._handles:
                    self._handles[uuid] = self._nvml.nvmlDeviceGetHandleByUUID(uuid)
                handle = self._handles[uuid]
                memory = self._read(
                    errors,
                    f"{prefix}/memory",
                    lambda: self._nvml.nvmlDeviceGetMemoryInfo(handle),
                )
                if memory is not None:
                    for label in ("used", "total"):
                        value = getattr(memory, label, None)
                        if value is not None:
                            metrics[f"{prefix}/nvml_memory_{label}_gib"] = value / GIB
                utilization = self._read(
                    errors,
                    f"{prefix}/utilization",
                    lambda: self._nvml.nvmlDeviceGetUtilizationRates(handle),
                )
                if utilization is not None and utilization.gpu is not None:
                    metrics[f"{prefix}/nvml_utilization_percent"] = utilization.gpu
            except Exception as error:
                errors[prefix] = f"{type(error).__name__}: {error}"
                self._warn(prefix, error)
                self._handles.pop(uuid, None)

    def _sample(self):
        """Collect one sample, update statistics, and attempt to persist it.

        Returns:
            dict: Record containing identity, timing, metrics, and reading errors.
        """
        metrics, errors = {}, {}
        prefix = f"process/rank_{self.rank}"

        rss = self._read(errors, "rss", lambda: psutil.Process().memory_info().rss)
        if rss is not None:
            metrics[f"{prefix}/rss_gib"] = rss / GIB

        allocated, reserved = [], []

        for device in self._devices:
            index = device["cuda_index"]

            for label, function in (
                ("allocated", torch.cuda.memory_allocated),
                ("reserved", torch.cuda.memory_reserved),
                ("peak_allocated_since_reset", torch.cuda.max_memory_allocated),
            ):
                key = f"{prefix}/cuda_{index}/torch_{label}_gib"
                value = self._read(errors, key, lambda f=function, i=index: f(i))

                if value is not None:
                    metrics[key] = value / GIB
                    if label == "allocated":
                        allocated.append(value / GIB)
                    elif label == "reserved":
                        reserved.append(value / GIB)

        if self._devices and len(allocated) == len(self._devices):
            metrics[f"{prefix}/torch_allocated_sum_gib"] = sum(allocated)
        if self._devices and len(reserved) == len(self._devices):
            metrics[f"{prefix}/torch_reserved_sum_gib"] = sum(reserved)

        self._read(
            errors, "collect_devices", lambda: self._device_metrics(metrics, errors)
        )

        # Unavailable/nonfinite values never become fictitious zero readings.
        metrics = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
        }

        record = {
            "schema_version": 1,
            "session_id": self.session_id,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "node": self.node,
            "timestamp_unix_seconds": time.time(),
            "elapsed_seconds": self.elapsed_seconds,
            "sequence": self._sequence,
            "train_global_step": self._train_step,
            "metrics": metrics,
            "errors": errors,
        }

        self._sequence += 1
        self.records.append(record)

        _add_stats(self.stats, metrics)

        try:
            with self.history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
        except Exception as error:
            self._unpersisted += 1
            self._warn("persist_history", error)

        self._save_summary(done=False)

        return record

    def _save_summary(self, *, done):
        """Persist this rank's current statistics and completion state.

        Args:
            done: Whether this rank has finished sampling.

        Returns:
            None.
        """
        try:
            with self._warning_lock:
                errors = dict(self.errors)
            _atomic_json(
                self.summary_path,
                {
                    "session_id": self.session_id,
                    "rank": self.rank,
                    "node": self.node,
                    "done": done,
                    "elapsed_seconds": self.elapsed_seconds,
                    "samples": self._sequence,
                    "unpersisted_records": self._unpersisted,
                    "errors": errors,
                    "stats": self.stats,
                },
            )
        except Exception as error:
            self._warn("persist_summary", error)

    def _monitor(self):
        """Sample until stopped, then finalize NVML and the rank summary.

        Returns:
            None.
        """
        try:
            while True:
                try:
                    self._sample()
                except Exception as error:
                    self._warn("collect_sample", error)
                if self._stop_event.wait(self.interval):
                    # Capture allocator high-water marks even between regular polls.
                    try:
                        self._sample()
                    except Exception as error:
                        self._warn("final_sample", error)
                    break
        finally:
            self._stop_time = time.monotonic()
            if self._nvml is not None:
                try:
                    self._nvml.nvmlShutdown()
                except Exception as error:
                    self._warn("nvml_shutdown", error)
            self._save_summary(done=True)

    def configure_metrics(self):
        """Configure W&B's resource axis while preserving Trainer's definitions.

        Returns:
            None.
        """
        if self.run is not None:
            try:
                self.run.define_metric("resources/elapsed_seconds")
                self.run.define_metric(
                    "resources/*", step_metric="resources/elapsed_seconds"
                )
                self._metrics_configured = True
            except Exception as error:
                self._metrics_configured = False
                self._warn("define_metrics", error)

    def callback(self):
        """Build a local Trainer callback that records the observed training step.

        Returns:
            TrainerCallback: Callback that tracks steps without peer communication.
        """
        from transformers.trainer_callback import TrainerCallback

        tracker = self

        class ResourceStepCallback(TrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                """Capture the initial step and configure resource metrics.

                Args:
                    args: Trainer arguments (unused).
                    state: Trainer state supplying global_step.
                    control: Trainer control object (unused).
                    **kwargs: Additional Trainer callback context (unused).

                Returns:
                    None.
                """
                tracker._train_step = state.global_step
                # Trainer configures its integration lazily at on_train_begin.
                tracker.configure_metrics()

            def on_step_end(self, args, state, control, **kwargs):
                """Refresh the observed training step after an optimizer step.

                Args:
                    args: Trainer arguments (unused).
                    state: Trainer state supplying global_step.
                    control: Trainer control object (unused).
                    **kwargs: Additional Trainer callback context (unused).

                Returns:
                    None.
                """
                tracker._train_step = state.global_step

        return ResourceStepCallback()

    def _axis_time(self):
        """Persist cumulative writer monitoring time for the resource chart axis.

        Returns:
            float: Nondecreasing elapsed seconds across sessions in this directory.
        """
        path = self.output_dir / "elapsed.json"

        if self._axis_offset is None:
            try:
                offset = json.loads(path.read_text(encoding="utf-8"))["elapsed_seconds"]
                if not math.isfinite(offset) or offset < 0:
                    raise ValueError("Invalid elapsed-time offset")
                self._axis_offset = offset
            except FileNotFoundError:
                self._axis_offset = 0.0

        value = max(self._last_axis, self._axis_offset + self.elapsed_seconds)

        # Persist BEFORE logging, so a restarted writer cannot move the axis back.
        _atomic_json(path, {"elapsed_seconds": value, "session_id": self.session_id})

        self._last_axis = value

        return value

    def _aggregates(self):
        """Combine latest forwarded readings with rank and device coverage checks.

        Returns:
            dict: Coverage counts and available job memory sums/maxima.
        """
        metrics = {
            "job/ranks_expected": self.world_size,
            "job/ranks_seen": len(self._latest),
        }

        for label in ("rss_gib", "torch_allocated_sum_gib", "torch_reserved_sum_gib"):
            values = []

            for rank, record in self._latest.items():
                value = record["metrics"].get(f"process/rank_{rank}/{label}")
                if value is not None:
                    values.append(value)

            metrics[f"job/ranks_with_{label}"] = len(values)

            if len(values) == self.world_size:
                # These are sums/maxima of latest rank samples, not synchronized measurements.
                metrics[f"job/latest_rank_{label}_sum"] = sum(values)
                metrics[f"job/latest_rank_{label}_max"] = max(values)

        identities = self._identities()

        expected = {
            (identity["node_key"], device["uuid"])
            for identity in identities
            for device in identity["devices"]
            if device["uuid"] is not None
        }

        device_values = {}

        for record in self._latest.values():
            for key, value in record["metrics"].items():
                if key.startswith("device/") and key.endswith("/nvml_memory_used_gib"):
                    device_values[key] = value

        metrics["job/nvml_devices_with_memory"] = len(device_values)
        metrics["job/nvml_devices_identified"] = len(expected)

        all_identified = len(identities) == self.world_size and all(
            device["uuid"] is not None
            for identity in identities
            for device in identity["devices"]
        )

        if all_identified and expected and len(device_values) == len(expected):
            metrics["job/latest_nvml_device_memory_used_sum_gib"] = sum(
                device_values.values()
            )
            metrics["job/latest_nvml_device_memory_used_max_gib"] = max(
                device_values.values()
            )

        return metrics

    def _forward_once(self):
        """Forward a bounded history batch, retaining failed row positions for retry.

        Returns:
            None.
        """
        if not self._metrics_configured:
            self.configure_metrics()

        changed = False

        for path in sorted(self.session_dir.glob("rank-*.jsonl")):
            try:
                with path.open("rb") as handle:
                    handle.seek(self._offsets.get(path.name, 0))

                    for _ in range(256):
                        if self._writer_stop.is_set():
                            return

                        line = handle.readline(1024 * 1024)

                        if not line or not line.endswith(b"\n"):
                            break  # Writer has not completed this record yet.
                        try:
                            record = json.loads(line)
                            for key in (
                                "rank",
                                "metrics",
                                "session_id",
                                "elapsed_seconds",
                                "timestamp_unix_seconds",
                                "sequence",
                                "train_global_step",
                            ):
                                record[key]
                        except (ValueError, KeyError, TypeError) as error:
                            # A partial disk write followed by recovery can leave a damaged line.
                            # Keep the raw file and report the gap while forwarding later records.
                            self._offsets[path.name] = handle.tell()
                            self._warn("invalid_history_record", error)
                            continue

                        elapsed = self._axis_time()
                        prefix = f"resources/process/rank_{record['rank']}"
                        payload = {
                            f"resources/{key}": value
                            for key, value in record["metrics"].items()
                        }
                        payload.update(
                            {
                                "resources/elapsed_seconds": elapsed,
                                "resources/session_id": record["session_id"],
                                f"{prefix}/sample_elapsed_seconds": record[
                                    "elapsed_seconds"
                                ],
                                f"{prefix}/sample_timestamp_unix_seconds": record[
                                    "timestamp_unix_seconds"
                                ],
                                f"{prefix}/sample_sequence": record["sequence"],
                            }
                        )

                        if record["train_global_step"] is not None:
                            payload[f"{prefix}/observed_train_global_step"] = record[
                                "train_global_step"
                            ]

                        self.run.log(payload)

                        self._offsets[path.name] = handle.tell()
                        self._latest[record["rank"]] = record
                        changed = True
            except Exception as error:
                self._warn("forward_history", error)
        if changed and not self._writer_stop.is_set():
            metrics = self._aggregates()
            try:
                self.run.log(
                    {
                        **{f"resources/{key}": value for key, value in metrics.items()},
                        "resources/elapsed_seconds": self._axis_time(),
                        "resources/session_id": self.session_id,
                    }
                )
                _add_stats(self._aggregate_stats, metrics)
            except Exception as error:
                self._warn("log_aggregates", error)

    def _forward(self):
        """Run the W&B forwarding loop until writer shutdown is requested.

        Returns:
            None.
        """
        while not self._writer_stop.is_set():
            try:
                self._forward_once()
            except Exception as error:
                self._warn("forward_loop", error)
            self._writer_stop.wait(min(self.interval, 0.5))

    def _all_done(self):
        """Check rank completion and, when logging, history forwarding progress.

        Returns:
            bool: Whether every rank is done and required forwarding is complete.
        """
        for rank in range(self.world_size):
            path = self.session_dir / f"rank-{rank}.summary.json"

            try:
                if not json.loads(path.read_text(encoding="utf-8"))["done"]:
                    return False

                history = self.session_dir / f"rank-{rank}.jsonl"

                if (
                    self.run is not None
                    and history.exists()
                    and self._offsets.get(history.name, 0) < history.stat().st_size
                ):
                    return False
            except (OSError, ValueError, KeyError):
                return False
        return True

    def _attempt_summary(self):
        """Combine all session summaries on disk and update the W&B summary.

        Returns:
            None.
        """
        # Rank summaries include ALL samples, independently of buffer eviction or W&B failures.

        _atomic_json(
            self.session_dir / "job.summary.json", {"stats": self._aggregate_stats}
        )

        combined = {}
        sessions, unfinished = set(), []
        unpersisted = 0

        for path in self.output_dir.glob("session-*/*.summary.json"):
            item = json.loads(path.read_text(encoding="utf-8"))
            sessions.add(path.parent.name)

            if path.name != "job.summary.json":
                unpersisted += item["unpersisted_records"]
                if not item["done"]:
                    unfinished.append(str(path))

            for key, value in item["stats"].items():
                total = combined.setdefault(
                    key, {"count": 0, "sum": 0.0, "max": value["max"]}
                )
                total["count"] += value["count"]
                total["sum"] += value["sum"]
                total["max"] = max(total["max"], value["max"])

        summary = {
            "stats": combined,
            "sessions": len(sessions),
            "unfinished_rank_files": unfinished,
            "unpersisted_records": unpersisted,
            "current_session_complete": self._all_done(),
        }

        _atomic_json(self.output_dir / "attempt_summary.json", summary)

        if self.run is not None:
            values = {
                "resources/attempt/session_count": len(sessions),
                "resources/attempt/current_session_complete": summary[
                    "current_session_complete"
                ],
                "resources/attempt/unpersisted_records": unpersisted,
            }

            for key, value in combined.items():
                prefix = f"resources/attempt/{key}"
                maximum = (
                    "observed_max"
                    if "peak_allocated_since_reset" in key
                    else "sampled_max"
                )
                values[f"{prefix}/{maximum}"] = value["max"]
                values[f"{prefix}/sample_mean"] = value["sum"] / value["count"]
                values[f"{prefix}/valid_samples"] = value["count"]

            self.run.summary.update(values)

    def start(self):
        """Create session files and start sampling plus the optional W&B writer.

        Returns:
            None.
        """
        if self._started:
            return

        self._started = True
        self._start_time = time.monotonic()

        try:
            self.session_dir.mkdir(parents=True, exist_ok=True)

            if self.history_path.exists():
                raise FileExistsError(
                    f"Monitoring session already has rank history: {self.history_path}"
                )

            self._visible_devices = self._discover_devices()
            self._devices = [
                device
                for device in self._visible_devices
                if self.device_indices is None
                or device["cuda_index"] in self.device_indices
            ]

            _atomic_json(
                self.identity_path,
                {
                    "rank": self.rank,
                    "local_rank": self.local_rank,
                    "node": self.node,
                    "node_key": self.node_key,
                    "devices": self._visible_devices,
                },
            )

            self._thread = threading.Thread(
                target=self._monitor, name=f"resources-rank-{self.rank}", daemon=True
            )

            self._thread.start()

            if self.run is not None:
                self.configure_metrics()
                self._writer = threading.Thread(
                    target=self._forward, name="resources-wandb", daemon=True
                )
                self._writer.start()

            logger.info("Full resource history: %s", self.history_path)

        except Exception as error:
            self._warn("start", error)
            if self._thread is None:
                self._started = False

    def stop(self):
        """Request shutdown, wait within the timeout, and summarize on rank 0.

        Returns:
            None.
        """
        if not self._started or self._stopped:
            return

        self._stopped = True
        self._stop_event.set()

        deadline = time.monotonic() + self.shutdown_timeout

        if self._thread is not None:
            self._thread.join(max(0.0, deadline - time.monotonic()))
            if self._thread.is_alive():
                self._warn("stop_sampler", "Sampler did not stop before timeout")

        if self._writer is not None:
            # Wait only for files, never for distributed collectives. Failed peers time out.
            drain_deadline = deadline - min(0.5, self.shutdown_timeout / 2)

            while not self._all_done() and time.monotonic() < drain_deadline:
                self._writer_stop.wait(0.02)

            self._writer_stop.set()
            self._writer.join(max(0.0, deadline - time.monotonic()))

            if self._writer.is_alive():
                self._warn(
                    "stop_writer",
                    "Writer did not stop before timeout; raw history retained",
                )
                return

        if self.rank == 0 and self.session_dir.exists():
            try:
                self._attempt_summary()
            except Exception as error:
                self._warn("attempt_summary", error)

        if self._stop_time is None:
            self._stop_time = time.monotonic()
