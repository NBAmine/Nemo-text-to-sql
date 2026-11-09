"""Bounded, read-only SQLite result comparison for local evaluation.

This is a custom diagnostic protocol, not the official Spider metric. Output
columns are positional; unordered results use bag (duplicate-preserving) equality.
"""

from __future__ import annotations

from collections import Counter
from contextlib import closing
from dataclasses import asdict, dataclass
import math
import json
import logging
from pathlib import Path
from queue import Empty, Queue
import re
import sqlite3
import subprocess
import sys
from threading import Thread
import time


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueryLimits:
    timeout: float = 5.0
    max_rows: int = 100_000
    max_bytes: int = 8 * 1024 * 1024
    wait_time: float = 60.0

    def __post_init__(self):
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if not math.isfinite(self.wait_time) or self.wait_time <= 0:
            raise ValueError("wait_time must be finite and positive")
        if any(
            type(value) is not int or value <= 0
            for value in (self.max_rows, self.max_bytes)
        ):
            raise ValueError("result limits must be positive integers")


# Skip quoted contents and comments before looking for OUTER ORDER BY. In
# particular, ORDER BY in a CTE/subquery/window does not order the final output.
_TOKENS = re.compile(
    r"--[^\r\n]*|/\*[\s\S]*?(?:\*/|$)|'(?:''|[^'])*'|"
    r'"(?:""|[^"])*"|`(?:``|[^`])*`|\[[^\]]*\]|\w+|[^\s]'
)


def requires_order(sql: str) -> bool:
    depth = 0
    previous = ""
    for match in _TOKENS.finditer(sql):
        token = match.group()
        if token.startswith(("--", "/*")):
            continue
        if token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
        if depth == 0:
            if previous == "order" and token.lower() == "by":
                return True
            previous = token.lower()
    return False


def results_equal(pred_rows, gold_rows, pred_width, gold_width, *, ordered):
    """Compare complete tuples, preserving column positions and multiplicity.

    Numeric int/float equality follows Python's exact numeric equality; no
    rounding or string coercion is applied. Empty results still compare arity.
    """
    if pred_width != gold_width:
        return False
    if ordered:
        return pred_rows == gold_rows
    return Counter(pred_rows) == Counter(gold_rows)


class QueryFailure(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def _read_query(database: str, sql: str, limits: QueryLimits, *, warnings=None):

    if len(sql.encode("utf-8")) > 1024 * 1024:
        raise QueryFailure("query_rejected", "SQL exceeds 1 MiB")

    deadline = time.monotonic() + limits.timeout
    denied = False
    invalid_text_seen = False

    def decode_text(data):
        nonlocal invalid_text_seen
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            if not invalid_text_seen and warnings is not None:
                warnings.append(
                    "Result contains invalid UTF-8 text; original bytes preserved "
                    "with surrogateescape."
                )
            invalid_text_seen = True
            # Keep distinct invalid bytes distinct, without changing valid text
            # or confusing SQLite TEXT with BLOB values. Results stay in Python;
            # strings containing surrogates must not be bound back into SQLite.
            return data.decode("utf-8", errors="surrogateescape")

    def authorize(action, arg1, arg2, _database, _trigger):
        nonlocal denied
        allowed = action in {
            sqlite3.SQLITE_SELECT,
            sqlite3.SQLITE_READ,
            sqlite3.SQLITE_FUNCTION,
            sqlite3.SQLITE_RECURSIVE,
        }
        if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in {
            "load_extension",
            "readfile",
            "writefile",
            "eval",
        }:
            allowed = False
        denied = denied or not allowed
        return sqlite3.SQLITE_OK if allowed else sqlite3.SQLITE_DENY

    try:
        uri = Path(database).resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True, timeout=0.1)) as connection:
            connection.text_factory = decode_text
            connection.execute("PRAGMA query_only = ON")
            connection.set_authorizer(authorize)
            connection.set_progress_handler(
                lambda: int(time.monotonic() >= deadline), 1000
            )

            if hasattr(connection, "setlimit"):
                connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, limits.max_bytes)

            cursor = connection.execute(sql)

            if cursor.description is None:
                raise QueryFailure("query_rejected", "Expected a SELECT result")

            rows = []
            byte_count = 0
            for row in cursor:
                if time.monotonic() >= deadline:
                    raise QueryFailure("timeout", "Query exceeded its deadline")
                if len(rows) >= limits.max_rows:
                    raise QueryFailure("result_limit", "Result row limit exceeded")
                # Include approximate Python object overhead, including NULLs
                # and empty strings, rather than counting text payload alone.
                byte_count += 64 + sum(
                    32
                    + (
                        len(value.encode("utf-8", errors="surrogateescape"))
                        if isinstance(value, str)
                        else len(value)
                        if isinstance(value, bytes)
                        else 8
                    )
                    for value in row
                )
                if byte_count > limits.max_bytes:
                    raise QueryFailure("result_limit", "Result byte limit exceeded")
                rows.append(row)
            return rows, len(cursor.description)
    except sqlite3.Error as error:
        if time.monotonic() >= deadline:
            status = "timeout"
        elif denied:
            status = "query_rejected"
        elif isinstance(error, sqlite3.DataError):
            status = "result_limit"
        else:
            status = "execution_error"
        raise QueryFailure(status, str(error)) from error


def _spider_result_map(rows, width, select_units):
    """Retain the bundled Spider evaluator's selected-expression comparison."""
    if len(select_units) > width:
        raise QueryFailure(
            "comparison_error", "Parsed SELECT has more expressions than result columns"
        )
    result = {}
    for index, unit in enumerate(select_units):
        key = (
            tuple(unit[1]) if not unit[2] else (unit[0], tuple(unit[1]), tuple(unit[2]))
        )
        result[key] = [row[index] for row in rows]
    return result


def _compare_request(send, database, gold_sql, pred_sql, limits, select_units=None):
    """Keep result data in the worker; send only small diagnostic messages."""
    phase = "gold"
    gold_warnings, pred_warnings = [], []

    def send_result(result):
        warnings = [
            {"stage": stage, "warning": message}
            for stage, messages in (
                ("gold_execution", gold_warnings),
                ("prediction_execution", pred_warnings),
            )
            for message in messages
        ]
        if warnings:
            result["warnings"] = warnings
        send({"result": result})

    try:
        send({"phase": phase})
        gold_rows, gold_width = _read_query(
            database, gold_sql, limits, warnings=gold_warnings
        )
        gold_map = None
        if select_units is not None:
            gold_map = _spider_result_map(gold_rows, gold_width, select_units[0])
        if pred_sql is None:
            send_result({"status": "skipped", "correct": False})
            return

        phase = "prediction"
        send({"phase": phase})

        pred_rows, pred_width = _read_query(
            database, pred_sql, limits, warnings=pred_warnings
        )

        ordered = True if select_units is not None else requires_order(gold_sql)

        if select_units is None:
            correct = results_equal(
                pred_rows, gold_rows, pred_width, gold_width, ordered=ordered
            )
        else:
            correct = (
                _spider_result_map(pred_rows, pred_width, select_units[1]) == gold_map
            )
        send_result(
            {
                "status": "match" if correct else "mismatch",
                "correct": correct,
                "ordered": ordered,
                "gold_rows": len(gold_rows),
                "prediction_rows": len(pred_rows),
                "gold_columns": gold_width,
                "prediction_columns": pred_width,
            }
        )
    except QueryFailure as error:
        send_result(
            {
                "status": error.status,
                "correct": False,
                "reference_error": phase == "gold",
                "error": str(error),
            }
        )
    except Exception as error:
        send_result(
            {
                "status": "worker_error",
                "error": f"{type(error).__name__}: {error}",
            }
        )


def compare_queries(database, gold_sql, pred_sql, limits=QueryLimits(), *, worker=None):
    """Execute in a lightweight SQL worker with a parent-enforced deadline.

    Each query gets its own connection, progress deadline, row and byte budgets.
    Pass a context-managed SQLComparisonWorker to reuse it across comparisons.
    Without one, a temporary worker is closed before this call returns.
    Startup allows limits.wait_time per attempt, with one retry; execution
    allows timeout + 0.5 s per phase.
    ``pred_sql=None`` validates the reference without executing a failed output.
    This restricts SQL operations, but is not a general hostile-code sandbox.
    """
    return _run_comparison(database, gold_sql, pred_sql, limits, worker=worker)


def compare_spider_queries(
    database,
    gold_sql,
    pred_sql,
    gold_select,
    pred_select,
    limits=QueryLimits(),
    *,
    worker=None,
):
    """Bounded execution using legacy Spider expression and row-order rules.

    Select units must already have the legacy value/foreign-key normalization.
    This deliberately retains legacy quirks, including column-key collisions;
    use compare_queries for the separate positional, duplicate-aware protocol.
    """
    return _run_comparison(
        database, gold_sql, pred_sql, limits, (gold_select, pred_select), worker=worker
    )


def _run_comparison(
    database, gold_sql, pred_sql, limits, select_units=None, *, worker=None
):
    if worker is not None:
        return worker.compare(database, gold_sql, pred_sql, limits, select_units)
    with SQLComparisonWorker() as temporary:
        return temporary.compare(database, gold_sql, pred_sql, limits, select_units)


def _read_message(stream):
    # Only diagnostics cross this pipe, never query result rows. Bound malformed
    # output as well as valid messages so a broken worker cannot emit huge lines.
    line = stream.readline(64 * 1024)
    if not line:
        raise EOFError("Evaluation worker exited without a result")
    if not line.endswith(b"\n"):
        raise ValueError("Evaluation worker message exceeds 64 KiB")
    message = json.loads(line)
    if not isinstance(message, dict):
        raise ValueError("Invalid evaluation worker message")
    return message


def _worker_io(process, requests, messages):
    """Keep pipe reads AND writes off the thread enforcing the deadlines."""
    try:
        messages.put(_read_message(process.stdout))
        while True:
            request = requests.get()
            if request is None:
                process.stdin.close()
                return
            process.stdin.write(json.dumps(request).encode("ascii") + b"\n")
            process.stdin.flush()
            while True:
                message = _read_message(process.stdout)
                messages.put(message)
                if "result" in message:
                    break
    except Exception as error:
        messages.put(error)


class SQLComparisonWorker:
    """One sequential, reusable SQL subprocess, owned by an evaluation pass.

    The worker starts lazily and never imports the caller's entry script. A
    timeout discards it; the next comparison starts a clean replacement. Use
    a context manager to close it on success, reference errors, or interruption.
    Instances are not intended for concurrent calls.
    """

    def __init__(self):
        self._process = None
        self._thread = None
        self._requests = None
        self._messages = None
        self._closed = False

    @property
    def pid(self):
        return self._process.pid if self._process is not None else None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def _command(self):
        # Running this stdlib-only file directly avoids multiprocessing.spawn's
        # re-execution of final_eval.py (and its torch/transformers imports).
        # Isolated mode also prevents utils/logging.py from shadowing the
        # standard library when executing this file directly. No site hooks or
        # caller PYTHONPATH are needed by the SQL worker.
        return [sys.executable, "-I", "-S", "-u", str(Path(__file__).resolve())]

    def _receive(self, deadline, phase):
        try:
            message = self._messages.get(timeout=max(0, deadline - time.monotonic()))
        except Empty as error:
            raise TimeoutError(
                f"Evaluation worker exceeded its {phase} deadline"
            ) from error
        if isinstance(message, Exception):
            raise RuntimeError(
                f"Evaluation worker PID {self.pid} failed during {phase} "
                f"(exit code {self._process.poll()}): {message}"
            ) from message
        return message

    def _ensure_started(self, wait_time):
        if self._process is not None:
            if self._process.poll() is None:
                return
            logger.warning(
                "SQL worker PID %s exited while idle; replacing it", self.pid
            )
            self._stop()
        for attempt in range(1, 3):
            started_at = time.monotonic()
            try:
                self._process = subprocess.Popen(
                    self._command(),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    creationflags=(
                        subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                    ),
                )
                self._requests, self._messages = Queue(), Queue()
                self._thread = Thread(
                    target=_worker_io,
                    args=(self._process, self._requests, self._messages),
                    name="sql-worker-io",
                    daemon=True,
                )
                self._thread.start()
                if self._receive(started_at + wait_time, "startup") != {"ready": True}:
                    raise RuntimeError("Invalid evaluation worker startup handshake")
                logger.info(
                    "SQL worker PID %s ready in %.3fs",
                    self.pid,
                    time.monotonic() - started_at,
                )
                return
            except (OSError, RuntimeError, TimeoutError) as error:
                logger.warning(
                    "SQL worker startup attempt %s/2 failed (PID %s, allowance %.1fs): %s",
                    attempt,
                    self.pid,
                    wait_time,
                    error,
                )
                self._stop()
                if attempt == 2:
                    raise RuntimeError(
                        f"Evaluation worker failed to start in {wait_time} seconds "
                        f"per attempt after 2 attempts: {error}"
                    ) from error
            except BaseException:
                self._stop()
                raise

    def compare(
        self, database, gold_sql, pred_sql, limits=QueryLimits(), select_units=None
    ):
        if self._closed:
            raise RuntimeError("Evaluation worker is closed")
        if not isinstance(limits, QueryLimits):
            raise TypeError("limits must be a QueryLimits instance")
        database = Path(database).resolve()
        if not database.is_file():
            raise FileNotFoundError(database)
        self._ensure_started(limits.wait_time)
        phase = "dispatch"
        deadline = time.monotonic() + limits.timeout + 0.5
        self._requests.put(
            {
                "database": str(database),
                "gold_sql": gold_sql,
                "pred_sql": pred_sql,
                "limits": asdict(limits),
                "select_units": select_units,
            }
        )
        try:
            while True:
                message = self._receive(deadline, phase)
                if "result" in message and phase != "dispatch":
                    result = message["result"]
                    if result["status"] == "worker_error":
                        raise RuntimeError(result["error"])
                    if result["status"] == "timeout":
                        self._stop()
                    return result
                next_phase = {"dispatch": "gold", "gold": "prediction"}.get(phase)
                if next_phase is None or message != {"phase": next_phase}:
                    raise RuntimeError(
                        f"Invalid evaluation worker message during {phase}"
                    )
                phase = next_phase
                deadline = time.monotonic() + limits.timeout + 0.5
        except TimeoutError as error:
            logger.warning(
                "SQL worker PID %s timed out during %s; replacing it", self.pid, phase
            )
            self._stop()
            if phase == "dispatch":
                raise RuntimeError(
                    "Evaluation worker failed to begin comparison"
                ) from error
            return {
                "status": "timeout",
                "correct": False,
                "reference_error": phase == "gold",
                "error": "Parent terminated worker after its deadline",
            }
        except BaseException:
            self._stop()
            raise

    def _stop(self, *, graceful=False):
        process = self._process
        if process is None:
            return
        if self._requests is not None:
            self._requests.put(None)
        try:
            if not graceful and process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        finally:
            if self._thread is not None:
                self._thread.join(timeout=1.0)
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    stream.close()
            self._process = self._thread = self._requests = self._messages = None

    def close(self):
        self._closed = True
        self._stop(graceful=True)


def _worker_main():
    """Private stdio entry point; intentionally depends only on the stdlib."""

    def send(message):
        if "result" in message and "error" in message["result"]:
            message["result"]["error"] = message["result"]["error"][:4096]
        sys.stdout.buffer.write(json.dumps(message).encode("ascii") + b"\n")
        sys.stdout.buffer.flush()

    send({"ready": True})
    for line in sys.stdin.buffer:
        request = json.loads(line)
        request["limits"] = QueryLimits(**request["limits"])
        _compare_request(send, **request)


if __name__ == "__main__":
    _worker_main()
