"""Lossless prediction records and a validated adapter for Spider's line files.

This module deliberately does not parse, repair, or score SQL. The upstream
evaluator remains responsible for correctness; this adapter owns sample identity
and the physical file format it consumes.

In function signatures, arguments after ``*`` must be passed by name. Type
annotations describe expected inputs and return values; ``T | None`` allows
either a value of type T or None. Defaults make arguments optional to supply.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FAILURE_SQL = "ERROR_GENERATION"
STATUSES = {"ok", "empty", "oom", "generation_error", "export_error"}
LINE_CONTROLS = "\r\n\t\v\f\x85\u2028\u2029"


def to_single_line_sql(sql: str) -> str:
    """Export SQLite SQL on one line without changing quoted contents.

    A literal or quoted identifier containing a line/tab control cannot be
    represented losslessly in Spider's TSV format and is rejected. This lexer
    never rewrites literals, identifiers, SQL expressions, or statement syntax.

    Args:
        sql: SQL text, possibly containing comments and multiple lines.
    Returns:
        A single-line SQL string with quoted contents preserved.
    Raises:
        ValueError: SQL is empty, has invalid quoting, or cannot be exported safely.
    """
    if not isinstance(sql, str) or "\x00" in sql:
        raise ValueError("SQL must be a string without NUL characters")
    out = []
    i = 0
    while i < len(sql):
        char = sql[i]
        if char in " \t\r\n\f":
            out.append(" ")
            i += 1
            while i < len(sql) and sql[i] in " \t\r\n\f":
                i += 1
        elif char in LINE_CONTROLS:
            raise ValueError("SQL contains an unsupported line control")
        elif sql.startswith("--", i):
            i += 2
            while i < len(sql) and sql[i] not in "\r\n":
                i += 1
            out.append(" ")
        elif sql.startswith("/*", i):
            end = sql.find("*/", i + 2)
            # SQLite also accepts block comments extending to end-of-input.
            i = len(sql) if end < 0 else end + 2
            out.append(" ")
        elif char in "'\"`[":
            closing = "]" if char == "[" else char
            start = i
            i += 1
            while i < len(sql):
                if sql[i] in LINE_CONTROLS:
                    raise ValueError(
                        "Quoted SQL contains an unrepresentable line/tab control"
                    )
                if sql[i] == closing:
                    # SQLite escapes quote characters by doubling them. Bracket
                    # identifiers end at the first closing bracket.
                    if closing != "]" and i + 1 < len(sql) and sql[i + 1] == closing:
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            else:
                raise ValueError("Unterminated SQL quoted value or identifier")
            out.append(sql[start:i])
        else:
            out.append(char)
            i += 1
    result = "".join(out).strip(" ")
    if not result:
        raise ValueError("SQL is empty after removing whitespace and comments")
    return result


def _text(record: dict, key: str) -> str:
    """Get a required string field from a record.

    Args:
        record: Dictionary containing the field.
        key: Name of the field to read.
    Returns:
        The field's string value.
    Raises:
        ValueError: The field is missing or is not a string.
    """
    value = record.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _json(value) -> str:
    """Serialize JSON with sorted keys and escaped Unicode line controls.

    Args:
        value: Any JSON-serializable Python value.
    Returns:
        JSON text on one physical line, without a trailing newline.
    """
    # Escape Unicode line separators too, keeping JSONL one physical line even
    # for raw outputs/questions containing non-ASCII line controls.
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True)
        .replace("\x85", "\\u0085")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _write_jsonl(path: Path, records) -> None:
    """Write UTF-8 JSONL with one serialized record per LF-terminated line.

    Args:
        path: Destination Path; any existing file is overwritten.
        records: Iterable of JSON-serializable values to write.
    Returns:
        None; writes the file without creating its parent directory.
    """
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(_json(record) + "\n")


def _sha256(path: Path) -> str:
    """Calculate a file's SHA256 hash.

    Args:
        path: Path of the file to read as bytes.
    Returns:
        The digest as a 64-character hexadecimal string.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(dataset, *, split: str) -> list[dict]:
    """Validate dataset rows and build an ordered manifest with snapshot-local IDs.

    Args:
        dataset: Iterable of rows with string db_id, question, and query fields;
            normally a Hugging Face datasets.Dataset.
        split: Keyword-only split label used as the sample ID prefix.
    Returns:
        A list of dictionaries containing sample_id, db_id, question, and gold_sql.
    Raises:
        ValueError: The dataset is empty or a required field/gold SQL is invalid.
    """
    manifest = []
    for index, sample in enumerate(dataset):
        db_id = _text(sample, "db_id").strip()
        gold_sql = _text(sample, "query")
        if not db_id or any(c.isspace() for c in db_id):
            raise ValueError(f"Sample {index} has an empty/invalid db_id")
        # Fail before expensive generation if gold cannot be exported losslessly.
        to_single_line_sql(gold_sql)
        manifest.append(
            {
                "sample_id": f"{split}:{index:08d}",
                "db_id": db_id,
                "question": _text(sample, "question"),
                "gold_sql": gold_sql,
            }
        )
    if not manifest:
        raise ValueError("Evaluation dataset is empty")
    return manifest


def prediction_record(
    sample: dict,
    raw_output: str = "",
    *,
    sql: str | None = None,
    status: str | None = None,
    error: str | None = None,
) -> dict:
    """Build one prediction record, retaining raw output, SQL, status, and error.

    Args:
        sample: Dictionary containing the input sample_id and db_id.
        raw_output: Original decoded model text; defaults to an empty string.
        sql: Keyword-only extracted SQL; None uses raw_output unchanged.
        status: Keyword-only outcome; None infers 'ok' or 'empty' from SQL.
        error: Keyword-only error description, or None when absent.
    Returns:
        A dictionary holding the sample identity, output, SQL, status, and error.
    Raises:
        ValueError: raw_output or the resolved SQL is not a string.
        KeyError: sample lacks sample_id or db_id.
    """
    if not isinstance(raw_output, str):
        raise ValueError("Model output must be a string")
    sql = raw_output if sql is None else sql
    if not isinstance(sql, str):
        raise ValueError("Extracted SQL must be a string")
    status = status or ("ok" if sql.strip() else "empty")
    return {
        "sample_id": sample["sample_id"],
        "db_id": sample["db_id"],
        "raw_output": raw_output,
        "sql": sql,
        "status": status,
        "error": error,
    }


def generation_batch(samples: list[dict], generate, *, oom_exception: type[Exception]):
    """Return one record per sample, marking the whole batch failed on an exception.

    Args:
        samples: List of dictionaries containing sample_id and db_id.
        generate: Zero-argument callable returning one decoded string per sample,
            in the same order as samples.
        oom_exception: Keyword-only exception class used to identify OOM failures,
            such as torch.cuda.OutOfMemoryError; pass the class, not an instance.
    Returns:
        A list of prediction dictionaries. A generation/count/conversion failure
        marks every sample 'oom' or 'generation_error' instead of dropping rows.
    """
    try:
        outputs = list(generate())
        if len(outputs) != len(samples):
            raise ValueError("Generation output count does not match the batch")
        return [
            prediction_record(sample, output)
            for sample, output in zip(samples, outputs)
        ]
    except Exception as exc:
        status = "oom" if isinstance(exc, oom_exception) else "generation_error"
        return [
            prediction_record(
                sample, status=status, error=f"{type(exc).__name__}: {exc}"
            )
            for sample in samples
        ]


def validate_and_order_predictions(
    manifest: list[dict], records: list[dict]
) -> list[dict]:
    """Validate record fields, IDs, and databases; return copies in manifest order.

    Args:
        manifest: Ordered list of reference sample dictionaries.
        records: Prediction dictionaries, possibly in a different order.
    Returns:
        Shallow copies of records ordered by the manifest's sample IDs.
    Raises:
        ValueError: IDs are missing/extra/duplicated, databases differ, or record
            fields and statuses violate the prediction contract.
    """

    expected = {sample["sample_id"]: sample for sample in manifest}

    if not expected or len(expected) != len(manifest):
        raise ValueError("Manifest is empty or contains duplicate IDs")

    by_id = {}

    for record in records:
        sample_id = _text(record, "sample_id")

        if sample_id in by_id:
            raise ValueError(f"Duplicate prediction ID: {sample_id}")

        if sample_id not in expected:
            raise ValueError(f"Unexpected prediction ID: {sample_id}")

        if _text(record, "db_id") != expected[sample_id]["db_id"]:
            raise ValueError(f"Database mismatch for {sample_id}")

        _text(record, "raw_output")
        sql = _text(record, "sql")

        status = _text(record, "status")
        if status not in STATUSES:
            raise ValueError(f"Invalid generation status for {sample_id}: {status}")

        if status == "ok" and not sql.strip():
            raise ValueError(f"Blank SQL incorrectly marked ok for {sample_id}")

        if status == "empty" and sql.strip():
            raise ValueError(f"Nonblank SQL incorrectly marked empty for {sample_id}")

        if record.get("error") is not None and not isinstance(record["error"], str):
            raise ValueError(f"Invalid error field for {sample_id}")

        by_id[sample_id] = record

    missing = expected.keys() - by_id.keys()
    if missing:
        raise ValueError(
            f"Missing {len(missing)} prediction IDs, samples: {sorted(missing)[:3]}"
        )

    return [dict(by_id[sample["sample_id"]]) for sample in manifest]


@dataclass(frozen=True)
class EvaluationRun:
    """Hold a published run's directory and reference manifest.

    Args:
        path: Path to the published run directory.
        manifest: Ordered list of reference sample dictionaries.
    """

    path: Path
    manifest: list[dict]

    @property
    def gold_path(self) -> Path:
        """Return path / 'gold.sql' as a Path without accessing the file.

        Access as eval_run.gold_path, without parentheses; self is implicit.
        """
        return self.path / "gold.sql"


def create_evaluation_run(
    dataset, *, output_dir="artifacts/evaluation", dataset_id: str, split: str
) -> EvaluationRun:
    """Atomically publish a unique run containing its manifest, gold SQL, and metadata.

    Args:
        dataset: Evaluation rows accepted by build_manifest; a Hugging Face
            Dataset also provides the optional snapshot fingerprint.
        output_dir: Keyword-only parent directory, default 'artifacts/evaluation';
            relative paths are resolved against PROJECT_ROOT.
        dataset_id: Required keyword-only source identifier saved in metadata.
        split: Required keyword-only split label saved in metadata and sample IDs.
    Returns:
        An EvaluationRun holding the published directory and reference manifest.
    Raises:
        ValueError: The dataset fails manifest validation.
        OSError: Creating, writing, or publishing the run fails.
    """

    manifest = build_manifest(dataset, split=split)

    root = Path(output_dir)

    if not root.is_absolute():
        root = PROJECT_ROOT / root
    root.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:10]

    destination = root / run_id

    with tempfile.TemporaryDirectory(prefix=".pending-", dir=root) as temporary_dir:
        stage = Path(temporary_dir)
        _write_jsonl(stage / "manifest.jsonl", manifest)

        with (stage / "gold.sql").open("w", encoding="utf-8", newline="\n") as stream:
            for sample in manifest:
                stream.write(
                    to_single_line_sql(sample["gold_sql"])
                    + "\t"
                    + sample["db_id"]
                    + "\n"
                )

        metadata = {
            "schema_version": 1,
            "dataset_id": dataset_id,
            "split": split,
            "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
            "sample_count": len(manifest),
            "manifest_sha256": _sha256(stage / "manifest.jsonl"),
            "gold_sha256": _sha256(stage / "gold.sql"),
        }

        (stage / "dataset.json").write_text(_json(metadata) + "\n", encoding="utf-8")
        stage.rename(destination)

    return EvaluationRun(destination, manifest)


def attach_sample_ids(dataset, eval_run: EvaluationRun):
    """Return a dataset with manifest IDs attached to its rows after checking counts.

    Args:
        dataset: Hugging Face Dataset in the same row order used to build eval_run.
        eval_run: EvaluationRun containing the reference manifest.
    Returns:
        A Dataset with the added _evaluation_sample_id column; the input is unchanged.
    Raises:
        ValueError: Dataset and manifest lengths differ.
    """
    if len(dataset) != len(eval_run.manifest):
        raise ValueError("Dataset count differs from the manifest")
    return dataset.add_column(
        "_evaluation_sample_id", [sample["sample_id"] for sample in eval_run.manifest]
    )


@dataclass(frozen=True)
class PredictionArtifacts:
    """Hold a model export's paths, sample count, and status counts.

    Args:
        path: Path to the published model/step export directory.
        gold_path: Path to the shared reference gold SQL file.
        sample_count: Number of samples expected by the evaluator.
        status_counts: Dictionary mapping each supported status to its count.

    The dataclass generates the constructor; fields cannot be reassigned.
    """

    path: Path
    gold_path: Path
    sample_count: int
    status_counts: dict[str, int]

    @property
    def prediction_path(self) -> Path:
        """Return path / 'predictions.sql' as a Path without accessing the file.

        Access as artifacts.prediction_path, without parentheses; self is implicit.
        """
        return self.path / "predictions.sql"


def export_predictions(
    eval_run: EvaluationRun, records: list[dict], *, model_key: str, model_id: str = ""
) -> PredictionArtifacts:
    """Validate and order records, then atomically publish JSONL, SQL, and export metadata.

    Preserve raw outputs and export failed or unrepresentable SQL as a nonblank
    failure marker. Reject changed reference files and existing destinations.

    Args:
        eval_run: EvaluationRun containing the saved manifest and shared gold SQL.
        records: List of prediction dictionaries, in any order.
        model_key: Required keyword-only name for the export subdirectory.
        model_id: Keyword-only model identifier saved in metadata; defaults to ''.
    Returns:
        PredictionArtifacts with the published paths, sample count, and statuses.
    Raises:
        ValueError: Reference integrity, record validation, or model_key is invalid.
        FileExistsError: The model/step export directory already exists.
        OSError: Reading, writing, or publishing artifacts fails.
    """

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", model_key):
        raise ValueError("model_key must be a single safe directory name")

    destination = eval_run.path / model_key

    if destination.exists():
        raise FileExistsError(f"Evaluation artifacts already exist: {destination}")

    metadata = json.loads((eval_run.path / "dataset.json").read_text(encoding="utf-8"))

    if _sha256(eval_run.path / "manifest.jsonl") != metadata["manifest_sha256"]:
        raise ValueError("Saved evaluation manifest has changed")

    if _sha256(eval_run.gold_path) != metadata["gold_sha256"]:
        raise ValueError("Saved evaluation gold file has changed")

    saved_manifest = [
        json.loads(line)
        for line in (eval_run.path / "manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    if saved_manifest != eval_run.manifest:
        raise ValueError("In-memory manifest differs from the saved snapshot")

    ordered = validate_and_order_predictions(eval_run.manifest, records)
    exported = []

    for record in ordered:
        line = FAILURE_SQL
        if record["status"] == "ok":
            try:
                line = to_single_line_sql(record["sql"])
            except ValueError as exc:
                record["generation_status"] = record["status"]
                record["status"] = "export_error"
                record["error"] = str(exc)
        exported.append(line)

    counter = Counter(record["status"] for record in ordered)
    counts = {status: counter[status] for status in sorted(STATUSES)}

    with tempfile.TemporaryDirectory(
        prefix=".pending-", dir=eval_run.path
    ) as temporary_dir:
        stage = Path(temporary_dir)

        _write_jsonl(stage / "predictions.jsonl", ordered)

        (stage / "predictions.sql").write_text(
            "\n".join(exported) + "\n", encoding="utf-8", newline="\n"
        )

        export_metadata = {
            "schema_version": 1,
            "model_id": model_id,
            "sample_count": len(ordered),
            "status_counts": counts,
            "manifest_sha256": metadata["manifest_sha256"],
            "gold_sha256": metadata["gold_sha256"],
            "predictions_sha256": _sha256(stage / "predictions.sql"),
        }

        (stage / "export.json").write_text(
            _json(export_metadata) + "\n", encoding="utf-8"
        )

        stage.rename(destination)

    return PredictionArtifacts(destination, eval_run.gold_path, len(ordered), counts)


def evaluate_artifacts(
    artifacts: PredictionArtifacts, *, evaluate, db_dir, etype, kmaps
):
    """Verify input hashes/counts, run the scorer, and reject a changed denominator.

    Args:
        artifacts: PredictionArtifacts returned by export_predictions.
        evaluate: Keyword-only upstream scorer callable accepting gold path,
            prediction path, database directory, evaluation type, and key maps.
        db_dir: Keyword-only root directory of Spider databases, as a string or Path.
        etype: Keyword-only scoring mode: 'all', 'exec', or 'match'.
        kmaps: Keyword-only database-to-foreign-key mappings used by the scorer.
    Returns:
        The scorer's scores dictionary after verifying its overall sample count.
    Raises:
        ValueError: Input hashes/line counts changed or the scorer's count differs.
        OSError: Reading exported inputs fails. Scorer exceptions also propagate.
    """

    metadata = json.loads((artifacts.path / "export.json").read_text(encoding="utf-8"))

    for path, key in (
        (artifacts.gold_path, "gold_sha256"),
        (artifacts.prediction_path, "predictions_sha256"),
    ):
        if _sha256(path) != metadata[key]:
            raise ValueError(f"Evaluation input changed after export: {path}")
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) != artifacts.sample_count or any(
            not line.strip() for line in lines
        ):
            raise ValueError(f"Evaluation input has missing/extra/blank lines: {path}")

    scores = evaluate(
        str(artifacts.gold_path),
        str(artifacts.prediction_path),
        str(db_dir),
        etype,
        kmaps,
    )

    if (
        not isinstance(scores, dict)
        or scores.get("all", {}).get("count") != artifacts.sample_count
    ):
        raise ValueError(
            f"Evaluator denominator differs from expected {artifacts.sample_count}"
        )

    return scores
