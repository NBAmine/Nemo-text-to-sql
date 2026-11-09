"""Shared application logging; call once from each script's entry point."""

import logging
from enum import IntEnum
from logging.handlers import RotatingFileHandler
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_FORMAT = "%(asctime)s [%(name)s][%(levelname)s] %(message)s"


class LogLevel(IntEnum):
    NOTSET = logging.NOTSET
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


def normalize_logging_level(value: LogLevel | int | str) -> LogLevel:
    try:
        if isinstance(value, str):
            return LogLevel[value.upper()]
        return LogLevel(value)
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid logging level: {value!r}") from exc


def setup_logging(
    level: int | str = logging.INFO,
    *,
    httpx_level: int | str = logging.WARNING,
    httpcore_level: int | str = logging.WARNING,
    log_dir: str | Path | None = None,
    filename: str = "nemo.log",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    rank: int | None = None,
) -> Path:
    """Configure console and UTF-8 rotating file logs, returning the log path.

    Levels accept logging constants or uppercase names such as ``"DEBUG"``.
    Relative directories resolve against the project root, including under Hydra.
    Existing root handlers are replaced so repeated calls do not duplicate logs.
    Pass the global distributed rank to give each worker its own rotating file.
    Independently running jobs should also use distinct filenames/directories.
    """

    if max_bytes <= 0 or backup_count < 1:
        raise ValueError("max_bytes and backup_count must be positive")
    if not filename or Path(filename).name != filename:
        raise ValueError("filename must be a file name; use log_dir for directories")
    # Validate all levels before changing the active logging configuration.
    level = normalize_logging_level(level)
    httpx_level = normalize_logging_level(httpx_level)
    httpcore_level = normalize_logging_level(httpcore_level)

    directory = Path(log_dir) if log_dir is not None else Path("logs")
    if not directory.is_absolute():
        directory = PROJECT_ROOT / directory
    directory.mkdir(parents=True, exist_ok=True)
    log_file_path = directory / filename

    log_format = LOG_FORMAT

    if rank is not None:
        log_file_path = log_file_path.with_name(
            f"{log_file_path.stem}.rank{rank}{log_file_path.suffix}"
        )
        log_format = f"%(asctime)s [rank{rank}][%(name)s][%(levelname)s] %(message)s"

    file_handler = RotatingFileHandler(
        log_file_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[logging.StreamHandler(), file_handler],
        force=True,
    )
    logging.getLogger("httpx").setLevel(httpx_level)
    logging.getLogger("httpcore").setLevel(httpcore_level)
    return log_file_path
