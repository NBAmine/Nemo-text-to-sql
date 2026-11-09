"""Resilient loading, model publication, and checkpoint downloads from the Hub."""

import logging
from pathlib import Path
import ssl
from typing import Callable, TypeVar

import httpx
from huggingface_hub import HfApi, snapshot_download
from requests.exceptions import (
    ChunkedEncodingError,
    ConnectionError as RequestsConnectionError,
    SSLError,
    Timeout,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")


def _transient_hub_error(error: BaseException) -> bool:
    """Recognize transient requests/httpx failures, including loader wrappers."""
    seen = set()
    transient = False
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        # A certificate/configuration failure will not heal with another attempt.
        if isinstance(error, (MemoryError, PermissionError, ssl.SSLError, SSLError)):
            return False
        status = getattr(getattr(error, "response", None), "status_code", None)
        if isinstance(status, int):
            return status in (408, 429) or 500 <= status < 600
        transient = transient or isinstance(
            error,
            (
                RequestsConnectionError, Timeout, ChunkedEncodingError,
                httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError,
                ConnectionError, TimeoutError,
            ),
        )
        # Transformers/PEFT can wrap a failed download in OSError/ValueError.
        error = error.__cause__ or (
            error.__context__ if not error.__suppress_context__ else None
        )
    return transient


def retry_hub_io(*, attempts: int = 3, min_wait: int = 2, max_wait: int = 30):
    """Retry a single Hub operation with bounded backoff and the original error."""
    return retry(
        retry=retry_if_exception(_transient_hub_error),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        reraise=True,
    )


@retry_hub_io()
def load_with_retry(loader: Callable[..., T], *args, **kwargs) -> T:
    """Retry a model, tokenizer, adapter, or dataset loader independently."""
    return loader(*args, **kwargs)


@retry_hub_io()
def push_to_hub_with_retry(artifact, repo_id: str, *, private: bool):
    """Retry publication without reloading or merging the model again."""
    return artifact.push_to_hub(repo_id, private=private)


@retry_hub_io()
def upload_folder_with_retry(folder_path: str | Path, repo_id: str):
    """Upload an existing model export without serializing its weights again."""
    return HfApi().upload_folder(
        folder_path=folder_path, repo_id=repo_id, repo_type="model",
    )


@retry_hub_io()
def download_hub_checkpoint(repo_id: str, *, api: HfApi | None = None) -> str | None:
    """Download last-checkpoint at one immutable revision.

    None means an accessible repository has no checkpoint files. Missing or
    inaccessible repositories and failed downloads raise; they are never
    interpreted as permission to start a new training trajectory.
    """
    api = api if api is not None else HfApi()
    revision = api.repo_info(repo_id=repo_id, repo_type="model").sha
    if not isinstance(revision, str) or not revision:
        raise RuntimeError(f"Hub did not provide a commit for '{repo_id}'.")
    files = api.list_repo_files(repo_id=repo_id, repo_type="model", revision=revision)
    checkpoint_files = [name for name in files if name.startswith("last-checkpoint/")]
    if not checkpoint_files:
        return None
    if "last-checkpoint/trainer_state.json" not in checkpoint_files:
        raise RuntimeError(f"Incomplete Hub checkpoint in '{repo_id}': missing trainer_state.json.")
    snapshot = snapshot_download(
        repo_id=repo_id, repo_type="model", revision=revision,
        allow_patterns=["last-checkpoint/*", "attempt-completions/*.json"], token=api.token,
    )
    checkpoint = Path(snapshot) / "last-checkpoint"
    if not checkpoint.is_dir():
        raise RuntimeError(f"Downloaded checkpoint directory is missing: {checkpoint}")
    return str(checkpoint)


@retry_hub_io()
def publish_hub_completion(repo_id: str, record_path: str, *, attempt_id: str) -> None:
    """Publish a small, attempt-specific completion record after final exports."""
    HfApi().upload_file(
        repo_id=repo_id, repo_type="model", path_or_fileobj=record_path,
        path_in_repo=f"attempt-completions/{attempt_id}.json",
        commit_message=f"Record completion of training attempt {attempt_id}",
    )


@retry_hub_io()
def create_model_repo(repo_id: str, *, private: bool, api: HfApi | None = None) -> None:
    """Create a model repo if needed and warn about unexpected visibility.

    Creation flags do not change an existing repo's visibility. Never upload
    when the Hub cannot report its visibility, and never change
    an existing repo's visibility implicitly.
    """
    if not isinstance(private, bool):
        raise TypeError("Model repository privacy must be an explicit boolean.")

    if api is None:
        api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    actual_private = api.repo_info(repo_id=repo_id, repo_type="model").private
    if not isinstance(actual_private, bool):
        raise RuntimeError(
            f"Cannot verify visibility for '{repo_id}': "
            f"the Hub reported private={actual_private!r}."
        )
    if actual_private is not private:
        requested = "private" if private else "public"
        logger.warning(
            "Repository '%s' was requested as %s, but the Hub reports "
            "private=%s. Continuing publication with the existing visibility.",
            repo_id, requested, actual_private,
        )
