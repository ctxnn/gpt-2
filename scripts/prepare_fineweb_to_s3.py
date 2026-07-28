#!/usr/bin/env python3
"""Prepare FineWeb-Edu shards and upload each verified shard to S3."""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing as mp
import os
import random
import re
import shutil
import subprocess
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence
from urllib.parse import urlparse

import boto3
import numpy as np
from boto3.s3.transfer import TransferConfig
from botocore.config import Config

from fineweb import (
    DATASET_ID,
    DEFAULT_DATASET_CONFIG,
    DEFAULT_SHARD_SIZE,
    tokenize,
    tokenizer,
    validate_shard_filenames,
)

LOGGER = logging.getLogger("fineweb_s3")
TOKENIZER_NAME = "gpt2"
SHARD_FORMAT = "NumPy .npy"
DTYPE = np.dtype(np.uint16)
REQUIRED_ENVIRONMENT = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
    "AWS_ENDPOINT_URL",
    "GMN_DATA_BUCKET",
    "GMN_DATA_PREFIX",
)
SHARD_PREFIX = "shards"
METADATA_PREFIX = "metadata"
STATUS_PREFIX = "status"
PROBE_PREFIX = "probes"
PROGRESS_NAME = "progress.json"
MANIFEST_NAME = "dataset_manifest.json"
CHECKSUMS_NAME = "checksums.sha256"
REPORT_NAME = "preparation_report.md"
COMPLETE_NAME = "COMPLETE"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
GIT_SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")
GIB = 1024**3
# FineWeb is loaded through the ordinary cached Hugging Face path.  This is a
# conservative allowance for the compressed source files and their metadata;
# completed token shards are still removed locally after S3 verification.
DATASET_CACHE_GIB = 48.0
DEPENDENCY_INSTALL_HEADROOM_GIB = 3.0
UPLOAD_HEADROOM_GIB = 0.5
METADATA_HEADROOM_GIB = 0.1
SAFETY_MARGIN_GIB = 2.0
HF_LOAD_ATTEMPTS = 5
HF_RETRY_BACKOFF_SECONDS = 2.0
HF_HUB_DOWNLOAD_TIMEOUT_SECONDS = 120
HF_HUB_ETAG_TIMEOUT_SECONDS = 30


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, repr=False)
class Settings:
    access_key_id: str
    secret_access_key: str
    region: str
    default_region: str
    endpoint_url: str
    bucket: str
    prefix: str
    preparation_run_id: str
    job_id: str | None
    output_dir: Path | None
    result_path: Path | None

    @classmethod
    def from_environment(cls, environment: Mapping[str, str] | None = None) -> "Settings":
        source = os.environ if environment is None else environment
        missing = [name for name in REQUIRED_ENVIRONMENT if not source.get(name)]
        if missing:
            raise ValueError(
                "missing required environment variables: " + ", ".join(sorted(missing))
            )
        prefix = source["GMN_DATA_PREFIX"].strip("/")
        if not prefix:
            raise ValueError("GMN_DATA_PREFIX must not be empty")
        supplied_run_id = source.get("GMN_PREPARATION_RUN_ID", "").strip()
        preparation_run_id = supplied_run_id or f"prep-{uuid.uuid4().hex}"
        if RUN_ID_PATTERN.fullmatch(preparation_run_id) is None:
            raise ValueError("GMN_PREPARATION_RUN_ID contains unsafe characters")
        raw_job_id = source.get("GIVEMEANODE_JOB_ID", "").strip()
        job_id = None if raw_job_id.lower() in {"", "null", "none"} else raw_job_id
        endpoint = source["AWS_ENDPOINT_URL"]
        parsed = urlparse(endpoint)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("AWS_ENDPOINT_URL must be an HTTPS URL with a hostname")
        return cls(
            access_key_id=source["AWS_ACCESS_KEY_ID"],
            secret_access_key=source["AWS_SECRET_ACCESS_KEY"],
            region=source["AWS_REGION"],
            default_region=source["AWS_DEFAULT_REGION"],
            endpoint_url=endpoint,
            bucket=source["GMN_DATA_BUCKET"],
            prefix=prefix,
            preparation_run_id=preparation_run_id,
            job_id=job_id,
            output_dir=Path(source["GMN_OUTPUT_DIR"])
            if source.get("GMN_OUTPUT_DIR")
            else None,
            result_path=Path(source["GMN_RESULT_PATH"])
            if source.get("GMN_RESULT_PATH")
            else None,
        )

    def key(self, *parts: str) -> str:
        suffix = "/".join(part.strip("/") for part in parts if part)
        return f"{self.prefix}/{suffix}" if suffix else self.prefix

    @property
    def endpoint_hostname(self) -> str:
        return urlparse(self.endpoint_url).hostname or ""


@dataclass(frozen=True)
class DiskBudget:
    """Conservative peak-disk estimate for cached, progressive preparation.

    FineWeb is downloaded through the ordinary Hugging Face cache. The budget
    accounts for that source cache, dependency installation, one uint16 token
    buffer, one shard during atomic write/upload, one validation download,
    upload overhead, and a safety margin. Verified shards are deleted after
    remote size/SHA verification, so completed shards do not accumulate.
    """

    source_cache_gib: float
    dependency_headroom_gib: float
    active_token_buffer_gib: float
    in_progress_shard_gib: float
    upload_headroom_gib: float
    validation_download_gib: float
    metadata_headroom_gib: float
    safety_margin_gib: float
    required_gib: float


@dataclass(frozen=True)
class DiskFacts:
    filesystem: str
    mount_path: str
    path: str
    total_bytes: int
    used_bytes: int
    available_bytes: int
    total_gib: float
    used_gib: float
    available_gib: float
    budget: DiskBudget


def calculate_disk_budget(shard_size: int = DEFAULT_SHARD_SIZE) -> DiskBudget:
    shard_gib = (shard_size * DTYPE.itemsize) / GIB
    required = (
        DATASET_CACHE_GIB
        + DEPENDENCY_INSTALL_HEADROOM_GIB
        + shard_gib
        + shard_gib
        + UPLOAD_HEADROOM_GIB
        + shard_gib
        + METADATA_HEADROOM_GIB
        + SAFETY_MARGIN_GIB
    )
    return DiskBudget(
        source_cache_gib=DATASET_CACHE_GIB,
        dependency_headroom_gib=DEPENDENCY_INSTALL_HEADROOM_GIB,
        active_token_buffer_gib=shard_gib,
        in_progress_shard_gib=shard_gib,
        upload_headroom_gib=UPLOAD_HEADROOM_GIB,
        validation_download_gib=shard_gib,
        metadata_headroom_gib=METADATA_HEADROOM_GIB,
        safety_margin_gib=SAFETY_MARGIN_GIB,
        required_gib=round(required, 3),
    )


def configure_logging() -> None:
    logging.disable(logging.DEBUG)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("s3transfer").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def create_s3_client(settings: Settings):
    return boto3.client(
        "s3",
        endpoint_url=settings.endpoint_url,
        region_name=settings.region,
        aws_access_key_id=settings.access_key_id,
        aws_secret_access_key=settings.secret_access_key,
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
        ),
    )


def git_sha() -> str | None:
    """Return the authoritative source SHA without requiring git metadata."""

    explicit = os.environ.get("GMN_SOURCE_GIT_SHA", "").strip()
    if explicit:
        if GIT_SHA_PATTERN.fullmatch(explicit) is None:
            raise ValueError("GMN_SOURCE_GIT_SHA must be a 40-character hexadecimal SHA")
        return explicit.lower()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    discovered = result.stdout.strip()
    return discovered or None


def filesystem_identity(path: Path) -> tuple[str, str]:
    try:
        result = subprocess.run(
            ["df", "-P", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
        fields = result.stdout.strip().splitlines()[-1].split()
        if len(fields) >= 6:
            return fields[0], " ".join(fields[5:])
    except (OSError, subprocess.CalledProcessError, IndexError):
        pass
    return "unknown", str(path)


def select_work_base(candidates: Sequence[Path] | None = None) -> Path:
    options = list(candidates or (Path.cwd(), Path("/tmp")))
    existing = [path for path in options if path.exists()]
    if not existing:
        raise RuntimeError("no usable work filesystem was found")
    return max(existing, key=lambda path: shutil.disk_usage(path).free)


def require_free_disk(
    path: Path,
    *,
    shard_size: int = DEFAULT_SHARD_SIZE,
) -> DiskFacts:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    budget = calculate_disk_budget(shard_size)
    filesystem, mount_path = filesystem_identity(path)
    facts = DiskFacts(
        filesystem=filesystem,
        mount_path=mount_path,
        path=str(path),
        total_bytes=usage.total,
        used_bytes=usage.used,
        available_bytes=usage.free,
        total_gib=round(usage.total / GIB, 3),
        used_gib=round(usage.used / GIB, 3),
        available_gib=round(usage.free / GIB, 3),
        budget=budget,
    )
    LOGGER.info(
        "disk filesystem=%s mount=%s path=%s available_gib=%.3f required_gib=%.3f "
        "safety_margin_gib=%.3f",
        facts.filesystem,
        facts.mount_path,
        facts.path,
        facts.available_gib,
        facts.budget.required_gib,
        facts.budget.safety_margin_gib,
    )
    if usage.free < budget.required_gib * GIB:
        raise RuntimeError(
            "insufficient local disk for calculated preparation budget"
        )
    return facts


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, canonical_json_bytes(value))


def atomic_write_npy(path: Path, tokens: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, tokens, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def configure_huggingface_cache(cache_dir: Path | str) -> Path:
    """Set deterministic Hugging Face cache and network-timeout locations."""

    cache_root = Path(cache_dir).expanduser()
    datasets_cache = cache_root / "datasets"
    hub_cache = cache_root / "hub"
    cache_root.mkdir(parents=True, exist_ok=True)
    datasets_cache.mkdir(parents=True, exist_ok=True)
    hub_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ.setdefault(
        "HF_HUB_DOWNLOAD_TIMEOUT", str(HF_HUB_DOWNLOAD_TIMEOUT_SECONDS)
    )
    os.environ.setdefault(
        "HF_HUB_ETAG_TIMEOUT", str(HF_HUB_ETAG_TIMEOUT_SECONDS)
    )
    return cache_root


def _safe_error_detail(error: BaseException) -> str:
    detail = " ".join(str(error).split())
    for variable_name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        secret = os.environ.get(variable_name)
        if secret:
            detail = detail.replace(secret, "<redacted>")
    return (detail or type(error).__name__)[:500]


def _exception_chain(error: BaseException) -> Iterator[BaseException]:
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _is_transient_dataset_error(error: BaseException) -> bool:
    """Recognize transport/service failures without retrying auth/config errors."""

    permanent_markers = (
        "401",
        "403",
        "unauthorized",
        "forbidden",
        "authentication",
        "invalid token",
        "unknown configuration",
        "invalid configuration",
        "revision",
        "not found",
        "does not exist",
    )
    transient_markers = (
        "timeout",
        "timed out",
        "connection reset",
        "connection aborted",
        "connection refused",
        "temporary failure",
        "temporarily unavailable",
        "incomplete read",
        "incomplete download",
        "server error",
        "rate limit",
        "too many requests",
        "remote end closed",
        "protocol error",
        "chunkedencodingerror",
        "502",
        "503",
        "504",
    )
    try:
        import requests
        from urllib3 import exceptions as urllib3_exceptions

        transport_types: tuple[type[BaseException], ...] = (
            TimeoutError,
            ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
            urllib3_exceptions.ProtocolError,
            urllib3_exceptions.ReadTimeoutError,
            urllib3_exceptions.MaxRetryError,
        )
    except ImportError:
        transport_types = (TimeoutError, ConnectionError)

    for candidate in _exception_chain(error):
        message = str(candidate).lower()
        response = getattr(candidate, "response", None)
        status = getattr(response, "status_code", None)
        if status is not None:
            if int(status) in {408, 425, 429, 500, 502, 503, 504}:
                return True
            if int(status) in {401, 403, 404}:
                return False
        if any(marker in message for marker in permanent_markers):
            return False
        if isinstance(candidate, transport_types):
            return True
        if any(marker in message for marker in transient_markers):
            return True
        # Datasets can wrap a transport failure in a RuntimeError without
        # preserving a typed cause. Retry that bounded, otherwise-unknown
        # wrapper, while the permanent markers above remain non-retryable.
        if isinstance(candidate, RuntimeError) and not message:
            return True
    return False


def load_and_validate_shard(
    path: Path,
    *,
    shard_size: int,
    allow_partial: bool,
    vocabulary_size: int,
) -> np.ndarray:
    tokens = np.load(path, allow_pickle=False, mmap_mode="r")
    if tokens.dtype != DTYPE:
        raise ValueError(f"{path.name} has dtype {tokens.dtype}, expected uint16")
    if tokens.ndim != 1 or tokens.size <= 0:
        raise ValueError(f"{path.name} must contain a non-empty one-dimensional array")
    if not allow_partial and tokens.size != shard_size:
        raise ValueError(
            f"{path.name} has {tokens.size} tokens, expected {shard_size}"
        )
    if tokens.size > shard_size:
        raise ValueError(f"{path.name} exceeds configured shard size")
    if int(tokens.max()) >= vocabulary_size:
        raise ValueError(f"{path.name} contains a token outside the GPT-2 vocabulary")
    return tokens


def put_json(client: Any, settings: Settings, key: str, value: Any) -> bytes:
    content = canonical_json_bytes(value)
    client.put_object(
        Bucket=settings.bucket,
        Key=key,
        Body=content,
        ContentType="application/json",
        Metadata={"sha256": sha256_bytes(content)},
    )
    return content


def get_object_bytes(client: Any, settings: Settings, key: str) -> bytes:
    response = client.get_object(Bucket=settings.bucket, Key=key)
    body = response["Body"]
    return body.read() if hasattr(body, "read") else bytes(body)


def object_exists(client: Any, settings: Settings, key: str) -> bool:
    try:
        client.head_object(Bucket=settings.bucket, Key=key)
    except Exception as error:
        response = getattr(error, "response", {})
        code = str(response.get("Error", {}).get("Code", ""))
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code in {"404", "NoSuchKey", "NotFound"} or status == 404:
            return False
        raise
    return True


def list_objects(client: Any, settings: Settings, prefix: str) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    continuation: str | None = None
    while True:
        request: dict[str, Any] = {
            "Bucket": settings.bucket,
            "Prefix": prefix,
        }
        if continuation:
            request["ContinuationToken"] = continuation
        response = client.list_objects_v2(**request)
        objects.extend(response.get("Contents", []))
        if not response.get("IsTruncated"):
            return objects
        continuation = response["NextContinuationToken"]


def storage_probe(client: Any, settings: Settings) -> None:
    probe_key = settings.key(
        PROBE_PREFIX,
        settings.preparation_run_id,
        "probe.bin",
    )
    probe = os.urandom(64)
    try:
        client.put_object(Bucket=settings.bucket, Key=probe_key, Body=probe)
        downloaded = get_object_bytes(client, settings, probe_key)
        if downloaded != probe:
            raise RuntimeError("S3 probe byte verification failed")
    finally:
        try:
            client.delete_object(Bucket=settings.bucket, Key=probe_key)
        except Exception:
            LOGGER.warning("S3 probe cleanup did not complete")


def parse_remote_shard_name(name: str) -> tuple[str, int]:
    path = Path(name)
    parts = path.name.removesuffix(".npy").split("_")
    if (
        len(parts) != 3
        or parts[0] != "edufineweb"
        or parts[1] not in {"train", "val"}
        or len(parts[2]) != 6
        or not parts[2].isdigit()
        or path.suffix != ".npy"
    ):
        raise ValueError(f"malformed or unexpected remote shard filename: {path.name}")
    return parts[1], int(parts[2])


def load_progress(client: Any, settings: Settings) -> dict[str, Any]:
    key = settings.key(METADATA_PREFIX, PROGRESS_NAME)
    if not object_exists(client, settings, key):
        return {
            "schema_version": 1,
            "source_dataset": DATASET_ID,
            "dataset_configuration": DEFAULT_DATASET_CONFIG,
            "shard_size": DEFAULT_SHARD_SIZE,
            "updated_at": utc_now(),
            "complete": False,
            "shards": [],
        }
    return json.loads(get_object_bytes(client, settings, key))


def progress_by_filename(progress: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        item["filename"]: dict(item)
        for item in progress.get("shards", [])
        if isinstance(item, Mapping) and isinstance(item.get("filename"), str)
    }


def progress_record_matches_head(
    record: Mapping[str, Any] | None,
    *,
    split: str,
    index: int,
    key: str,
    remote_bytes: int,
    metadata: Mapping[str, str],
    shard_size: int,
) -> bool:
    if record is None:
        return False
    try:
        token_count = int(record.get("token_count", 0))
        expected_sha = str(record.get("sha256", ""))
        return (
            record.get("split") == split
            and int(record.get("index", -1)) == index
            and record.get("remote_key") == key
            and 0 < token_count <= shard_size
            and int(record.get("local_bytes", -1)) == remote_bytes
            and int(record.get("remote_bytes", -1)) == remote_bytes
            and bool(expected_sha)
            and metadata.get("sha256") == expected_sha
            and metadata.get("token-count") == str(token_count)
            and metadata.get("split") == split
            and metadata.get("shard-index") == str(index)
        )
    except (TypeError, ValueError):
        return False


def upload_progress(
    client: Any,
    settings: Settings,
    progress: dict[str, Any],
) -> None:
    progress["updated_at"] = utc_now()
    progress["shards"] = sorted(
        progress["shards"], key=lambda item: int(item["index"])
    )
    put_json(
        client,
        settings,
        settings.key(METADATA_PREFIX, PROGRESS_NAME),
        progress,
    )


def inspect_downloaded_shard(
    client: Any,
    settings: Settings,
    key: str,
    destination: Path,
    *,
    shard_size: int,
    allow_partial: bool,
    vocabulary_size: int,
) -> tuple[int, int, str]:
    client.download_file(settings.bucket, key, str(destination))
    tokens = load_and_validate_shard(
        destination,
        shard_size=shard_size,
        allow_partial=allow_partial,
        vocabulary_size=vocabulary_size,
    )
    return int(tokens.size), destination.stat().st_size, sha256_file(destination)


def verify_existing_shards(
    client: Any,
    settings: Settings,
    progress: dict[str, Any],
    work_dir: Path,
    *,
    shard_size: int,
    vocabulary_size: int,
) -> dict[str, dict[str, Any]]:
    records = progress_by_filename(progress)
    verified: dict[str, dict[str, Any]] = {}
    remote_objects = list_objects(client, settings, settings.key(SHARD_PREFIX) + "/")
    seen_indices: set[tuple[str, int]] = set()
    for remote in remote_objects:
        key = remote["Key"]
        filename = Path(key).name
        split, index = parse_remote_shard_name(filename)
        identity = (split, index)
        if identity in seen_indices:
            raise ValueError(f"duplicate remote shard identity: {split} {index}")
        seen_indices.add(identity)
        record = records.get(filename)
        head = client.head_object(Bucket=settings.bucket, Key=key)
        remote_bytes = int(head["ContentLength"])
        metadata = {
            str(k).lower(): str(v) for k, v in head.get("Metadata", {}).items()
        }
        expected_sha = str(record.get("sha256", "")) if record else ""
        if progress_record_matches_head(
            record,
            split=split,
            index=index,
            key=key,
            remote_bytes=remote_bytes,
            metadata=metadata,
            shard_size=shard_size,
        ):
            verified[filename] = record
            continue

        local = work_dir / f"inspect-{filename}"
        try:
            token_count, local_bytes, actual_sha = inspect_downloaded_shard(
                client,
                settings,
                key,
                local,
                shard_size=shard_size,
                allow_partial=True,
                vocabulary_size=vocabulary_size,
            )
            if expected_sha and actual_sha != expected_sha:
                continue
            verified[filename] = {
                "filename": filename,
                "split": split,
                "index": index,
                "token_count": token_count,
                "local_bytes": local_bytes,
                "remote_bytes": remote_bytes,
                "sha256": actual_sha,
                "etag": str(head.get("ETag", "")).strip('"'),
                "remote_key": key,
                "upload_timestamp": record.get("upload_timestamp", utc_now())
                if record
                else utc_now(),
            }
        except (OSError, ValueError):
            continue
        finally:
            local.unlink(missing_ok=True)
    progress["shards"] = list(verified.values())
    return verified


def upload_verified_shard(
    client: Any,
    settings: Settings,
    path: Path,
    *,
    split: str,
    index: int,
    shard_size: int,
    allow_partial: bool,
    vocabulary_size: int,
) -> dict[str, Any]:
    tokens = load_and_validate_shard(
        path,
        shard_size=shard_size,
        allow_partial=allow_partial,
        vocabulary_size=vocabulary_size,
    )
    digest = sha256_file(path)
    local_bytes = path.stat().st_size
    key = settings.key(SHARD_PREFIX, path.name)
    client.upload_file(
        str(path),
        settings.bucket,
        key,
        ExtraArgs={
            "Metadata": {
                "sha256": digest,
                "token-count": str(int(tokens.size)),
                "split": split,
                "shard-index": str(index),
            }
        },
        Config=TransferConfig(),
    )
    head = client.head_object(Bucket=settings.bucket, Key=key)
    remote_bytes = int(head["ContentLength"])
    if remote_bytes != local_bytes:
        raise RuntimeError(
            f"remote size mismatch for {path.name}: {remote_bytes} != {local_bytes}"
        )
    metadata = {str(k).lower(): str(v) for k, v in head.get("Metadata", {}).items()}
    if metadata.get("sha256") != digest:
        raise RuntimeError(f"remote SHA-256 metadata mismatch for {path.name}")
    return {
        "filename": path.name,
        "split": split,
        "index": index,
        "token_count": int(tokens.size),
        "local_bytes": local_bytes,
        "remote_bytes": remote_bytes,
        "sha256": digest,
        "etag": str(head.get("ETag", "")).strip('"'),
        "remote_key": key,
        "upload_timestamp": utc_now(),
    }


def replace_progress_record(progress: dict[str, Any], record: Mapping[str, Any]) -> None:
    progress["shards"] = [
        item
        for item in progress["shards"]
        if item.get("filename") != record["filename"]
    ]
    progress["shards"].append(dict(record))


@contextmanager
def tokenized_documents(
    dataset: Iterable[dict[str, Any]],
    workers: int | None = None,
) -> Iterator[Iterable[np.ndarray]]:
    worker_count = workers or max(1, os.cpu_count() or 1)
    with mp.Pool(worker_count) as pool:
        yield pool.imap(tokenize, dataset, chunksize=16)


def load_source_dataset(
    cache_dir: Path | str | None = None,
    *,
    attempts: int = HF_LOAD_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
) -> Iterable[dict[str, Any]]:
    """Load FineWeb through the ordinary cached Hugging Face dataset path.

    ``datasets.load_dataset`` materializes the source dataset into the
    explicit cache, allowing retries to reuse completed downloads instead of
    repeatedly issuing fragile streaming range requests.
    """

    from datasets import load_dataset

    if attempts < 1:
        raise ValueError("attempts must be at least one")
    cache_root = Path(
        cache_dir
        or os.environ.get("HF_HOME")
        or Path.cwd() / "hf-cache"
    )
    configure_huggingface_cache(cache_root)
    dataset_cache = os.environ["HF_DATASETS_CACHE"]
    LOGGER.info(
        "loading FineWeb dataset through cached non-streaming path cache=%s",
        dataset_cache,
    )
    for attempt in range(1, attempts + 1):
        try:
            return load_dataset(
                DATASET_ID,
                name=DEFAULT_DATASET_CONFIG,
                split="train",
                streaming=False,
                cache_dir=dataset_cache,
            )
        except Exception as error:
            if attempt >= attempts or not _is_transient_dataset_error(error):
                raise
            delay = HF_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
            LOGGER.warning(
                "FineWeb dataset load retry=%d/%d delay_seconds=%.1f error=%s",
                attempt,
                attempts,
                delay,
                _safe_error_detail(error),
            )
            sleep(delay)
    raise RuntimeError("FineWeb dataset load exhausted retry attempts")


def create_and_upload_shards(
    client: Any,
    settings: Settings,
    progress: dict[str, Any],
    verified: Mapping[str, Mapping[str, Any]],
    work_dir: Path,
    *,
    dataset: Iterable[dict[str, Any]],
    token_stream_factory: Callable[
        [Iterable[dict[str, Any]]], Any
    ] = tokenized_documents,
    shard_size: int = DEFAULT_SHARD_SIZE,
    vocabulary_size: int,
) -> list[dict[str, Any]]:
    buffer = np.empty((shard_size,), dtype=DTYPE)
    shard_index = 0
    token_count = 0

    def finish_shard(tokens: np.ndarray, *, final_partial: bool) -> None:
        nonlocal shard_index
        split = "val" if shard_index == 0 else "train"
        filename = f"edufineweb_{split}_{shard_index:06d}.npy"
        existing = verified.get(filename)
        if existing and int(existing["token_count"]) == int(tokens.size):
            replace_progress_record(progress, existing)
            upload_progress(client, settings, progress)
            shard_index += 1
            return
        destination = work_dir / filename
        verified_remote = False
        try:
            atomic_write_npy(destination, tokens)
            record = upload_verified_shard(
                client,
                settings,
                destination,
                split=split,
                index=shard_index,
                shard_size=shard_size,
                allow_partial=final_partial,
                vocabulary_size=vocabulary_size,
            )
            replace_progress_record(progress, record)
            upload_progress(client, settings, progress)
            verified_remote = True
        finally:
            if verified_remote:
                destination.unlink(missing_ok=True)
        shard_index += 1

    with token_stream_factory(dataset) as tokenized_stream:
        for tokens in tokenized_stream:
            consumed = 0
            while consumed < len(tokens):
                amount = min(shard_size - token_count, len(tokens) - consumed)
                buffer[token_count : token_count + amount] = tokens[
                    consumed : consumed + amount
                ]
                token_count += amount
                consumed += amount
                if token_count == shard_size:
                    finish_shard(buffer, final_partial=False)
                    token_count = 0
        if token_count:
            finish_shard(buffer[:token_count].copy(), final_partial=True)
    return sorted(progress["shards"], key=lambda item: int(item["index"]))


def verify_manifest_reference(
    client: Any,
    settings: Settings,
    marker: Mapping[str, Any],
) -> bool:
    manifest_key = marker.get("manifest_path")
    expected_sha = marker.get("manifest_sha256")
    if not isinstance(manifest_key, str) or not isinstance(expected_sha, str):
        return False
    try:
        content = get_object_bytes(client, settings, manifest_key)
        if sha256_bytes(content) != expected_sha:
            return False
        manifest = json.loads(content)
        if manifest.get("git_sha") != marker.get("git_sha"):
            return False
        if manifest.get("shard_count") != marker.get("shard_count"):
            return False
        for shard in manifest.get("shards", []):
            head = client.head_object(
                Bucket=settings.bucket,
                Key=shard["remote_key"],
            )
            metadata = {
                str(k).lower(): str(v)
                for k, v in head.get("Metadata", {}).items()
            }
            if int(head["ContentLength"]) != int(shard["remote_bytes"]):
                return False
            if metadata.get("sha256") != shard["sha256"]:
                return False
    except Exception:
        return False
    return True


def already_complete(client: Any, settings: Settings) -> bool:
    key = settings.key(STATUS_PREFIX, COMPLETE_NAME)
    if not object_exists(client, settings, key):
        return False
    try:
        marker = json.loads(get_object_bytes(client, settings, key))
    except Exception:
        return False
    return verify_manifest_reference(client, settings, marker)


def sample_indices(count: int) -> list[int]:
    candidates = {0, 1, count - 1, count // 2}
    return sorted(index for index in candidates if 0 <= index < count)


def verify_random_batch(arrays: Sequence[np.ndarray]) -> dict[str, int]:
    available = [array for array in arrays if array.size >= 2]
    if not available:
        raise ValueError("no shard has enough tokens to form a training batch")
    random_source = random.Random(0)
    array = random_source.choice(available)
    sequence_length = min(32, int(array.size) - 1)
    batch_size = min(4, max(1, (int(array.size) - 1) // sequence_length))
    starts = [
        random_source.randrange(0, int(array.size) - sequence_length)
        for _ in range(batch_size)
    ]
    inputs = np.stack([array[start : start + sequence_length] for start in starts])
    targets = np.stack(
        [array[start + 1 : start + sequence_length + 1] for start in starts]
    )
    if inputs.shape != targets.shape or inputs.size == 0:
        raise ValueError("representative random batch verification failed")
    return {"batch_size": batch_size, "sequence_length": sequence_length}


def final_validation(
    client: Any,
    settings: Settings,
    records: Sequence[Mapping[str, Any]],
    work_dir: Path,
    *,
    shard_size: int,
    vocabulary_size: int,
) -> dict[str, Any]:
    ordered = validate_shard_filenames(
        [Path(str(record["filename"])) for record in records]
    )
    records_by_name = {str(record["filename"]): record for record in records}
    if len(records_by_name) != len(records):
        raise ValueError("duplicate shard filename in progress metadata")
    remote = list_objects(client, settings, settings.key(SHARD_PREFIX) + "/")
    remote_names: set[str] = set()
    for item in remote:
        remote_name = Path(item["Key"]).name
        if item["Key"] != settings.key(SHARD_PREFIX, remote_name):
            raise ValueError(f"unexpected remote shard key: {item['Key']}")
        if remote_name in remote_names:
            raise ValueError(f"duplicate remote shard filename: {remote_name}")
        remote_names.add(remote_name)
    expected_names = set(records_by_name)
    if remote_names != expected_names:
        raise ValueError("remote shard listing differs from verified progress metadata")

    for position, shard in enumerate(ordered):
        record = records_by_name[shard.path.name]
        head = client.head_object(
            Bucket=settings.bucket,
            Key=str(record["remote_key"]),
        )
        metadata = {
            str(k).lower(): str(v) for k, v in head.get("Metadata", {}).items()
        }
        if int(head["ContentLength"]) != int(record["remote_bytes"]):
            raise ValueError(f"remote size mismatch for {shard.path.name}")
        if metadata.get("sha256") != record["sha256"]:
            raise ValueError(f"remote SHA-256 mismatch for {shard.path.name}")
        expected_count = int(record["token_count"])
        if position < len(ordered) - 1 and expected_count != shard_size:
            raise ValueError(f"non-final shard is partial: {shard.path.name}")
        if position == len(ordered) - 1 and expected_count > shard_size:
            raise ValueError(f"final shard exceeds shard size: {shard.path.name}")

    sampled_arrays: list[np.ndarray] = []
    sampled_names: list[str] = []
    for index in sample_indices(len(ordered)):
        shard = ordered[index]
        record = records_by_name[shard.path.name]
        destination = work_dir / f"sample-{shard.path.name}"
        try:
            token_count, local_bytes, actual_sha = inspect_downloaded_shard(
                client,
                settings,
                str(record["remote_key"]),
                destination,
                shard_size=shard_size,
                allow_partial=index == len(ordered) - 1,
                vocabulary_size=vocabulary_size,
            )
            if token_count != int(record["token_count"]):
                raise ValueError(f"sample token count mismatch for {shard.path.name}")
            if local_bytes != int(record["remote_bytes"]):
                raise ValueError(f"sample byte count mismatch for {shard.path.name}")
            if actual_sha != record["sha256"]:
                raise ValueError(f"sample SHA-256 mismatch for {shard.path.name}")
            sampled = np.load(destination, allow_pickle=False, mmap_mode="r")
            sampled_arrays.append(np.array(sampled[: min(128, sampled.size)]))
            sampled_names.append(shard.path.name)
        finally:
            destination.unlink(missing_ok=True)
    batch = verify_random_batch(sampled_arrays)

    val_count = sum(1 for shard in ordered if shard.split == "val")
    train_count = sum(1 for shard in ordered if shard.split == "train")
    return {
        "numeric_filename_validation": True,
        "exactly_one_validation_shard": val_count == 1,
        "validation_index_zero": ordered[0].split == "val"
        and ordered[0].index == 0,
        "training_indices_contiguous": True,
        "unexpected_remote_files_absent": True,
        "all_remote_sizes_match": True,
        "all_recorded_sha256_match": True,
        "sampled_shards": sampled_names,
        "sampled_shards_readable_uint16": True,
        "token_ids_within_gpt2_vocabulary": True,
        "representative_random_batch": batch,
        "validation_shard_count": val_count,
        "training_shard_count": train_count,
    }


def build_manifest(
    settings: Settings,
    records: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
    *,
    preparation_git_sha: str | None,
    preparation_started_at: str,
    preparation_finished_at: str,
    disk_facts: DiskFacts | None = None,
    shard_size: int,
) -> dict[str, Any]:
    ordered = sorted(records, key=lambda item: int(item["index"]))
    return {
        "schema_version": 1,
        "source_dataset": DATASET_ID,
        "dataset_configuration": DEFAULT_DATASET_CONFIG,
        "tokenizer": TOKENIZER_NAME,
        "shard_format": SHARD_FORMAT,
        "dtype": "uint16",
        "shard_size": shard_size,
        "git_sha": preparation_git_sha,
        "preparation_run_id": settings.preparation_run_id,
        "givemeanode_job_id": settings.job_id,
        "preparation_command": "python scripts/prepare_fineweb_to_s3.py",
        "preparation_started_at": preparation_started_at,
        "preparation_finished_at": preparation_finished_at,
        "disk_preflight": (
            {
                "filesystem": disk_facts.filesystem,
                "mount_path": disk_facts.mount_path,
                "work_path": disk_facts.path,
                "total_bytes": disk_facts.total_bytes,
                "used_bytes": disk_facts.used_bytes,
                "available_bytes": disk_facts.available_bytes,
                "total_gib": disk_facts.total_gib,
                "used_gib": disk_facts.used_gib,
                "available_gib": disk_facts.available_gib,
                "budget": asdict(disk_facts.budget),
            }
            if disk_facts is not None
            else None
        ),
        "bucket": settings.bucket,
        "prefix": settings.prefix,
        "shard_count": len(ordered),
        "training_shard_count": sum(
            1 for item in ordered if item["split"] == "train"
        ),
        "validation_shard_count": sum(
            1 for item in ordered if item["split"] == "val"
        ),
        "total_token_count": sum(int(item["token_count"]) for item in ordered),
        "total_bytes": sum(int(item["remote_bytes"]) for item in ordered),
        "shards": ordered,
        "validation": dict(validation),
    }


def build_report(
    manifest: Mapping[str, Any] | None,
    *,
    complete: bool,
    failure_type: str | None = None,
) -> str:
    status = "complete" if complete else "incomplete"
    lines = [
        "# FineWeb-Edu S3 preparation report",
        "",
        f"- Status: {status}",
        f"- Generated: {utc_now()}",
    ]
    if manifest:
        lines.extend(
            [
                f"- Git SHA: {manifest['git_sha']}",
                f"- Preparation run ID: {manifest['preparation_run_id']}",
                f"- Job ID: {manifest['givemeanode_job_id']}",
                f"- Bucket: {manifest['bucket']}",
                f"- Prefix: {manifest['prefix']}",
                f"- Shards: {manifest['shard_count']}",
                f"- Tokens: {manifest['total_token_count']}",
                f"- Bytes: {manifest['total_bytes']}",
            ]
        )
        disk = manifest.get("disk_preflight")
        if disk:
            lines.extend(
                [
                    f"- Disk filesystem: {disk['filesystem']}",
                    f"- Disk mount: {disk['mount_path']}",
                    f"- Disk available GiB: {disk['available_gib']}",
                    f"- Disk required GiB: {disk['budget']['required_gib']}",
                    "- Disk model: streaming source/cache with progressive shard upload and local deletion",
                ]
            )
    if failure_type:
        lines.append(f"- Failure type: {failure_type}")
    lines.append("")
    return "\n".join(lines)


def upload_text(
    client: Any,
    settings: Settings,
    key: str,
    content: str,
    *,
    content_type: str,
) -> bytes:
    encoded = content.encode("utf-8")
    client.put_object(
        Bucket=settings.bucket,
        Key=key,
        Body=encoded,
        ContentType=content_type,
        Metadata={"sha256": sha256_bytes(encoded)},
    )
    return encoded


def write_small_output(settings: Settings, value: Mapping[str, Any]) -> None:
    if settings.output_dir is not None:
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            settings.output_dir / "dataset_preparation_result.json",
            value,
        )
    if settings.result_path is not None:
        atomic_write_json(settings.result_path, value)


def publish_final_metadata(
    client: Any,
    settings: Settings,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    manifest_key = settings.key(METADATA_PREFIX, MANIFEST_NAME)
    manifest_bytes = put_json(client, settings, manifest_key, manifest)
    manifest_sha = sha256_bytes(manifest_bytes)
    checksums = "".join(
        f"{item['sha256']}  {item['filename']}\n" for item in manifest["shards"]
    )
    checksums_key = settings.key(METADATA_PREFIX, CHECKSUMS_NAME)
    upload_text(
        client,
        settings,
        checksums_key,
        checksums,
        content_type="text/plain",
    )
    report_key = settings.key(METADATA_PREFIX, REPORT_NAME)
    report = build_report(manifest, complete=True)
    upload_text(
        client,
        settings,
        report_key,
        report,
        content_type="text/markdown",
    )
    final_status = {
        "status": "complete",
        "preparation_run_id": settings.preparation_run_id,
        "givemeanode_job_id": settings.job_id,
        "git_sha": manifest["git_sha"],
        "finished_at": manifest["preparation_finished_at"],
        "shard_count": manifest["shard_count"],
        "total_token_count": manifest["total_token_count"],
        "total_bytes": manifest["total_bytes"],
        "manifest_path": manifest_key,
        "manifest_sha256": manifest_sha,
    }
    put_json(
        client,
        settings,
        settings.key(
            "runs",
            settings.preparation_run_id,
            "final_status.json",
        ),
        final_status,
    )
    marker = {
        "manifest_path": manifest_key,
        "manifest_sha256": manifest_sha,
        "completion_timestamp": utc_now(),
        "git_sha": manifest["git_sha"],
        "preparation_run_id": settings.preparation_run_id,
        "givemeanode_job_id": settings.job_id,
        "shard_count": manifest["shard_count"],
        "total_token_count": manifest["total_token_count"],
        "total_bytes": manifest["total_bytes"],
    }
    put_json(
        client,
        settings,
        settings.key(STATUS_PREFIX, COMPLETE_NAME),
        marker,
    )
    return marker


def run(
    settings: Settings,
    *,
    client: Any | None = None,
    dataset_loader: Callable[[], Iterable[dict[str, Any]]] = load_source_dataset,
    token_stream_factory: Callable[
        [Iterable[dict[str, Any]]], Any
    ] = tokenized_documents,
    shard_size: int = DEFAULT_SHARD_SIZE,
) -> dict[str, Any]:
    active_client = client or create_s3_client(settings)
    current_sha = git_sha()
    started_at = utc_now()
    LOGGER.info(
        "configuration bucket=%s prefix=%s endpoint=%s region=%s "
        "preparation_run_id=%s job_id=%s git_sha=%s",
        settings.bucket,
        settings.prefix,
        settings.endpoint_hostname,
        settings.region,
        settings.preparation_run_id,
        settings.job_id,
        current_sha,
    )

    work_base = select_work_base()
    temp_root = work_base / "fineweb-s3" / settings.preparation_run_id
    disk_facts = require_free_disk(temp_root, shard_size=shard_size)
    phase = "storage_probe"
    progress: dict[str, Any] = {
        "schema_version": 1,
        "source_dataset": DATASET_ID,
        "dataset_configuration": DEFAULT_DATASET_CONFIG,
        "shard_size": shard_size,
        "git_sha": current_sha,
        "preparation_run_id": settings.preparation_run_id,
        "givemeanode_job_id": settings.job_id,
        "complete": False,
        "shards": [],
    }
    manifest: dict[str, Any] | None = None
    try:
        storage_probe(active_client, settings)
        put_json(
            active_client,
            settings,
            settings.key(
                "runs",
                settings.preparation_run_id,
                "startup.json",
            ),
            {
                "status": "startup_verified",
                "preparation_run_id": settings.preparation_run_id,
                "givemeanode_job_id": settings.job_id,
                "git_sha": current_sha,
                "started_at": started_at,
                "bucket": settings.bucket,
                "prefix": settings.prefix,
                "endpoint_hostname": settings.endpoint_hostname,
                "region": settings.region,
                "filesystem": disk_facts.filesystem,
                "mount_path": disk_facts.mount_path,
                "work_path": disk_facts.path,
                "total_bytes": disk_facts.total_bytes,
                "used_bytes": disk_facts.used_bytes,
                "available_bytes": disk_facts.available_bytes,
                "total_gib": disk_facts.total_gib,
                "used_gib": disk_facts.used_gib,
                "available_gib": disk_facts.available_gib,
                "disk_budget": asdict(disk_facts.budget),
                "storage_probe_passed": True,
            },
        )

        phase = "complete_marker_check"
        if already_complete(active_client, settings):
            result = {
                "status": "already_complete",
                "bucket": settings.bucket,
                "prefix": settings.prefix,
                "git_sha": current_sha,
                "preparation_run_id": settings.preparation_run_id,
                "givemeanode_job_id": settings.job_id,
            }
            put_json(
                active_client,
                settings,
                settings.key(
                    "runs",
                    settings.preparation_run_id,
                    "final_status.json",
                ),
                {
                    **result,
                    "finished_at": utc_now(),
                    "storage_probe_passed": True,
                },
            )
            write_small_output(settings, result)
            LOGGER.info(
                "verified COMPLETE marker; no dataset preparation required"
            )
            return result

        hf_cache = Path(os.environ.get("HF_HOME") or (temp_root / "hf-cache"))
        configure_huggingface_cache(hf_cache)
        vocabulary_size = tokenizer().n_vocab

        phase = "resume_validation"
        progress = load_progress(active_client, settings)
        progress.update(
            {
                "source_dataset": DATASET_ID,
                "dataset_configuration": DEFAULT_DATASET_CONFIG,
                "shard_size": shard_size,
                "git_sha": current_sha,
                "preparation_run_id": settings.preparation_run_id,
                "givemeanode_job_id": settings.job_id,
                "complete": False,
            }
        )
        verified = verify_existing_shards(
            active_client,
            settings,
            progress,
            temp_root,
            shard_size=shard_size,
            vocabulary_size=vocabulary_size,
        )
        upload_progress(active_client, settings, progress)

        phase = "dataset_download_and_tokenization"
        dataset = dataset_loader()
        records = create_and_upload_shards(
            active_client,
            settings,
            progress,
            verified,
            temp_root,
            dataset=dataset,
            token_stream_factory=token_stream_factory,
            shard_size=shard_size,
            vocabulary_size=vocabulary_size,
        )

        phase = "final_validation"
        validation = final_validation(
            active_client,
            settings,
            records,
            temp_root,
            shard_size=shard_size,
            vocabulary_size=vocabulary_size,
        )
        finished_at = utc_now()
        manifest = build_manifest(
            settings,
            records,
            validation,
            preparation_git_sha=current_sha,
            preparation_started_at=started_at,
            preparation_finished_at=finished_at,
            disk_facts=disk_facts,
            shard_size=shard_size,
        )

        phase = "final_metadata_publication"
        progress["complete"] = True
        upload_progress(active_client, settings, progress)
        marker = publish_final_metadata(active_client, settings, manifest)

        phase = "complete_marker_verification"
        if not verify_manifest_reference(active_client, settings, marker):
            active_client.delete_object(
                Bucket=settings.bucket,
                Key=settings.key(STATUS_PREFIX, COMPLETE_NAME),
            )
            raise RuntimeError("post-publication COMPLETE verification failed")
        result = {
            "status": "complete",
            "bucket": settings.bucket,
            "prefix": settings.prefix,
            "manifest_path": marker["manifest_path"],
            "manifest_sha256": marker["manifest_sha256"],
            "shard_count": marker["shard_count"],
            "total_token_count": marker["total_token_count"],
            "total_bytes": marker["total_bytes"],
            "git_sha": current_sha,
            "preparation_run_id": settings.preparation_run_id,
            "givemeanode_job_id": settings.job_id,
        }
        try:
            write_small_output(settings, result)
        except OSError:
            LOGGER.warning("could not write the small local result pointer")
        return result
    except Exception as error:
        progress["complete"] = False
        try:
            upload_progress(active_client, settings, progress)
            upload_text(
                active_client,
                settings,
                settings.key(METADATA_PREFIX, REPORT_NAME),
                build_report(
                    manifest,
                    complete=False,
                    failure_type=type(error).__name__,
                ),
                content_type="text/markdown",
            )
            put_json(
                active_client,
                settings,
                settings.key(
                    "runs",
                    settings.preparation_run_id,
                    "final_status.json",
                ),
                {
                    "status": "incomplete",
                    "failing_phase": phase,
                    "failure_type": type(error).__name__,
                    "preparation_run_id": settings.preparation_run_id,
                    "givemeanode_job_id": settings.job_id,
                    "git_sha": current_sha,
                    "started_at": started_at,
                    "finished_at": utc_now(),
                },
            )
        except Exception:
            LOGGER.error("failed to publish incomplete progress metadata")
        try:
            write_small_output(
                settings,
                {
                    "status": "incomplete",
                    "bucket": settings.bucket,
                    "prefix": settings.prefix,
                    "git_sha": current_sha,
                    "preparation_run_id": settings.preparation_run_id,
                    "givemeanode_job_id": settings.job_id,
                    "failing_phase": phase,
                    "failure_type": type(error).__name__,
                },
            )
        except OSError:
            LOGGER.warning("could not write the small local failure pointer")
        LOGGER.error("dataset preparation failed (%s)", type(error).__name__)
        raise
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def main() -> int:
    configure_logging()
    settings: Settings | None = None
    try:
        settings = Settings.from_environment()
        run(settings)
        return 0
    except Exception as error:
        if settings is None:
            LOGGER.error("startup validation failed (%s)", type(error).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
