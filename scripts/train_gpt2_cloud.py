"""Run resumable GPT-2 training against the verified FineWeb-Edu S3 dataset."""

from __future__ import annotations

import concurrent.futures
import contextlib
import dataclasses
import datetime as dt
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Mapping

import boto3
import numpy as np

from fineweb import validate_shard_filenames


EXPECTED_MANIFEST_SHA256 = (
    "5a994f484eccd41a3c10270a843d50021ef9ee7be3bc59c4c57f51dda6601386"
)
EXPECTED_SHARDS = 100
EXPECTED_TOKENS = 9_953_989_344
CHECKPOINT_RE = re.compile(r"(?:checkpoint|final)_step_(\d{6})\.pt$")
MILESTONE_STEPS = {5_000, 10_000, 15_000, 19_073}
S3_RETRY_ATTEMPTS = 4
SYNC_FAILURE_LIMIT = 3
SENSITIVE_ENV_NAMES = {
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "WANDB_API_KEY",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sanitized(value: str) -> str:
    result = value
    for name in SENSITIVE_ENV_NAMES:
        secret = os.environ.get(name)
        if secret:
            result = result.replace(secret, f"<redacted:{name}>")
    return result


def require_environment(names: Iterable[str]) -> None:
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"missing required environment variables: {sorted(missing)}")


@dataclasses.dataclass(frozen=True)
class CloudSettings:
    bucket: str
    dataset_prefix: str
    training_run_id: str
    source_git_sha: str
    endpoint_url: str
    region: str
    data_root: Path
    output_dir: Path
    training_prefix: str
    segment_seconds: int
    download_workers: int
    result_path: Path | None
    batch_output_dir: Path | None

    @classmethod
    def from_environment(cls) -> "CloudSettings":
        require_environment(
            (
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "AWS_ENDPOINT_URL",
                "GMN_DATA_BUCKET",
                "GMN_DATA_PREFIX",
                "GMN_TRAINING_RUN_ID",
                "GMN_SOURCE_GIT_SHA",
                "WANDB_API_KEY",
            )
        )
        run_id = os.environ["GMN_TRAINING_RUN_ID"].strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}", run_id):
            raise RuntimeError("GMN_TRAINING_RUN_ID contains unsafe characters")
        data_root = Path(os.environ.get("GMN_TRAINING_DATA_ROOT", "/workspace/edu_fineweb10B"))
        output_dir = Path(
            os.environ.get("GMN_TRAINING_OUTPUT_DIR", f"/workspace/training/{run_id}")
        )
        segment_seconds = int(os.environ.get("GMN_TRAINING_SEGMENT_SECONDS", "40800"))
        if not 60 <= segment_seconds <= 41_400:
            raise RuntimeError("GMN_TRAINING_SEGMENT_SECONDS must be between 60 and 41400")
        return cls(
            bucket=os.environ["GMN_DATA_BUCKET"],
            dataset_prefix=os.environ["GMN_DATA_PREFIX"].strip("/"),
            training_run_id=run_id,
            source_git_sha=os.environ["GMN_SOURCE_GIT_SHA"],
            endpoint_url=os.environ["AWS_ENDPOINT_URL"],
            region=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION", "auto"),
            data_root=data_root,
            output_dir=output_dir,
            training_prefix=f"training/{run_id}",
            segment_seconds=segment_seconds,
            download_workers=max(
                1, min(8, int(os.environ.get("GMN_DATA_DOWNLOAD_WORKERS", "4")))
            ),
            result_path=(
                Path(os.environ["GMN_RESULT_PATH"])
                if os.environ.get("GMN_RESULT_PATH")
                else None
            ),
            batch_output_dir=(
                Path(os.environ["GMN_OUTPUT_DIR"])
                if os.environ.get("GMN_OUTPUT_DIR")
                else None
            ),
        )

    def dataset_key(self, *parts: str) -> str:
        return "/".join((self.dataset_prefix, *(part.strip("/") for part in parts)))

    def training_key(self, *parts: str) -> str:
        return "/".join((self.training_prefix, *(part.strip("/") for part in parts)))


def create_s3_client(settings: CloudSettings) -> Any:
    return boto3.client(
        "s3",
        endpoint_url=settings.endpoint_url,
        region_name=settings.region,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


def get_object_bytes(client: Any, bucket: str, key: str) -> bytes:
    body = client.get_object(Bucket=bucket, Key=key)["Body"]
    try:
        return body.read()
    finally:
        with contextlib.suppress(Exception):
            body.close()


def retry_s3(
    operation: Any,
    description: str,
    *,
    attempts: int = S3_RETRY_ATTEMPTS,
    base_delay: float = 1.0,
) -> Any:
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except BaseException as exc:
            last_error = exc
            if attempt == attempts:
                break
            delay = base_delay * (2 ** (attempt - 1))
            print(
                f"{description} failed on attempt {attempt}/{attempts}; "
                f"retrying in {delay:g}s: {sanitized(str(exc))}",
                file=sys.stderr,
            )
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def put_json(client: Any, bucket: str, key: str, value: Mapping[str, Any]) -> bytes:
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")

    def publish() -> None:
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=encoded,
            ContentType="application/json",
            Metadata={"sha256": sha256_bytes(encoded)},
        )
        head = client.head_object(Bucket=bucket, Key=key)
        if int(head["ContentLength"]) != len(encoded):
            raise RuntimeError(f"remote size mismatch for {key}")

    retry_s3(publish, f"publish {key}")
    return encoded


def load_verified_dataset_manifest(
    client: Any, settings: CloudSettings
) -> tuple[dict[str, Any], dict[str, Any]]:
    marker_key = settings.dataset_key("status", "COMPLETE")
    marker = json.loads(get_object_bytes(client, settings.bucket, marker_key))
    manifest_key = str(marker["manifest_path"])
    manifest_bytes = get_object_bytes(client, settings.bucket, manifest_key)
    actual_manifest_sha = sha256_bytes(manifest_bytes)
    if actual_manifest_sha != marker.get("manifest_sha256"):
        raise RuntimeError("dataset COMPLETE marker references a corrupt manifest")
    if actual_manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            "dataset manifest SHA-256 differs from the approved preparation manifest"
        )
    manifest = json.loads(manifest_bytes)
    if int(manifest.get("shard_count", -1)) != EXPECTED_SHARDS:
        raise RuntimeError("dataset manifest does not contain exactly 100 shards")
    if int(manifest.get("total_token_count", -1)) != EXPECTED_TOKENS:
        raise RuntimeError("dataset manifest token count differs from the approved value")
    records = list(manifest.get("shards", []))
    if len(records) != EXPECTED_SHARDS:
        raise RuntimeError("dataset manifest shard records are incomplete")
    validate_shard_filenames(record["filename"] for record in records)
    for record in records:
        if (
            not record.get("sha256")
            or int(record.get("remote_bytes", 0)) <= 0
            or int(record.get("token_count", 0)) <= 0
            or not str(record.get("remote_key", "")).startswith(
                settings.dataset_key("shards") + "/"
            )
        ):
            raise RuntimeError(f"invalid dataset record for {record.get('filename')}")
    return marker, manifest


def _iter_body(body: Any, chunk_size: int = 8 * 1024 * 1024) -> Iterable[bytes]:
    if hasattr(body, "iter_chunks"):
        yield from body.iter_chunks(chunk_size=chunk_size)
        return
    while chunk := body.read(chunk_size):
        yield chunk


def download_verified_shard(
    client: Any,
    settings: CloudSettings,
    record: Mapping[str, Any],
) -> Path:
    destination = settings.data_root / str(record["filename"])
    expected_bytes = int(record["remote_bytes"])
    expected_sha = str(record["sha256"])
    expected_tokens = int(record["token_count"])
    if destination.is_file() and destination.stat().st_size == expected_bytes:
        if sha256_file(destination) == expected_sha:
            return destination
        destination.unlink()
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    response = client.get_object(Bucket=settings.bucket, Key=str(record["remote_key"]))
    body = response["Body"]
    digest = hashlib.sha256()
    written = 0
    try:
        with temporary.open("wb") as handle:
            for chunk in _iter_body(body):
                handle.write(chunk)
                digest.update(chunk)
                written += len(chunk)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        with contextlib.suppress(Exception):
            body.close()
    if written != expected_bytes or digest.hexdigest() != expected_sha:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"download verification failed for {record['filename']}")
    os.replace(temporary, destination)
    tokens = np.load(destination, mmap_mode="r")
    if tokens.dtype != np.uint16 or int(tokens.size) != expected_tokens:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"NumPy validation failed for {record['filename']}")
    return destination


def stage_dataset(
    client: Any, settings: CloudSettings, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    settings.data_root.mkdir(parents=True, exist_ok=True)
    records = list(manifest["shards"])
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=settings.download_workers
    ) as executor:
        futures = {
            executor.submit(download_verified_shard, client, settings, record): record
            for record in records
        }
        for future in concurrent.futures.as_completed(futures):
            future.result()
            completed += 1
            if completed == 1 or completed % 10 == 0 or completed == len(records):
                print(f"dataset staging: verified {completed}/{len(records)} shards")
    ordered = validate_shard_filenames(settings.data_root.iterdir())
    stored_bytes = sum(item.path.stat().st_size for item in ordered)
    return {
        "shard_count": len(ordered),
        "stored_bytes": stored_bytes,
        "available_disk_bytes": shutil.disk_usage(settings.data_root).free,
    }


def upload_verified_file(
    client: Any, settings: CloudSettings, path: Path, key: str
) -> dict[str, Any]:
    local_bytes = path.stat().st_size
    local_sha = sha256_file(path)
    retry_s3(
        lambda: client.upload_file(
            str(path),
            settings.bucket,
            key,
            ExtraArgs={"Metadata": {"sha256": local_sha}},
        ),
        f"upload {key}",
    )

    def verify_head() -> None:
        head = client.head_object(Bucket=settings.bucket, Key=key)
        if int(head["ContentLength"]) != local_bytes:
            raise RuntimeError(f"remote checkpoint size mismatch for {key}")
        remote_sha = str(head.get("Metadata", {}).get("sha256", ""))
        if remote_sha and remote_sha != local_sha:
            raise RuntimeError(f"remote checkpoint checksum metadata mismatch for {key}")

    retry_s3(verify_head, f"verify {key}")
    checksum_key = f"{key}.sha256"
    checksum_bytes = f"{local_sha}  {path.name}\n".encode("utf-8")
    retry_s3(
        lambda: client.put_object(
            Bucket=settings.bucket,
            Key=checksum_key,
            Body=checksum_bytes,
            ContentType="text/plain",
        ),
        f"publish {checksum_key}",
    )

    def verify_checksum_sidecar() -> None:
        remote_checksum = get_object_bytes(client, settings.bucket, checksum_key)
        if remote_checksum != checksum_bytes:
            raise RuntimeError(f"remote checkpoint checksum sidecar mismatch for {key}")

    retry_s3(verify_checksum_sidecar, f"verify {checksum_key}")
    return {
        "key": key,
        "filename": path.name,
        "bytes": local_bytes,
        "sha256": local_sha,
        "verified_at": utc_now(),
    }


def checkpoint_step(path_or_key: str | Path) -> int | None:
    match = CHECKPOINT_RE.search(Path(path_or_key).name)
    return int(match.group(1)) if match else None


def _list_keys(client: Any, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        page = client.list_objects_v2(**kwargs)
        keys.extend(item["Key"] for item in page.get("Contents", []))
        if not page.get("IsTruncated"):
            return keys
        token = page["NextContinuationToken"]


def prune_remote_checkpoints(client: Any, settings: CloudSettings) -> None:
    prefix = settings.training_key("checkpoints") + "/"
    checkpoint_keys = [
        key
        for key in _list_keys(client, settings.bucket, prefix)
        if checkpoint_step(key) is not None and not key.endswith(".sha256")
    ]
    by_step: dict[int, str] = {}
    for key in sorted(checkpoint_keys):
        step = checkpoint_step(key)
        assert step is not None
        if step not in by_step or Path(key).name.startswith("checkpoint_"):
            by_step[step] = key
    rolling = sorted(step for step in by_step if step not in MILESTONE_STEPS)
    retained = set(rolling[-3:]) | (set(by_step) & MILESTONE_STEPS)
    for step, key in by_step.items():
        if step in retained:
            continue
        client.delete_object(Bucket=settings.bucket, Key=key)
        client.delete_object(Bucket=settings.bucket, Key=f"{key}.sha256")


def find_latest_verified_checkpoint(
    client: Any, settings: CloudSettings
) -> tuple[Path | None, dict[str, Any] | None]:
    latest_key = settings.training_key("checkpoints", "LATEST.json")
    try:
        latest = json.loads(get_object_bytes(client, settings.bucket, latest_key))
    except Exception as exc:
        response = getattr(exc, "response", {})
        code = str(response.get("Error", {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound"}:
            return None, None
        raise
    key = str(latest["key"])
    if not key.startswith(settings.training_key("checkpoints") + "/"):
        raise RuntimeError("LATEST.json points outside this training run")
    destination = settings.output_dir / "resume" / Path(key).name
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_bytes = int(latest["bytes"])
    expected_sha = str(latest["sha256"])

    def verify_remote_head() -> None:
        head = client.head_object(Bucket=settings.bucket, Key=key)
        if int(head["ContentLength"]) != expected_bytes:
            raise RuntimeError("latest remote checkpoint size does not match LATEST.json")
        remote_sha = str(head.get("Metadata", {}).get("sha256", ""))
        if remote_sha and remote_sha != expected_sha:
            raise RuntimeError("latest remote checkpoint metadata does not match LATEST.json")
        if not remote_sha:
            print(
                "resume checkpoint has no SHA-256 object metadata; "
                "using the verified LATEST.json checksum and validating the "
                "complete download before torch.load"
            )

    retry_s3(verify_remote_head, f"verify resume checkpoint {key}")
    temporary = destination.with_suffix(".part")
    if destination.is_file():
        if (
            destination.stat().st_size == expected_bytes
            and sha256_file(destination) == expected_sha
        ):
            return destination, latest
        destination.unlink()
    if temporary.is_file() and temporary.stat().st_size > expected_bytes:
        temporary.unlink()

    last_error: BaseException | None = None
    for attempt in range(1, S3_RETRY_ATTEMPTS + 1):
        offset = temporary.stat().st_size if temporary.is_file() else 0
        kwargs: dict[str, Any] = {"Bucket": settings.bucket, "Key": key}
        if offset:
            kwargs["Range"] = f"bytes={offset}-"
        body: Any | None = None
        try:
            response = client.get_object(**kwargs)
            body = response["Body"]
            if offset and "ContentRange" not in response:
                temporary.unlink(missing_ok=True)
                raise RuntimeError("checkpoint server ignored requested Range")
            with temporary.open("ab") as handle:
                for chunk in _iter_body(body):
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            if temporary.stat().st_size != expected_bytes:
                raise RuntimeError("checkpoint download ended before the expected byte size")
            if sha256_file(temporary) != expected_sha:
                temporary.unlink(missing_ok=True)
                raise RuntimeError("latest remote checkpoint failed checksum verification")
            os.replace(temporary, destination)
            return destination, latest
        except BaseException as exc:
            last_error = exc
            if attempt == S3_RETRY_ATTEMPTS:
                break
            delay = 2 ** (attempt - 1)
            print(
                f"checkpoint download failed on attempt "
                f"{attempt}/{S3_RETRY_ATTEMPTS}; retrying in {delay}s: "
                f"{sanitized(str(exc))}",
                file=sys.stderr,
            )
            time.sleep(delay)
        finally:
            if body is not None:
                with contextlib.suppress(Exception):
                    body.close()
    if temporary.is_file() and temporary.stat().st_size == expected_bytes:
        temporary.unlink(missing_ok=True)
    assert last_error is not None
    raise RuntimeError("latest remote checkpoint failed download verification") from last_error


class ArtifactSync:
    def __init__(self, client: Any, settings: CloudSettings) -> None:
        self.client = client
        self.settings = settings
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.error: BaseException | None = None
        self.synced: dict[int, dict[str, Any]] = {}
        self.latest: dict[str, Any] | None = None
        self.failure_step: int | None = None
        self.consecutive_failures = 0

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join()
        if self.error is not None:
            raise RuntimeError("cloud artifact synchronization failed") from self.error

    def _checkpoint_candidates(self) -> list[Path]:
        directory = self.settings.output_dir / "checkpoints"
        candidates = list(directory.glob("checkpoint_step_*.pt"))
        candidates.extend(directory.glob("final_step_*.pt"))
        by_step: dict[int, Path] = {}
        for path in sorted(candidates):
            step = checkpoint_step(path)
            if step is None:
                continue
            if step not in by_step or path.name.startswith("checkpoint_"):
                by_step[step] = path
        return [by_step[step] for step in sorted(by_step)]

    def _sync_small_artifacts(self) -> None:
        mappings = {
            self.settings.output_dir / "logs" / "train.log": "logs/train.log",
            self.settings.output_dir
            / "results"
            / "training_history.csv": "results/training_history.csv",
            self.settings.output_dir
            / "results"
            / "run_status.json": "results/run_status.json",
            self.settings.output_dir
            / "results"
            / "generated_samples.md": "results/generated_samples.md",
            self.settings.output_dir
            / "results"
            / "cloud_status.json": "results/cloud_status.json",
        }
        for path, suffix in mappings.items():
            if not path.is_file():
                continue
            content = path.read_bytes()
            key = self.settings.training_key(suffix)
            retry_s3(
                lambda: self.client.put_object(
                    Bucket=self.settings.bucket,
                    Key=key,
                    Body=content,
                ),
                f"publish {key}",
            )

    def _next_unsynced_step(self) -> int | None:
        for path in self._checkpoint_candidates():
            step = checkpoint_step(path)
            if step is not None and step not in self.synced:
                return step
        return None

    def sync_once(self) -> None:
        for path in self._checkpoint_candidates():
            step = checkpoint_step(path)
            assert step is not None
            if step in self.synced:
                continue
            key = self.settings.training_key("checkpoints", path.name)
            record = upload_verified_file(self.client, self.settings, path, key)
            record["step"] = step
            put_json(
                self.client,
                self.settings.bucket,
                self.settings.training_key("checkpoints", "LATEST.json"),
                record,
            )
            self.synced[step] = record
            self.latest = record
            print(f"checkpoint sync: verified step {step} at s3://{self.settings.bucket}/{key}")
            prune_remote_checkpoints(self.client, self.settings)
        self._sync_small_artifacts()

    def _run(self) -> None:
        while True:
            stopping = self.stop_event.wait(15)
            try:
                self.sync_once()
                self.failure_step = None
                self.consecutive_failures = 0
                if stopping:
                    return
            except BaseException as exc:
                step = self._next_unsynced_step()
                if step == self.failure_step:
                    self.consecutive_failures += 1
                else:
                    self.failure_step = step
                    self.consecutive_failures = 1
                if self.consecutive_failures >= SYNC_FAILURE_LIMIT:
                    self.error = exc
                    self.stop_event.set()
                    return
                delay = 2 ** (self.consecutive_failures - 1)
                print(
                    f"artifact sync failed for checkpoint step {step}; "
                    f"retry {self.consecutive_failures}/{SYNC_FAILURE_LIMIT} "
                    f"in {delay}s: {sanitized(str(exc))}",
                    file=sys.stderr,
                )
                time.sleep(delay)


def write_local_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_result(settings: CloudSettings, value: Mapping[str, Any]) -> None:
    if settings.result_path:
        write_local_json(settings.result_path, value)
    if settings.batch_output_dir:
        write_local_json(settings.batch_output_dir / "training_result.json", value)


def publish_segment_status(
    client: Any, settings: CloudSettings, result: Mapping[str, Any]
) -> None:
    put_json(
        client,
        settings.bucket,
        settings.training_key("status", "LATEST.json"),
        result,
    )
    if result.get("status") == "completed":
        put_json(
            client,
            settings.bucket,
            settings.training_key("status", "COMPLETE"),
            result,
        )


def training_command(
    settings: CloudSettings, resume_path: Path | None = None
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "train_gpt2.py",
        "--config",
        "configs/gpt2_124m_fineweb10b.yaml",
        "--data-root",
        str(settings.data_root),
        "--output-dir",
        str(settings.output_dir),
        "--max-steps",
        "19073",
        "--checkpoint-interval",
        "500",
        "--wandb-mode",
        "online",
        "--wandb-project",
        "gpt2-from-scratch",
        "--wandb-run-name",
        settings.training_run_id,
        "--max-runtime-seconds",
        str(settings.segment_seconds),
    ]
    if resume_path is not None:
        command.extend(("--resume", str(resume_path)))
    return command


def run_training(
    client: Any,
    settings: CloudSettings,
    resume_path: Path | None,
) -> tuple[int, ArtifactSync]:
    sync = ArtifactSync(client, settings)
    sync.start()
    process = subprocess.Popen(training_command(settings, resume_path))
    try:
        while process.poll() is None:
            if sync.error is not None:
                process.terminate()
                process.wait(timeout=120)
                raise RuntimeError("artifact synchronization failed during training")
            time.sleep(5)
        return_code = int(process.returncode)
    finally:
        sync.stop()
    return return_code, sync


def main() -> int:
    settings: CloudSettings | None = None
    phase = "environment"
    started_at = utc_now()
    try:
        settings = CloudSettings.from_environment()
        print(
            "cloud training startup: "
            f"bucket={settings.bucket} dataset_prefix={settings.dataset_prefix} "
            f"training_prefix={settings.training_prefix} git_sha={settings.source_git_sha}"
        )
        if os.environ.get("GMN_STOP_AFTER_VERIFIED_SHARDS"):
            raise RuntimeError("smoke settings are forbidden for production training")
        os.environ.setdefault("WANDB_SILENT", "true")
        os.environ.setdefault("WANDB_CONSOLE", "off")
        os.environ.setdefault("WANDB_DIR", str(settings.output_dir / "wandb"))
        phase = "gpu_preflight"
        try:
            import torch

            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError("production training requires exactly one visible CUDA GPU")
            print(f"gpu preflight: one {torch.cuda.get_device_name(0)} is visible")
        except ImportError as exc:
            raise RuntimeError("PyTorch is unavailable") from exc
        phase = "s3_and_manifest"
        client = create_s3_client(settings)
        marker, manifest = load_verified_dataset_manifest(client, settings)
        print(
            "dataset manifest verified: "
            f"sha256={marker['manifest_sha256']} shards={manifest['shard_count']} "
            f"tokens={manifest['total_token_count']}"
        )
        phase = "dataset_staging"
        staged = stage_dataset(client, settings, manifest)
        print(
            f"dataset staging complete: shards={staged['shard_count']} "
            f"bytes={staged['stored_bytes']} free_bytes={staged['available_disk_bytes']}"
        )
        phase = "checkpoint_resume"
        resume_path, latest = find_latest_verified_checkpoint(client, settings)
        if latest:
            print(
                f"resume checkpoint verified: step={latest['step']} "
                f"key={latest['key']}"
            )
        else:
            print("resume checkpoint: none; starting from random initialization")
        status_path = settings.output_dir / "results" / "cloud_status.json"
        write_local_json(
            status_path,
            {
                "status": "running",
                "phase": "training",
                "training_run_id": settings.training_run_id,
                "source_git_sha": settings.source_git_sha,
                "started_at": started_at,
                "dataset": staged,
                "resume": latest,
            },
        )
        phase = "training"
        return_code, sync = run_training(client, settings, resume_path)
        if return_code:
            raise RuntimeError(f"training process exited with status {return_code}")
        train_status_path = settings.output_dir / "results" / "run_status.json"
        train_status = json.loads(train_status_path.read_text(encoding="utf-8"))
        terminal_status = str(train_status.get("status"))
        if terminal_status not in {"paused", "completed"}:
            raise RuntimeError(f"unexpected training status: {terminal_status}")
        result = {
            "status": terminal_status,
            "training_run_id": settings.training_run_id,
            "source_git_sha": settings.source_git_sha,
            "started_at": started_at,
            "finished_at": utc_now(),
            "final_step": train_status.get("final_step"),
            "tokens_seen": train_status.get("tokens_seen"),
            "elapsed_wall_time": train_status.get("elapsed_wall_time"),
            "latest_checkpoint": sync.latest or latest,
            "dataset": staged,
            "wandb_run_id": train_status.get("wandb_run_id"),
        }
        write_local_json(status_path, result)
        sync._sync_small_artifacts()
        publish_segment_status(client, settings, result)
        write_result(settings, result)
        print(
            f"training segment finished: status={terminal_status} "
            f"step={result['final_step']}"
        )
        return 0
    except BaseException as exc:
        message = sanitized(str(exc))
        rendered_traceback = sanitized(traceback.format_exc())
        print(rendered_traceback, file=sys.stderr)
        failure = {
            "status": "error",
            "phase": phase,
            "error_type": type(exc).__name__,
            "error_message": message,
            "started_at": started_at,
            "finished_at": utc_now(),
        }
        if settings is not None:
            failure.update(
                {
                    "training_run_id": settings.training_run_id,
                    "source_git_sha": settings.source_git_sha,
                }
            )
            write_local_json(
                settings.output_dir / "results" / "cloud_status.json", failure
            )
            with contextlib.suppress(Exception):
                client = create_s3_client(settings)
                put_json(
                    client,
                    settings.bucket,
                    settings.training_key("status", "LATEST.json"),
                    failure,
                )
            with contextlib.suppress(Exception):
                write_result(settings, failure)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
