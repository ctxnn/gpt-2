"""Deterministic, resumable FineWeb processing from one Parquet file at a time."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence
from urllib.parse import quote

import numpy as np
import requests

from fineweb import DATASET_ID, DEFAULT_DATASET_CONFIG, tokenize

SOURCE_SUBDIRECTORY = "sample/10BT"
# Resolved from the Hub before submission. Keeping the commit in source makes
# first runs and resumes independent of later changes to the repository's main.
PINNED_DATASET_REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
SOURCE_MANIFEST_KEY = "source/source_manifest.json"
RESUME_PREFIX = "resume"
CHECKPOINT_VERSION = 1
DEFAULT_BATCH_SIZE = 128
DEFAULT_CHECKPOINT_DOCUMENTS = 10_000
DOWNLOAD_ATTEMPTS = 5
DOWNLOAD_BACKOFF_SECONDS = 2.0
DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = DOWNLOAD_CHUNK_BYTES) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def natural_key(value: str) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
        if part
    )


def _sibling_value(sibling: Any, name: str, default: Any = None) -> Any:
    if isinstance(sibling, Mapping):
        return sibling.get(name, default)
    return getattr(sibling, name, default)


def _metadata_value(metadata: Any, name: str) -> Any:
    if metadata is None:
        return None
    if isinstance(metadata, Mapping):
        return metadata.get(name)
    return getattr(metadata, name, None)


def build_source_manifest(
    *,
    api: Any | None = None,
    requested_revision: str = PINNED_DATASET_REVISION,
) -> dict[str, Any]:
    """Resolve a commit and enumerate only sample-10BT Parquet objects."""

    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    info = api.dataset_info(
        DATASET_ID,
        revision=requested_revision,
        files_metadata=True,
    )
    revision = str(_sibling_value(info, "sha", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError("Hugging Face did not return a pinned dataset commit")
    if re.fullmatch(r"[0-9a-f]{40}", requested_revision) and (
        revision != requested_revision
    ):
        raise RuntimeError("Hugging Face resolved a different dataset commit")

    candidates: list[dict[str, Any]] = []
    for sibling in _sibling_value(info, "siblings", ()) or ():
        path = str(_sibling_value(sibling, "rfilename", ""))
        if not path.startswith(f"{SOURCE_SUBDIRECTORY}/") or not path.endswith(
            ".parquet"
        ):
            continue
        lfs = _sibling_value(sibling, "lfs")
        xet = _sibling_value(sibling, "xet")
        size = _sibling_value(sibling, "size")
        if size is None:
            size = _metadata_value(lfs, "size") or _metadata_value(xet, "size")
        checksum = (
            _metadata_value(lfs, "sha256")
            or _metadata_value(xet, "sha256")
            or _metadata_value(xet, "hash")
        )
        object_id = (
            _sibling_value(sibling, "blob_id")
            or _metadata_value(lfs, "oid")
            or _metadata_value(xet, "hash")
        )
        candidates.append(
            {
                "repository_path": path,
                "download_url": (
                    f"https://huggingface.co/datasets/{DATASET_ID}/resolve/"
                    f"{revision}/{quote(path, safe='/')}"
                ),
                "expected_bytes": int(size) if size is not None else None,
                "checksum": str(checksum) if checksum else None,
                "hub_object_id": str(object_id) if object_id else None,
                "pinned_revision": revision,
            }
        )
    candidates.sort(key=lambda item: natural_key(item["repository_path"]))
    if not candidates:
        raise RuntimeError("no sample-10BT Parquet files were found")
    for ordinal, item in enumerate(candidates):
        item["ordinal"] = ordinal
    manifest = {
        "schema_version": 1,
        "source_dataset": DATASET_ID,
        "dataset_configuration": DEFAULT_DATASET_CONFIG,
        "source_subdirectory": SOURCE_SUBDIRECTORY,
        "requested_revision": requested_revision,
        "pinned_revision": revision,
        "file_count": len(candidates),
        "files": candidates,
    }
    manifest["manifest_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    return manifest


def download_with_resume(
    source: Mapping[str, Any],
    destination: Path,
    *,
    session: Any | None = None,
    attempts: int = DOWNLOAD_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Download to ``.part``, using Range when a partial file exists."""

    if attempts < 1:
        raise ValueError("attempts must be at least one")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    active_session = session or requests.Session()
    expected_bytes = source.get("expected_bytes")
    for attempt in range(1, attempts + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            response = active_session.get(
                str(source["download_url"]),
                headers=headers,
                stream=True,
                timeout=(30, 180),
            )
            response.raise_for_status()
            append = offset > 0 and int(response.status_code) == 206
            if offset and not append:
                offset = 0
            mode = "ab" if append else "wb"
            with partial.open(mode) as handle:
                for chunk in response.iter_content(DOWNLOAD_CHUNK_BYTES):
                    if chunk:
                        handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            actual_bytes = partial.stat().st_size
            if expected_bytes is not None and actual_bytes != int(expected_bytes):
                raise IOError(
                    f"source size mismatch: {actual_bytes} != {int(expected_bytes)}"
                )
            checksum = source.get("checksum")
            if checksum and re.fullmatch(r"[0-9a-fA-F]{64}", str(checksum)):
                actual_sha = sha256_file(partial)
                if actual_sha.lower() != str(checksum).lower():
                    raise IOError("source SHA-256 mismatch")
            os.replace(partial, destination)
            return destination
        except Exception:
            if attempt >= attempts:
                raise
            if (
                partial.exists()
                and expected_bytes is not None
                and partial.stat().st_size >= int(expected_bytes)
            ):
                partial.unlink()
            sleep(DOWNLOAD_BACKOFF_SECONDS * (2 ** (attempt - 1)))
    raise RuntimeError("source download exhausted retry attempts")


def iter_parquet_rows(
    path: Path,
    *,
    start_row: int = 0,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield rows in stored order without Arrow dataset materialization."""

    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    row_number = 0
    for batch in parquet.iter_batches(batch_size=batch_size, columns=["text"]):
        texts = batch.column(0).to_pylist()
        for text in texts:
            if row_number >= start_row:
                yield row_number, {"text": text}
            row_number += 1


def _put_json(client: Any, settings: Any, key: str, value: Any) -> bytes:
    content = canonical_json_bytes(value)
    client.put_object(
        Bucket=settings.bucket,
        Key=key,
        Body=content,
        ContentType="application/json",
        Metadata={"sha256": sha256_bytes(content)},
    )
    head = client.head_object(Bucket=settings.bucket, Key=key)
    if int(head["ContentLength"]) != len(content):
        raise RuntimeError(f"remote size mismatch for {key}")
    return content


def _get_bytes(client: Any, settings: Any, key: str) -> bytes:
    response = client.get_object(Bucket=settings.bucket, Key=key)
    body = response["Body"]
    return body.read() if hasattr(body, "read") else bytes(body)


def _exists(client: Any, settings: Any, key: str) -> bool:
    try:
        client.head_object(Bucket=settings.bucket, Key=key)
    except Exception as error:
        response = getattr(error, "response", {})
        if (
            str(response.get("Error", {}).get("Code", ""))
            in {"404", "NoSuchKey", "NotFound"}
            or response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404
        ):
            return False
        raise
    return True


def ensure_source_manifest(
    client: Any,
    settings: Any,
    *,
    resolver: Callable[[], dict[str, Any]] = build_source_manifest,
) -> dict[str, Any]:
    key = settings.key(SOURCE_MANIFEST_KEY)
    if _exists(client, settings, key):
        manifest = json.loads(_get_bytes(client, settings, key))
        expected = manifest.get("manifest_sha256")
        unsigned = dict(manifest)
        unsigned.pop("manifest_sha256", None)
        if expected != sha256_bytes(canonical_json_bytes(unsigned)):
            raise RuntimeError("stored source manifest checksum is invalid")
        return manifest
    manifest = resolver()
    _put_json(client, settings, key, manifest)
    return manifest


@dataclass
class ResumeState:
    source_ordinal: int
    next_row: int
    processed_documents: int
    processed_tokens: int
    next_shard_index: int
    shards: list[dict[str, Any]]
    partial: np.ndarray
    generation: int = 0


def initial_resume_state() -> ResumeState:
    return ResumeState(0, 0, 0, 0, 0, [], np.empty(0, dtype=np.uint16))


def persist_checkpoint(
    client: Any,
    settings: Any,
    state: ResumeState,
    *,
    source_manifest: Mapping[str, Any],
    source_git_sha: str | None,
) -> dict[str, str]:
    """Publish buffer and state first, then atomically advance LATEST."""

    latest_key = settings.key(RESUME_PREFIX, "LATEST.json")
    previous_latest: Mapping[str, Any] | None = None
    if _exists(client, settings, latest_key):
        previous_latest = json.loads(_get_bytes(client, settings, latest_key))
    generation = state.generation + 1
    generation_id = f"{generation:08d}-{uuid.uuid4().hex}"
    root = settings.key(RESUME_PREFIX, "generations", generation_id)
    buffer_key = f"{root}/partial.npy"
    state_key = f"{root}/state.json"
    local_root = Path(os.environ.get("TMPDIR", "/tmp")) / "fineweb-checkpoints"
    local_root.mkdir(parents=True, exist_ok=True)
    local_buffer = local_root / f"{generation_id}.npy"
    try:
        with local_buffer.open("wb") as handle:
            np.save(handle, state.partial.astype(np.uint16, copy=False), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        buffer_sha = sha256_file(local_buffer)
        client.upload_file(
            str(local_buffer),
            settings.bucket,
            buffer_key,
            ExtraArgs={"Metadata": {"sha256": buffer_sha}},
        )
        buffer_head = client.head_object(Bucket=settings.bucket, Key=buffer_key)
        if int(buffer_head["ContentLength"]) != local_buffer.stat().st_size:
            raise RuntimeError("partial checkpoint remote size mismatch")
        metadata = {
            str(key).lower(): str(value)
            for key, value in buffer_head.get("Metadata", {}).items()
        }
        if metadata.get("sha256") != buffer_sha:
            raise RuntimeError("partial checkpoint SHA-256 metadata mismatch")
        files = source_manifest["files"]
        current_path = (
            files[state.source_ordinal]["repository_path"]
            if state.source_ordinal < len(files)
            else None
        )
        payload = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "pinned_dataset_revision": source_manifest["pinned_revision"],
            "source_manifest_sha256": source_manifest["manifest_sha256"],
            "current_source_ordinal": state.source_ordinal,
            "current_source_path": current_path,
            "next_row": state.next_row,
            "processed_document_count": state.processed_documents,
            "processed_token_count": state.processed_tokens,
            "next_shard_index": state.next_shard_index,
            "verified_shards": state.shards,
            "partial_token_buffer_length": int(state.partial.size),
            "partial_token_buffer_sha256": sha256_bytes(state.partial.tobytes()),
            "partial_buffer_object_sha256": buffer_sha,
            "partial_buffer_key": buffer_key,
            "preparation_run_id": settings.preparation_run_id,
            "source_git_sha": source_git_sha,
            "checkpoint_timestamp": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
            "generation": generation,
        }
        state_bytes = _put_json(client, settings, state_key, payload)
        latest = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "generation": generation,
            "state_key": state_key,
            "state_sha256": sha256_bytes(state_bytes),
            "partial_buffer_key": buffer_key,
            "partial_buffer_object_sha256": buffer_sha,
        }
        _put_json(
            client,
            settings,
            latest_key,
            latest,
        )
        if previous_latest is not None:
            for old_key_name in ("state_key", "partial_buffer_key"):
                old_key = previous_latest.get(old_key_name)
                if isinstance(old_key, str) and old_key not in {
                    state_key,
                    buffer_key,
                }:
                    try:
                        client.delete_object(
                            Bucket=settings.bucket,
                            Key=old_key,
                        )
                    except Exception:
                        # The new checkpoint is already authoritative. An
                        # orphaned old generation is safe and can be cleaned
                        # independently.
                        pass
        state.generation = generation
        return {
            "latest": latest_key,
            "state": state_key,
            "partial_buffer": buffer_key,
        }
    finally:
        local_buffer.unlink(missing_ok=True)


def load_checkpoint(
    client: Any,
    settings: Any,
    *,
    source_manifest: Mapping[str, Any],
) -> ResumeState | None:
    latest_key = settings.key(RESUME_PREFIX, "LATEST.json")
    if not _exists(client, settings, latest_key):
        return None
    latest = json.loads(_get_bytes(client, settings, latest_key))
    state_bytes = _get_bytes(client, settings, latest["state_key"])
    if sha256_bytes(state_bytes) != latest["state_sha256"]:
        raise RuntimeError("resume state checksum is invalid")
    payload = json.loads(state_bytes)
    if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise RuntimeError("unsupported resume checkpoint version")
    if payload.get("pinned_dataset_revision") != source_manifest["pinned_revision"]:
        raise RuntimeError("resume dataset revision differs from source manifest")
    if payload.get("source_manifest_sha256") != source_manifest["manifest_sha256"]:
        raise RuntimeError("resume source manifest checksum differs")
    buffer_bytes = _get_bytes(client, settings, payload["partial_buffer_key"])
    if sha256_bytes(buffer_bytes) != payload["partial_buffer_object_sha256"]:
        raise RuntimeError("resume partial buffer object is corrupt")
    temporary = Path(os.environ.get("TMPDIR", "/tmp")) / f"resume-{uuid.uuid4().hex}.npy"
    try:
        temporary.write_bytes(buffer_bytes)
        partial = np.load(temporary, allow_pickle=False)
        if partial.dtype != np.uint16 or partial.ndim != 1:
            raise RuntimeError("resume partial buffer has invalid format")
        if sha256_bytes(partial.tobytes()) != payload["partial_token_buffer_sha256"]:
            raise RuntimeError("resume partial token checksum is invalid")
        if int(partial.size) != int(payload["partial_token_buffer_length"]):
            raise RuntimeError("resume partial token length is invalid")
        return ResumeState(
            source_ordinal=int(payload["current_source_ordinal"]),
            next_row=int(payload["next_row"]),
            processed_documents=int(payload["processed_document_count"]),
            processed_tokens=int(payload["processed_token_count"]),
            next_shard_index=int(payload["next_shard_index"]),
            shards=[dict(item) for item in payload["verified_shards"]],
            partial=np.array(partial, copy=True),
            generation=int(payload["generation"]),
        )
    finally:
        temporary.unlink(missing_ok=True)


def process_sequential_parquet(
    client: Any,
    settings: Any,
    work_dir: Path,
    *,
    source_git_sha: str | None,
    shard_size: int,
    vocabulary_size: int,
    upload_shard: Callable[[Path, str, int, bool], dict[str, Any]],
    resolver: Callable[[], dict[str, Any]] = build_source_manifest,
    downloader: Callable[[Mapping[str, Any], Path], Path] = download_with_resume,
    row_reader: Callable[..., Iterable[tuple[int, dict[str, Any]]]] = iter_parquet_rows,
    tokenizer_fn: Callable[[dict[str, Any]], np.ndarray] = tokenize,
    verified_remote_shards: Mapping[str, Mapping[str, Any]] | None = None,
    stop_after_verified_shards: int | None = None,
    checkpoint_documents: int = DEFAULT_CHECKPOINT_DOCUMENTS,
) -> dict[str, Any]:
    """Run the production path, optionally stopping after N verified shards."""

    manifest = ensure_source_manifest(client, settings, resolver=resolver)
    state = load_checkpoint(client, settings, source_manifest=manifest)
    if state is None:
        state = initial_resume_state()
    elif verified_remote_shards is not None:
        for record in state.shards:
            verified = verified_remote_shards.get(str(record["filename"]))
            if (
                verified is None
                or verified.get("sha256") != record.get("sha256")
                or int(verified.get("remote_bytes", -1))
                != int(record.get("remote_bytes", -2))
            ):
                raise RuntimeError(
                    f"resume shard is not remotely verified: {record['filename']}"
                )
    pending_chunks: deque[np.ndarray] = deque(
        [state.partial] if state.partial.size else []
    )
    pending_count = int(state.partial.size)
    checkpoint_paths: dict[str, str] | None = None
    downloaded_files = 0
    completed_files = 0
    work_dir.mkdir(parents=True, exist_ok=True)

    def checkpoint() -> None:
        nonlocal checkpoint_paths, pending_chunks
        state.partial = (
            np.concatenate(pending_chunks)
            if len(pending_chunks) > 1
            else (
                np.array(pending_chunks[0], copy=True)
                if pending_chunks
                else np.empty(0, dtype=np.uint16)
            )
        )
        pending_chunks = deque([state.partial] if state.partial.size else [])
        checkpoint_paths = persist_checkpoint(
            client,
            settings,
            state,
            source_manifest=manifest,
            source_git_sha=source_git_sha,
        )

    def drain_full_shards() -> bool:
        nonlocal pending_chunks, pending_count
        while pending_count >= shard_size:
            tokens = np.empty(shard_size, dtype=np.uint16)
            written = 0
            while written < shard_size:
                chunk = pending_chunks.popleft()
                amount = min(shard_size - written, int(chunk.size))
                tokens[written : written + amount] = chunk[:amount]
                written += amount
                if amount < chunk.size:
                    pending_chunks.appendleft(chunk[amount:])
            pending_count -= shard_size
            index = state.next_shard_index
            split = "val" if index == 0 else "train"
            filename = f"edufineweb_{split}_{index:06d}.npy"
            local = work_dir / filename
            with local.open("wb") as handle:
                np.save(handle, tokens, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            uploaded = False
            try:
                record = upload_shard(local, split, index, False)
                state.shards = [
                    item for item in state.shards if item["filename"] != filename
                ]
                state.shards.append(record)
                state.shards.sort(key=lambda item: int(item["index"]))
                state.next_shard_index += 1
                checkpoint()
                uploaded = True
            finally:
                if uploaded:
                    local.unlink(missing_ok=True)
            if (
                stop_after_verified_shards is not None
                and len(state.shards) >= stop_after_verified_shards
            ):
                return True
        return False

    files: Sequence[Mapping[str, Any]] = manifest["files"]
    for ordinal in range(state.source_ordinal, len(files)):
        source = files[ordinal]
        local_source = work_dir / "source" / Path(source["repository_path"]).name
        downloader(source, local_source)
        downloaded_files += 1
        start_row = state.next_row if ordinal == state.source_ordinal else 0
        rows_since_checkpoint = 0
        file_completed = False
        try:
            for row_number, document in row_reader(local_source, start_row=start_row):
                tokens = tokenizer_fn(document)
                if tokens.dtype != np.uint16 or tokens.ndim != 1:
                    raise ValueError("tokenizer returned invalid uint16 token data")
                if tokens.size and int(tokens.max()) >= vocabulary_size:
                    raise ValueError("token ID exceeds GPT-2 vocabulary")
                if tokens.size:
                    pending_chunks.append(tokens)
                    pending_count += int(tokens.size)
                state.processed_documents += 1
                state.processed_tokens += int(tokens.size)
                state.source_ordinal = ordinal
                state.next_row = row_number + 1
                rows_since_checkpoint += 1
                if drain_full_shards():
                    return {
                        "status": "smoke_complete",
                        "pinned_dataset_revision": manifest["pinned_revision"],
                        "source_manifest_file_count": manifest["file_count"],
                        "source_manifest_sha256": manifest["manifest_sha256"],
                        "source_files_downloaded": downloaded_files,
                        "source_files_processed": completed_files,
                        "processed_documents": state.processed_documents,
                        "processed_tokens": state.processed_tokens,
                        "verified_shards": state.shards,
                        "remaining_partial_tokens": pending_count,
                        "resume_checkpoint": checkpoint_paths,
                    }
                if rows_since_checkpoint >= checkpoint_documents:
                    checkpoint()
                    rows_since_checkpoint = 0
            state.source_ordinal = ordinal + 1
            state.next_row = 0
            checkpoint()
            file_completed = True
            completed_files += 1
        finally:
            if file_completed:
                local_source.unlink(missing_ok=True)

    if pending_count:
        checkpoint()
        index = state.next_shard_index
        split = "val" if index == 0 else "train"
        filename = f"edufineweb_{split}_{index:06d}.npy"
        local = work_dir / filename
        with local.open("wb") as handle:
            np.save(handle, state.partial, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        uploaded = False
        try:
            record = upload_shard(local, split, index, True)
            state.shards.append(record)
            state.next_shard_index += 1
            state.partial = np.empty(0, dtype=np.uint16)
            pending_chunks = deque()
            pending_count = 0
            checkpoint()
            uploaded = True
        finally:
            if uploaded:
                local.unlink(missing_ok=True)
    return {
        "status": "source_complete",
        "pinned_dataset_revision": manifest["pinned_revision"],
        "source_manifest_file_count": manifest["file_count"],
        "source_manifest_sha256": manifest["manifest_sha256"],
        "source_files_downloaded": downloaded_files,
        "source_files_processed": completed_files,
        "processed_documents": state.processed_documents,
        "processed_tokens": state.processed_tokens,
        "verified_shards": state.shards,
        "remaining_partial_tokens": pending_count,
        "resume_checkpoint": checkpoint_paths,
    }
