#!/usr/bin/env python3
"""Run the original FineWeb loader/tokenizer with a verified S3 shard hook."""

from __future__ import annotations

import logging
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping

from fineweb import (
    DATASET_ID,
    DEFAULT_DATASET_CONFIG,
    DEFAULT_SHARD_SIZE,
    prepare_dataset,
    tokenizer,
)
from scripts.prepare_fineweb_to_s3 import (
    COMPLETE_NAME,
    METADATA_PREFIX,
    REPORT_NAME,
    STATUS_PREFIX,
    Settings,
    build_manifest,
    build_report,
    configure_logging,
    create_s3_client,
    final_validation,
    git_sha,
    load_progress,
    object_exists,
    publish_final_metadata,
    put_json,
    replace_progress_record,
    sha256_file,
    stop_after_verified_shards,
    storage_probe,
    upload_progress,
    upload_text,
    upload_verified_shard,
    utc_now,
    verify_existing_shards,
    write_small_output,
)

LOGGER = logging.getLogger("fineweb_s3.original")


def _redact_sensitive_text(value: str, settings: Settings | None = None) -> str:
    secrets = {
        os.environ.get("AWS_ACCESS_KEY_ID", ""),
        os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
    }
    if settings is not None:
        secrets.update({settings.access_key_id, settings.secret_access_key})
    redacted = value
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "<redacted>")
    return redacted


def _sanitized_exception_message(
    error: BaseException,
    settings: Settings | None = None,
) -> str:
    message = " ".join(str(error).split()) or type(error).__name__
    return _redact_sensitive_text(message, settings)[:1000]


def _sanitized_traceback(
    error: BaseException,
    settings: Settings | None = None,
) -> str:
    rendered = "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )
    return _redact_sensitive_text(rendered, settings)


def _sanitized_excepthook(
    exception_type: type[BaseException],
    error: BaseException,
    exception_traceback: Any,
) -> None:
    """Preserve the uncaught traceback while redacting credential values."""

    if issubclass(exception_type, KeyboardInterrupt):
        sys.__excepthook__(exception_type, error, exception_traceback)
        return
    rendered = "".join(
        traceback.format_exception(exception_type, error, exception_traceback)
    )
    sys.stderr.write(_redact_sensitive_text(rendered))


def run_original_loader(
    settings: Settings,
    *,
    client: Any | None = None,
    shard_size: int = DEFAULT_SHARD_SIZE,
    workers: int = 32,
    work_root: Path | None = None,
) -> dict[str, Any]:
    """Preserve ``fineweb.prepare_dataset`` and persist each closed shard."""

    active_client = client or create_s3_client(settings)
    source_sha = git_sha()
    started_at = utc_now()
    smoke_limit = stop_after_verified_shards()
    selected_work_root = work_root or (
        Path("/workspace") if Path("/workspace").is_dir() else Path.cwd()
    )
    work_dir = (
        selected_work_root / "fineweb-original-s3" / settings.preparation_run_id
    )
    shard_dir = work_dir / "edu_fineweb10B"
    shard_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(work_dir)
    disk = {
        "path": str(work_dir),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "available_bytes": usage.free,
        "available_gib": round(usage.free / (1024**3), 3),
        "model": (
            "full Hugging Face dataset cache with progressive token-shard "
            "upload and verified local-shard deletion"
        ),
    }
    progress: dict[str, Any] = {
        "schema_version": 1,
        "source_dataset": DATASET_ID,
        "dataset_configuration": DEFAULT_DATASET_CONFIG,
        "shard_size": shard_size,
        "git_sha": source_sha,
        "preparation_run_id": settings.preparation_run_id,
        "givemeanode_job_id": settings.job_id,
        "complete": False,
        "shards": [],
    }
    phase = "storage_probe"
    manifest: dict[str, Any] | None = None
    cleanup_work_dir = False
    try:
        storage_probe(active_client, settings)
        phase = "resume_validation"
        progress = load_progress(active_client, settings)
        progress.update(
            {
                "source_dataset": DATASET_ID,
                "dataset_configuration": DEFAULT_DATASET_CONFIG,
                "shard_size": shard_size,
                "git_sha": source_sha,
                "preparation_run_id": settings.preparation_run_id,
                "givemeanode_job_id": settings.job_id,
                "complete": False,
            }
        )
        vocabulary_size = tokenizer().n_vocab
        verified = verify_existing_shards(
            active_client,
            settings,
            progress,
            work_dir,
            shard_size=shard_size,
            vocabulary_size=vocabulary_size,
        )
        upload_progress(active_client, settings, progress)
        verified_this_run: list[dict[str, Any]] = []

        def persist_shard(
            path: Path,
            split: str,
            index: int,
            token_count: int,
        ) -> bool:
            existing = verified.get(path.name)
            local_bytes = path.stat().st_size
            local_sha = sha256_file(path)
            if (
                existing is not None
                and int(existing["token_count"]) == token_count
                and int(existing["remote_bytes"]) == local_bytes
                and existing["sha256"] == local_sha
            ):
                record = dict(existing)
            else:
                record = upload_verified_shard(
                    active_client,
                    settings,
                    path,
                    split=split,
                    index=index,
                    shard_size=shard_size,
                    allow_partial=token_count < shard_size,
                    vocabulary_size=vocabulary_size,
                )
            replace_progress_record(progress, record)
            upload_progress(active_client, settings, progress)
            verified_this_run.append(record)
            path.unlink()
            return smoke_limit is not None and len(verified_this_run) >= smoke_limit

        phase = "original_fineweb_preparation"
        prepare_dataset(
            output_dir=shard_dir,
            dataset_config=DEFAULT_DATASET_CONFIG,
            shard_size=shard_size,
            workers=workers,
            shard_callback=persist_shard,
        )

        records = sorted(
            (dict(item) for item in progress["shards"]),
            key=lambda item: int(item["index"]),
        )
        if smoke_limit is not None:
            if len(verified_this_run) < smoke_limit:
                raise RuntimeError("original loader ended before the smoke shard limit")
            complete_key = settings.key(STATUS_PREFIX, COMPLETE_NAME)
            if object_exists(active_client, settings, complete_key):
                raise RuntimeError("smoke mode requires COMPLETE to be absent")
            first = verified_this_run[0]
            result = {
                "status": "smoke_complete",
                "bucket": settings.bucket,
                "prefix": settings.prefix,
                "git_sha": source_sha,
                "preparation_run_id": settings.preparation_run_id,
                "givemeanode_job_id": settings.job_id,
                "storage_probe_passed": True,
                "fineweb_loading": "completed_until_first_shard",
                "tokenization": "completed_first_full_shard",
                "workers": workers,
                "disk": disk,
                "verified_shard": first,
                "complete_marker_absent": True,
                "finished_at": utc_now(),
            }
            write_small_output(settings, result)
            cleanup_work_dir = True
            return result

        phase = "final_validation"
        validation = final_validation(
            active_client,
            settings,
            records,
            work_dir,
            shard_size=shard_size,
            vocabulary_size=vocabulary_size,
        )
        finished_at = utc_now()
        manifest = build_manifest(
            settings,
            records,
            validation,
            preparation_git_sha=source_sha,
            preparation_started_at=started_at,
            preparation_finished_at=finished_at,
            shard_size=shard_size,
        )
        manifest.update(
            {
                "fineweb_loader": "datasets.load_dataset",
                "multiprocessing_workers": workers,
                "preparation_entry_point": (
                    "python -u -m scripts.prepare_fineweb_original_to_s3"
                ),
                "disk_preflight": disk,
            }
        )
        phase = "final_metadata_publication"
        progress["complete"] = True
        upload_progress(active_client, settings, progress)
        marker = publish_final_metadata(active_client, settings, manifest)
        result = {
            "status": "complete",
            "bucket": settings.bucket,
            "prefix": settings.prefix,
            "git_sha": source_sha,
            "preparation_run_id": settings.preparation_run_id,
            "manifest_path": marker["manifest_path"],
            "manifest_sha256": marker["manifest_sha256"],
            "shard_count": marker["shard_count"],
            "total_token_count": marker["total_token_count"],
            "total_bytes": marker["total_bytes"],
            "disk": disk,
        }
        write_small_output(settings, result)
        cleanup_work_dir = True
        return result
    except Exception as error:
        error_message = _sanitized_exception_message(error, settings)
        LOGGER.error(
            "dataset preparation failed phase=%s type=%s message=%s\n%s",
            phase,
            type(error).__name__,
            error_message,
            _sanitized_traceback(error, settings),
        )
        progress["complete"] = False
        try:
            upload_progress(active_client, settings, progress)
            failure_report = build_report(
                manifest,
                complete=False,
                failure_type=type(error).__name__,
            )
            failure_report += (
                f"- Failing phase: {phase}\n"
                f"- Error message: {error_message}\n"
            )
            upload_text(
                active_client,
                settings,
                settings.key(METADATA_PREFIX, REPORT_NAME),
                failure_report,
                content_type="text/markdown",
            )
            failure_status = {
                "status": "incomplete",
                "bucket": settings.bucket,
                "prefix": settings.prefix,
                "git_sha": source_sha,
                "preparation_run_id": settings.preparation_run_id,
                "givemeanode_job_id": settings.job_id,
                "failing_phase": phase,
                "failure_type": type(error).__name__,
                "error_message": error_message,
                "started_at": started_at,
                "finished_at": utc_now(),
                "disk": disk,
            }
            put_json(
                active_client,
                settings,
                settings.key(
                    "runs",
                    settings.preparation_run_id,
                    "final_status.json",
                ),
                failure_status,
            )
            write_small_output(
                settings,
                failure_status,
            )
        except Exception as publication_error:
            LOGGER.error(
                "failed to publish incomplete preparation metadata "
                "type=%s message=%s\n%s",
                type(publication_error).__name__,
                _sanitized_exception_message(publication_error, settings),
                _sanitized_traceback(publication_error, settings),
            )
        raise
    finally:
        if cleanup_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)


def main() -> int:
    configure_logging()
    sys.excepthook = _sanitized_excepthook
    settings = Settings.from_environment()
    run_original_loader(settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
