from __future__ import annotations

import hashlib
import io
import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pytest

from fineweb import validate_shard_filenames
from scripts.prepare_fineweb_to_s3 import (
    COMPLETE_NAME,
    METADATA_PREFIX,
    PROGRESS_NAME,
    SHARD_PREFIX,
    STATUS_PREFIX,
    Settings,
    already_complete,
    atomic_write_bytes,
    atomic_write_npy,
    build_manifest,
    calculate_disk_budget,
    create_and_upload_shards,
    final_validation,
    load_progress,
    load_source_dataset,
    parse_remote_shard_name,
    require_free_disk,
    run,
    select_work_base,
    sha256_bytes,
    sha256_file,
    storage_probe,
    upload_progress,
    upload_verified_shard,
    verify_existing_shards,
)


class MissingObject(Exception):
    def __init__(self) -> None:
        self.response = {
            "Error": {"Code": "NoSuchKey"},
            "ResponseMetadata": {"HTTPStatusCode": 404},
        }


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], dict[str, Any]] = {}
        self.fail_probe_read = False
        self.head_size_delta = 0
        self.uploads: list[str] = []

    def put_object(
        self,
        *,
        Bucket: str,
        Key: str,
        Body: bytes,
        Metadata: dict[str, str] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        content = bytes(Body)
        self.objects[(Bucket, Key)] = {
            "Body": content,
            "Metadata": dict(Metadata or {}),
            "ETag": hashlib.md5(content).hexdigest(),
        }
        return {}

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        try:
            item = self.objects[(Bucket, Key)]
        except KeyError as error:
            raise MissingObject() from error
        content = item["Body"]
        if self.fail_probe_read and "/probes/" in f"/{Key}/":
            content = b"mismatch"
        return {"Body": io.BytesIO(content)}

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        try:
            item = self.objects[(Bucket, Key)]
        except KeyError as error:
            raise MissingObject() from error
        return {
            "ContentLength": len(item["Body"]) + self.head_size_delta,
            "Metadata": dict(item["Metadata"]),
            "ETag": f'"{item["ETag"]}"',
        }

    def delete_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        self.objects.pop((Bucket, Key), None)
        return {}

    def list_objects_v2(self, *, Bucket: str, Prefix: str, **_: Any) -> dict[str, Any]:
        contents = [
            {"Key": key, "Size": len(item["Body"])}
            for (bucket, key), item in sorted(self.objects.items())
            if bucket == Bucket and key.startswith(Prefix)
        ]
        return {"Contents": contents, "IsTruncated": False}

    def upload_file(
        self,
        Filename: str,
        Bucket: str,
        Key: str,
        ExtraArgs: dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        self.uploads.append(Key)
        self.put_object(
            Bucket=Bucket,
            Key=Key,
            Body=Path(Filename).read_bytes(),
            Metadata=(ExtraArgs or {}).get("Metadata", {}),
        )

    def download_file(self, Bucket: str, Key: str, Filename: str) -> None:
        Path(Filename).write_bytes(self.objects[(Bucket, Key)]["Body"])


class UploadFailureS3(FakeS3):
    def upload_file(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("simulated upload failure")


@pytest.fixture
def environment(tmp_path: Path) -> dict[str, str]:
    return {
        "AWS_ACCESS_KEY_ID": "test-access-value",
        "AWS_SECRET_ACCESS_KEY": "test-secret-value",
        "AWS_REGION": "auto",
        "AWS_DEFAULT_REGION": "auto",
        "AWS_ENDPOINT_URL": "https://storage.example.invalid",
        "GMN_DATA_BUCKET": "test-bucket",
        "GMN_DATA_PREFIX": "v1",
        "GMN_PREPARATION_RUN_ID": "prep-test",
        "GMN_OUTPUT_DIR": str(tmp_path / "output"),
        "GMN_RESULT_PATH": str(tmp_path / "result.json"),
    }


@pytest.fixture
def settings(environment: dict[str, str]) -> Settings:
    return Settings.from_environment(environment)


def shard_key(settings: Settings, filename: str) -> str:
    return settings.key(SHARD_PREFIX, filename)


def stream_factory(values: list[np.ndarray]):
    @contextmanager
    def factory(_: Iterable[dict[str, Any]]):
        yield values

    return factory


def seed_shard(
    client: FakeS3,
    settings: Settings,
    tmp_path: Path,
    filename: str,
    tokens: np.ndarray,
    *,
    shard_size: int,
) -> dict[str, Any]:
    path = tmp_path / filename
    atomic_write_npy(path, tokens)
    split, index = parse_remote_shard_name(filename)
    return upload_verified_shard(
        client,
        settings,
        path,
        split=split,
        index=index,
        shard_size=shard_size,
        allow_partial=tokens.size < shard_size,
        vocabulary_size=50_257,
    )


def test_required_variable_validation(environment: dict[str, str]) -> None:
    del environment["AWS_SECRET_ACCESS_KEY"]
    with pytest.raises(
        ValueError, match="missing required environment variables: AWS_SECRET_ACCESS_KEY"
    ):
        Settings.from_environment(environment)


def test_missing_optional_job_id_is_null(environment: dict[str, str]) -> None:
    configured = Settings.from_environment(environment)
    assert configured.job_id is None


def test_injected_job_id_is_recorded(environment: dict[str, str]) -> None:
    environment["GIVEMEANODE_JOB_ID"] = "job-authoritative"
    configured = Settings.from_environment(environment)
    assert configured.job_id == "job-authoritative"


@pytest.mark.parametrize("value", ["", "null", "none", "NULL"])
def test_empty_or_null_job_id_is_optional(
    environment: dict[str, str],
    value: str,
) -> None:
    environment["GIVEMEANODE_JOB_ID"] = value
    configured = Settings.from_environment(environment)
    assert configured.job_id is None


def test_generated_preparation_run_id(environment: dict[str, str]) -> None:
    del environment["GMN_PREPARATION_RUN_ID"]
    configured = Settings.from_environment(environment)
    assert configured.preparation_run_id.startswith("prep-")
    assert len(configured.preparation_run_id) == 37


def test_supplied_preparation_run_id(environment: dict[str, str]) -> None:
    environment["GMN_PREPARATION_RUN_ID"] = "fineweb-run-20260728"
    configured = Settings.from_environment(environment)
    assert configured.preparation_run_id == "fineweb-run-20260728"


def test_credentials_never_appear_in_logs(
    environment: dict[str, str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    configured = Settings.from_environment(environment)
    with caplog.at_level(logging.INFO, logger="fineweb_s3"):
        logging.getLogger("fineweb_s3").info(
            "bucket=%s endpoint=%s region=%s",
            configured.bucket,
            configured.endpoint_hostname,
            configured.region,
        )
        logging.getLogger("fineweb_s3").error("failed (%s)", "RuntimeError")
    assert environment["AWS_ACCESS_KEY_ID"] not in caplog.text
    assert environment["AWS_SECRET_ACCESS_KEY"] not in caplog.text
    assert environment["GMN_DATA_BUCKET"] in caplog.text


def test_successful_s3_probe_leaves_no_object(settings: Settings) -> None:
    client = FakeS3()
    storage_probe(client, settings)
    assert not any("/probes/" in f"/{key}/" for _, key in client.objects)


def test_failed_s3_probe_raises_and_cleans_up(settings: Settings) -> None:
    client = FakeS3()
    client.fail_probe_read = True
    with pytest.raises(RuntimeError, match="probe byte verification"):
        storage_probe(client, settings)
    assert not any("/probes/" in f"/{key}/" for _, key in client.objects)


def test_atomic_local_writes(tmp_path: Path) -> None:
    binary = tmp_path / "metadata.json"
    atomic_write_bytes(binary, b'{"ok": true}\n')
    assert binary.read_bytes() == b'{"ok": true}\n'
    array_path = tmp_path / "edufineweb_val_000000.npy"
    expected = np.array([1, 2, 3], dtype=np.uint16)
    atomic_write_npy(array_path, expected)
    assert np.array_equal(np.load(array_path, allow_pickle=False), expected)
    assert not list(tmp_path.glob(".*.tmp"))


def test_sha256_generation(tmp_path: Path) -> None:
    content = b"fineweb-shard"
    path = tmp_path / "shard.npy"
    path.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    assert sha256_bytes(content) == expected
    assert sha256_file(path) == expected


def test_disk_budget_is_derived_from_shard_size() -> None:
    small = calculate_disk_budget(shard_size=4)
    normal = calculate_disk_budget(shard_size=100_000_000)
    assert normal.required_gib > small.required_gib
    assert normal.required_gib < 20
    assert normal.required_gib != 80
    assert normal.active_token_buffer_gib < 1
    assert normal.in_progress_shard_gib < 1


def test_disk_calculation_uses_actual_filesystem_availability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usage = type("Usage", (), {"total": 100 * 1024**3, "used": 88 * 1024**3, "free": 12 * 1024**3})()
    monkeypatch.setattr(
        "scripts.prepare_fineweb_to_s3.shutil.disk_usage",
        lambda _: usage,
    )
    monkeypatch.setattr(
        "scripts.prepare_fineweb_to_s3.filesystem_identity",
        lambda _: ("/dev/test", "/work"),
    )
    facts = require_free_disk(tmp_path, shard_size=100_000_000)
    assert facts.available_bytes == 12 * 1024**3
    assert facts.available_gib == 12
    assert facts.mount_path == "/work"
    assert facts.budget.required_gib < facts.available_gib


def test_disk_preflight_rejects_genuinely_insufficient_space(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usage = type("Usage", (), {"total": 100 * 1024**3, "used": 93 * 1024**3, "free": 7 * 1024**3})()
    monkeypatch.setattr(
        "scripts.prepare_fineweb_to_s3.shutil.disk_usage",
        lambda _: usage,
    )
    with pytest.raises(RuntimeError, match="calculated streaming preparation budget"):
        require_free_disk(tmp_path, shard_size=100_000_000)


def test_work_base_selects_filesystem_with_most_free_space(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    low = tmp_path / "low"
    high = tmp_path / "high"
    low.mkdir()
    high.mkdir()
    monkeypatch.setattr(
        "scripts.prepare_fineweb_to_s3.shutil.disk_usage",
        lambda path: type(
            "Usage",
            (),
            {"free": 1 if path == low else 2},
        )(),
    )
    assert select_work_base([low, high]) == high


def test_streaming_loader_requests_ordered_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}
    documents = [{"text": "first"}, {"text": "second"}]

    def fake_load_dataset(*args: Any, **kwargs: Any):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return iter(documents)

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    assert list(load_source_dataset()) == documents
    assert calls["args"] == ("HuggingFaceFW/fineweb-edu",)
    assert calls["kwargs"]["name"] == "sample-10BT"
    assert calls["kwargs"]["split"] == "train"
    assert calls["kwargs"]["streaming"] is True


def test_failed_upload_preserves_local_shard(
    settings: Settings,
    tmp_path: Path,
) -> None:
    client = UploadFailureS3()
    with pytest.raises(RuntimeError, match="simulated upload failure"):
        create_and_upload_shards(
            client,
            settings,
            {"shards": []},
            {},
            tmp_path,
            dataset=[],
            token_stream_factory=stream_factory(
                [np.array([1, 2, 3, 4], dtype=np.uint16)]
            ),
            shard_size=4,
            vocabulary_size=50_257,
        )
    assert (tmp_path / "edufineweb_val_000000.npy").exists()


def test_validation_downloads_are_cleaned(
    settings: Settings,
    tmp_path: Path,
) -> None:
    client = FakeS3()
    records = [
        seed_shard(
            client,
            settings,
            tmp_path,
            "edufineweb_val_000000.npy",
            np.array([1, 2, 3, 4], dtype=np.uint16),
            shard_size=4,
        ),
        seed_shard(
            client,
            settings,
            tmp_path,
            "edufineweb_train_000001.npy",
            np.array([5, 6, 7, 8], dtype=np.uint16),
            shard_size=4,
        ),
        seed_shard(
            client,
            settings,
            tmp_path,
            "edufineweb_train_000002.npy",
            np.array([9, 10], dtype=np.uint16),
            shard_size=4,
        ),
    ]
    final_validation(
        client,
        settings,
        records,
        tmp_path,
        shard_size=4,
        vocabulary_size=50_257,
    )
    assert not list(tmp_path.glob("sample-*.npy"))


def test_progressive_shard_upload_and_progress_updates(
    settings: Settings,
    tmp_path: Path,
) -> None:
    client = FakeS3()
    progress = {"shards": [], "complete": False}
    records = create_and_upload_shards(
        client,
        settings,
        progress,
        {},
        tmp_path,
        dataset=[],
        token_stream_factory=stream_factory(
            [
                np.array([1, 2, 3], dtype=np.uint16),
                np.array([4, 5, 6, 7, 8, 9], dtype=np.uint16),
            ]
        ),
        shard_size=4,
        vocabulary_size=50_257,
    )
    assert [item["filename"] for item in records] == [
        "edufineweb_val_000000.npy",
        "edufineweb_train_000001.npy",
        "edufineweb_train_000002.npy",
    ]
    assert len(client.uploads) == 3
    stored_progress = json.loads(
        client.objects[
            (settings.bucket, settings.key(METADATA_PREFIX, PROGRESS_NAME))
        ]["Body"]
    )
    assert [item["index"] for item in stored_progress["shards"]] == [0, 1, 2]


def test_remote_size_verification(settings: Settings, tmp_path: Path) -> None:
    client = FakeS3()
    record = seed_shard(
        client,
        settings,
        tmp_path,
        "edufineweb_val_000000.npy",
        np.array([1, 2, 3, 4], dtype=np.uint16),
        shard_size=4,
    )
    assert record["local_bytes"] == record["remote_bytes"]


def test_remote_size_mismatch_fails(settings: Settings, tmp_path: Path) -> None:
    client = FakeS3()
    client.head_size_delta = 1
    path = tmp_path / "edufineweb_val_000000.npy"
    atomic_write_npy(path, np.array([1, 2, 3, 4], dtype=np.uint16))
    with pytest.raises(RuntimeError, match="remote size mismatch"):
        upload_verified_shard(
            client,
            settings,
            path,
            split="val",
            index=0,
            shard_size=4,
            allow_partial=False,
            vocabulary_size=50_257,
        )


def test_resume_with_verified_shard_skips_upload(
    settings: Settings,
    tmp_path: Path,
) -> None:
    client = FakeS3()
    record = seed_shard(
        client,
        settings,
        tmp_path,
        "edufineweb_val_000000.npy",
        np.array([1, 2, 3, 4], dtype=np.uint16),
        shard_size=4,
    )
    client.uploads.clear()
    progress = {"shards": [record], "complete": False}
    verified = verify_existing_shards(
        client,
        settings,
        progress,
        tmp_path,
        shard_size=4,
        vocabulary_size=50_257,
    )
    create_and_upload_shards(
        client,
        settings,
        progress,
        verified,
        tmp_path,
        dataset=[],
        token_stream_factory=stream_factory(
            [np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint16)]
        ),
        shard_size=4,
        vocabulary_size=50_257,
    )
    assert shard_key(settings, "edufineweb_val_000000.npy") not in client.uploads
    assert shard_key(settings, "edufineweb_train_000001.npy") in client.uploads


def test_resume_with_missing_shard_regenerates_it(
    settings: Settings,
    tmp_path: Path,
) -> None:
    client = FakeS3()
    progress = {
        "shards": [
            {
                "filename": "edufineweb_val_000000.npy",
                "split": "val",
                "index": 0,
                "token_count": 4,
                "remote_bytes": 136,
                "sha256": "missing",
            }
        ],
        "complete": False,
    }
    verified = verify_existing_shards(
        client,
        settings,
        progress,
        tmp_path,
        shard_size=4,
        vocabulary_size=50_257,
    )
    create_and_upload_shards(
        client,
        settings,
        progress,
        verified,
        tmp_path,
        dataset=[],
        token_stream_factory=stream_factory(
            [np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint16)]
        ),
        shard_size=4,
        vocabulary_size=50_257,
    )
    assert shard_key(settings, "edufineweb_val_000000.npy") in client.uploads


def test_resume_with_corrupt_shard_regenerates_it(
    settings: Settings,
    tmp_path: Path,
) -> None:
    client = FakeS3()
    record = seed_shard(
        client,
        settings,
        tmp_path,
        "edufineweb_val_000000.npy",
        np.array([1, 2, 3, 4], dtype=np.uint16),
        shard_size=4,
    )
    stored = client.objects[
        (settings.bucket, shard_key(settings, "edufineweb_val_000000.npy"))
    ]
    stored["Body"] = b"corrupt"
    stored["Metadata"] = {}
    client.uploads.clear()
    progress = {"shards": [record], "complete": False}
    verified = verify_existing_shards(
        client,
        settings,
        progress,
        tmp_path,
        shard_size=4,
        vocabulary_size=50_257,
    )
    assert verified == {}
    create_and_upload_shards(
        client,
        settings,
        progress,
        verified,
        tmp_path,
        dataset=[],
        token_stream_factory=stream_factory(
            [np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint16)]
        ),
        shard_size=4,
        vocabulary_size=50_257,
    )
    assert shard_key(settings, "edufineweb_val_000000.npy") in client.uploads


def test_progress_manifest_round_trip(settings: Settings) -> None:
    client = FakeS3()
    progress = {"complete": False, "shards": [{"filename": "x", "index": 2}]}
    upload_progress(client, settings, progress)
    loaded = load_progress(client, settings)
    assert loaded["shards"][0]["filename"] == "x"


def test_manifest_generation(settings: Settings) -> None:
    records = [
        {
            "filename": "edufineweb_val_000000.npy",
            "split": "val",
            "index": 0,
            "token_count": 4,
            "remote_bytes": 136,
        },
        {
            "filename": "edufineweb_train_000001.npy",
            "split": "train",
            "index": 1,
            "token_count": 3,
            "remote_bytes": 134,
        },
    ]
    manifest = build_manifest(
        settings,
        records,
        {"numeric_filename_validation": True},
        preparation_git_sha="a" * 40,
        preparation_started_at="2026-07-28T00:00:00+00:00",
        preparation_finished_at="2026-07-28T01:00:00+00:00",
        shard_size=4,
    )
    assert manifest["source_dataset"] == "HuggingFaceFW/fineweb-edu"
    assert manifest["dataset_configuration"] == "sample-10BT"
    assert manifest["shard_count"] == 2
    assert manifest["total_token_count"] == 7
    assert manifest["training_shard_count"] == 1
    assert manifest["validation_shard_count"] == 1
    assert manifest["preparation_run_id"] == "prep-test"
    assert manifest["givemeanode_job_id"] is None


def test_numeric_shard_validation_is_used() -> None:
    ordered = validate_shard_filenames(
        [
            Path("edufineweb_train_000002.npy"),
            Path("edufineweb_val_000000.npy"),
            Path("edufineweb_train_000001.npy"),
        ]
    )
    assert [item.index for item in ordered] == [0, 1, 2]


def test_complete_written_only_after_full_success(
    settings: Settings,
) -> None:
    client = FakeS3()
    result = run(
        settings,
        client=client,
        dataset_loader=lambda: [],
        token_stream_factory=stream_factory(
            [np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.uint16)]
        ),
        shard_size=4,
    )
    complete_key = settings.key(STATUS_PREFIX, COMPLETE_NAME)
    assert result["status"] == "complete"
    assert (settings.bucket, complete_key) in client.objects
    marker = json.loads(client.objects[(settings.bucket, complete_key)]["Body"])
    assert already_complete(client, settings)
    assert marker["shard_count"] == 3
    assert marker["total_token_count"] == 9
    assert marker["preparation_run_id"] == "prep-test"
    assert marker["givemeanode_job_id"] is None
    assert (
        settings.bucket,
        settings.key("runs", "prep-test", "startup.json"),
    ) in client.objects
    assert (
        settings.bucket,
        settings.key("runs", "prep-test", "final_status.json"),
    ) in client.objects


def test_complete_absent_on_failure(settings: Settings) -> None:
    client = UploadFailureS3()
    with pytest.raises(RuntimeError, match="simulated upload failure"):
        run(
            settings,
            client=client,
            dataset_loader=lambda: [],
            token_stream_factory=stream_factory(
                [np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint16)]
            ),
            shard_size=4,
        )
    assert (
        settings.bucket,
        settings.key(STATUS_PREFIX, COMPLETE_NAME),
    ) not in client.objects


def test_malformed_remote_filename_is_rejected(
    settings: Settings,
    tmp_path: Path,
) -> None:
    client = FakeS3()
    client.put_object(
        Bucket=settings.bucket,
        Key=settings.key(SHARD_PREFIX, "edufineweb_test_000001.npy"),
        Body=b"unexpected",
    )
    with pytest.raises(ValueError, match="malformed or unexpected"):
        verify_existing_shards(
            client,
            settings,
            {"shards": []},
            tmp_path,
            shard_size=4,
            vocabulary_size=50_257,
        )
