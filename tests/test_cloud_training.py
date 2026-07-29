from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import train_gpt2_cloud as cloud


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.metadata: dict[str, dict[str, str]] = {}
        self.get_calls: list[tuple[str, str | None]] = []
        self.upload_calls = 0
        self.head_calls = 0

    def put_object(self, *, Bucket, Key, Body, Metadata=None, **kwargs):
        del Bucket, kwargs
        self.objects[Key] = Body.read() if hasattr(Body, "read") else bytes(Body)
        self.metadata[Key] = dict(Metadata or {})
        return {}

    def get_object(self, *, Bucket, Key, Range=None):
        del Bucket
        self.get_calls.append((Key, Range))
        content = self.objects[Key]
        response = {"Body": io.BytesIO(content)}
        if Range:
            offset = int(Range.removeprefix("bytes=").removesuffix("-"))
            response["Body"] = io.BytesIO(content[offset:])
            response["ContentRange"] = f"bytes {offset}-{len(content) - 1}/{len(content)}"
        return response

    def head_object(self, *, Bucket, Key):
        del Bucket
        self.head_calls += 1
        return {
            "ContentLength": len(self.objects[Key]),
            "Metadata": self.metadata.get(Key, {}),
        }

    def upload_file(self, Filename, Bucket, Key, ExtraArgs=None):
        del Bucket
        self.upload_calls += 1
        self.objects[Key] = Path(Filename).read_bytes()
        self.metadata[Key] = dict((ExtraArgs or {}).get("Metadata", {}))

    def list_objects_v2(self, *, Bucket, Prefix, **kwargs):
        del Bucket, kwargs
        return {
            "IsTruncated": False,
            "Contents": [
                {"Key": key} for key in sorted(self.objects) if key.startswith(Prefix)
            ],
        }

    def delete_object(self, *, Bucket, Key):
        del Bucket
        self.objects.pop(Key, None)
        self.metadata.pop(Key, None)


@pytest.fixture
def settings(tmp_path: Path) -> cloud.CloudSettings:
    return cloud.CloudSettings(
        bucket="bucket",
        dataset_prefix="v1",
        training_run_id="training-run-1",
        source_git_sha="a" * 40,
        endpoint_url="https://example.invalid",
        region="auto",
        data_root=tmp_path / "data",
        output_dir=tmp_path / "training",
        training_prefix="training/training-run-1",
        segment_seconds=600,
        download_workers=2,
        result_path=None,
        batch_output_dir=None,
    )


def npy_bytes(values: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, values)
    return buffer.getvalue()


def record(filename: str, key: str, content: bytes, token_count: int) -> dict:
    return {
        "filename": filename,
        "split": "val" if "_val_" in filename else "train",
        "index": int(filename.rsplit("_", 1)[1].split(".")[0]),
        "token_count": token_count,
        "remote_bytes": len(content),
        "sha256": cloud.sha256_bytes(content),
        "remote_key": key,
    }


def test_manifest_and_concurrent_dataset_staging(
    tmp_path: Path,
    settings: cloud.CloudSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeS3()
    val = npy_bytes(np.arange(8, dtype=np.uint16))
    train = npy_bytes(np.arange(12, dtype=np.uint16))
    records = [
        record("edufineweb_val_000000.npy", "v1/shards/edufineweb_val_000000.npy", val, 8),
        record(
            "edufineweb_train_000001.npy",
            "v1/shards/edufineweb_train_000001.npy",
            train,
            12,
        ),
    ]
    client.objects[records[0]["remote_key"]] = val
    client.objects[records[1]["remote_key"]] = train
    manifest = {
        "shard_count": 2,
        "total_token_count": 20,
        "shards": records,
    }
    manifest_bytes = json.dumps(manifest).encode()
    marker = {
        "manifest_path": "v1/metadata/dataset_manifest.json",
        "manifest_sha256": cloud.sha256_bytes(manifest_bytes),
    }
    client.objects[marker["manifest_path"]] = manifest_bytes
    client.objects["v1/status/COMPLETE"] = json.dumps(marker).encode()
    monkeypatch.setattr(cloud, "EXPECTED_MANIFEST_SHA256", marker["manifest_sha256"])
    monkeypatch.setattr(cloud, "EXPECTED_SHARDS", 2)
    monkeypatch.setattr(cloud, "EXPECTED_TOKENS", 20)

    _, verified = cloud.load_verified_dataset_manifest(client, settings)
    facts = cloud.stage_dataset(client, settings, verified)

    assert facts["shard_count"] == 2
    assert sorted(path.name for path in settings.data_root.iterdir()) == [
        "edufineweb_train_000001.npy",
        "edufineweb_val_000000.npy",
    ]
    assert not list(tmp_path.rglob("*.part"))


def test_checkpoint_upload_head_metadata_latest_pointer_and_resume(
    settings: cloud.CloudSettings,
) -> None:
    client = FakeS3()
    checkpoint = settings.output_dir / "checkpoints/checkpoint_step_000500.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint payload")
    sync = cloud.ArtifactSync(client, settings)

    sync.sync_once()

    assert checkpoint.is_file()
    latest_key = settings.training_key("checkpoints", "LATEST.json")
    latest = json.loads(client.objects[latest_key])
    assert latest["step"] == 500
    assert latest["sha256"] == cloud.sha256_file(checkpoint)
    assert not client.get_calls
    restored, restored_record = cloud.find_latest_verified_checkpoint(client, settings)
    assert restored is not None
    assert restored.read_bytes() == checkpoint.read_bytes()
    assert restored_record == latest


def test_legacy_checkpoint_without_object_metadata_resumes_after_full_hash(
    settings: cloud.CloudSettings,
) -> None:
    client = FakeS3()
    key = settings.training_key("checkpoints", "checkpoint_step_003000.pt")
    content = b"legacy verified checkpoint payload"
    client.objects[key] = content
    latest = {
        "key": key,
        "filename": Path(key).name,
        "bytes": len(content),
        "sha256": cloud.sha256_bytes(content),
        "step": 3000,
    }
    cloud.put_json(
        client,
        settings.bucket,
        settings.training_key("checkpoints", "LATEST.json"),
        latest,
    )

    restored, restored_record = cloud.find_latest_verified_checkpoint(client, settings)

    assert restored is not None
    assert restored.read_bytes() == content
    assert cloud.sha256_file(restored) == latest["sha256"]
    assert restored_record == latest


def test_checkpoint_upload_retries_transient_upload_failure(
    settings: cloud.CloudSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TransientUploadS3(FakeS3):
        def upload_file(self, Filename, Bucket, Key, ExtraArgs=None):
            if self.upload_calls == 0:
                self.upload_calls += 1
                raise OSError("temporary upload failure")
            super().upload_file(Filename, Bucket, Key, ExtraArgs)

    monkeypatch.setattr(cloud.time, "sleep", lambda _: None)
    client = TransientUploadS3()
    checkpoint = settings.output_dir / "checkpoints/checkpoint_step_000500.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint payload")

    record = cloud.upload_verified_file(
        client,
        settings,
        checkpoint,
        settings.training_key("checkpoints", checkpoint.name),
    )

    assert client.upload_calls == 2
    assert record["sha256"] == cloud.sha256_file(checkpoint)


def test_checkpoint_upload_retries_transient_head_failure(
    settings: cloud.CloudSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TransientHeadS3(FakeS3):
        def head_object(self, *, Bucket, Key):
            if self.head_calls == 0:
                self.head_calls += 1
                raise OSError("temporary HEAD failure")
            return super().head_object(Bucket=Bucket, Key=Key)

    monkeypatch.setattr(cloud.time, "sleep", lambda _: None)
    client = TransientHeadS3()
    checkpoint = settings.output_dir / "checkpoints/checkpoint_step_000500.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint payload")

    cloud.upload_verified_file(
        client,
        settings,
        checkpoint,
        settings.training_key("checkpoints", checkpoint.name),
    )

    assert client.head_calls == 2


def test_latest_pointer_not_advanced_before_checkpoint_verification(
    settings: cloud.CloudSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeS3()
    checkpoint = settings.output_dir / "checkpoints/checkpoint_step_000500.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint payload")
    original_head = client.head_object

    def corrupt_head(*, Bucket, Key):
        head = original_head(Bucket=Bucket, Key=Key)
        if Key.endswith(".pt"):
            head["Metadata"] = {"sha256": "0" * 64}
        return head

    monkeypatch.setattr(client, "head_object", corrupt_head)
    monkeypatch.setattr(cloud.time, "sleep", lambda _: None)
    sync = cloud.ArtifactSync(client, settings)

    with pytest.raises(RuntimeError, match="checksum metadata mismatch"):
        sync.sync_once()

    assert settings.training_key("checkpoints", "LATEST.json") not in client.objects


def test_synchronizer_recovers_after_temporary_error(
    settings: cloud.CloudSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sync = cloud.ArtifactSync(FakeS3(), settings)
    calls = 0
    waits = iter((False, False, True))

    def sync_once() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("temporary sync error")

    monkeypatch.setattr(sync, "sync_once", sync_once)
    monkeypatch.setattr(sync.stop_event, "wait", lambda _: next(waits))
    monkeypatch.setattr(cloud.time, "sleep", lambda _: None)

    sync._run()

    assert calls == 3
    assert sync.error is None


def test_synchronizer_failure_is_bounded(
    settings: cloud.CloudSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sync = cloud.ArtifactSync(FakeS3(), settings)
    calls = 0

    def sync_once() -> None:
        nonlocal calls
        calls += 1
        raise OSError("persistent sync error")

    monkeypatch.setattr(sync, "sync_once", sync_once)
    monkeypatch.setattr(sync.stop_event, "wait", lambda _: False)
    monkeypatch.setattr(cloud.time, "sleep", lambda _: None)

    sync._run()

    assert calls == cloud.SYNC_FAILURE_LIMIT
    assert isinstance(sync.error, OSError)


def test_interrupted_checkpoint_download_resumes_with_range(
    settings: cloud.CloudSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InterruptedBody(io.BytesIO):
        def __init__(self, content: bytes) -> None:
            super().__init__(content)
            self.failed = False

        def read(self, size=-1):
            if self.failed:
                raise OSError("interrupted response stream")
            chunk = super().read(min(size, 5) if size >= 0 else 5)
            self.failed = True
            return chunk

    class InterruptedDownloadS3(FakeS3):
        def get_object(self, *, Bucket, Key, Range=None):
            if Key.endswith(".pt") and Range is None:
                self.get_calls.append((Key, Range))
                return {"Body": InterruptedBody(self.objects[Key])}
            return super().get_object(Bucket=Bucket, Key=Key, Range=Range)

    monkeypatch.setattr(cloud.time, "sleep", lambda _: None)
    client = InterruptedDownloadS3()
    key = settings.training_key("checkpoints", "checkpoint_step_003000.pt")
    content = b"verified checkpoint payload"
    client.objects[key] = content
    client.metadata[key] = {"sha256": cloud.sha256_bytes(content)}
    latest = {
        "key": key,
        "filename": Path(key).name,
        "bytes": len(content),
        "sha256": cloud.sha256_bytes(content),
        "step": 3000,
    }
    cloud.put_json(
        client,
        settings.bucket,
        settings.training_key("checkpoints", "LATEST.json"),
        latest,
    )

    restored, _ = cloud.find_latest_verified_checkpoint(client, settings)

    assert restored is not None
    assert restored.read_bytes() == content
    assert (key, "bytes=5-") in client.get_calls


def test_corrupt_resumed_checkpoint_download_is_rejected(
    settings: cloud.CloudSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cloud.time, "sleep", lambda _: None)
    client = FakeS3()
    key = settings.training_key("checkpoints", "checkpoint_step_003000.pt")
    expected = b"expected checkpoint payload"
    corrupt = b"corrupt! checkpoint payload"
    assert len(corrupt) == len(expected)
    client.objects[key] = corrupt
    client.metadata[key] = {"sha256": cloud.sha256_bytes(expected)}
    latest = {
        "key": key,
        "filename": Path(key).name,
        "bytes": len(expected),
        "sha256": cloud.sha256_bytes(expected),
        "step": 3000,
    }
    cloud.put_json(
        client,
        settings.bucket,
        settings.training_key("checkpoints", "LATEST.json"),
        latest,
    )
    temporary = (
        settings.output_dir
        / "resume"
        / "checkpoint_step_003000.part"
    )
    temporary.parent.mkdir(parents=True)
    temporary.write_bytes(corrupt[:5])

    with pytest.raises(RuntimeError, match="failed download verification"):
        cloud.find_latest_verified_checkpoint(client, settings)

    assert not temporary.exists()


def test_only_completed_training_publishes_complete(
    settings: cloud.CloudSettings,
) -> None:
    client = FakeS3()
    cloud.publish_segment_status(client, settings, {"status": "paused", "step": 500})
    assert settings.training_key("status", "COMPLETE") not in client.objects

    cloud.publish_segment_status(
        client, settings, {"status": "completed", "step": 19_073}
    )
    assert settings.training_key("status", "COMPLETE") in client.objects


def test_training_command_is_production_and_resumable(
    settings: cloud.CloudSettings,
) -> None:
    resume = settings.output_dir / "resume/checkpoint_step_000500.pt"
    command = cloud.training_command(settings, resume)
    rendered = " ".join(command)
    assert "--max-steps 19073" in rendered
    assert "--checkpoint-interval 500" in rendered
    assert "--wandb-mode online" in rendered
    assert f"--resume {resume}" in rendered
    assert "smoke" not in rendered


def test_sanitized_redacts_credential_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "top-secret-value")
    assert "top-secret-value" not in cloud.sanitized("failure top-secret-value")
