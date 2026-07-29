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

    def put_object(self, *, Bucket, Key, Body, Metadata=None, **kwargs):
        del Bucket, kwargs
        self.objects[Key] = Body.read() if hasattr(Body, "read") else bytes(Body)
        self.metadata[Key] = dict(Metadata or {})
        return {}

    def get_object(self, *, Bucket, Key):
        del Bucket
        return {"Body": io.BytesIO(self.objects[Key])}

    def head_object(self, *, Bucket, Key):
        del Bucket
        return {
            "ContentLength": len(self.objects[Key]),
            "Metadata": self.metadata.get(Key, {}),
        }

    def upload_file(self, Filename, Bucket, Key, ExtraArgs=None):
        del Bucket
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


def test_checkpoint_upload_remote_hash_latest_pointer_and_resume(
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
    restored, restored_record = cloud.find_latest_verified_checkpoint(client, settings)
    assert restored is not None
    assert restored.read_bytes() == checkpoint.read_bytes()
    assert restored_record == latest


def test_remote_checkpoint_corruption_is_rejected(
    settings: cloud.CloudSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeS3()
    checkpoint = settings.output_dir / "checkpoints/checkpoint_step_000500.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint payload")

    monkeypatch.setattr(cloud, "_remote_sha256", lambda *args: "0" * 64)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        cloud.upload_verified_file(
            client,
            settings,
            checkpoint,
            settings.training_key("checkpoints", checkpoint.name),
        )
    assert checkpoint.is_file()


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
