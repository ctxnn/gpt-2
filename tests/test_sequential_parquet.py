from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.prepare_fineweb_to_s3 import Settings
from scripts.sequential_parquet import (
    SOURCE_MANIFEST_KEY,
    PINNED_DATASET_REVISION,
    build_source_manifest,
    canonical_json_bytes,
    download_with_resume,
    iter_parquet_rows,
    load_checkpoint,
    natural_key,
    process_sequential_parquet,
    sha256_bytes,
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
        }
        return {}

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        try:
            content = self.objects[(Bucket, Key)]["Body"]
        except KeyError as error:
            raise MissingObject() from error
        return {"Body": io.BytesIO(content)}

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        try:
            item = self.objects[(Bucket, Key)]
        except KeyError as error:
            raise MissingObject() from error
        return {
            "ContentLength": len(item["Body"]),
            "Metadata": dict(item["Metadata"]),
            "ETag": '"fake"',
        }

    def upload_file(
        self,
        Filename: str,
        Bucket: str,
        Key: str,
        ExtraArgs: dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        self.put_object(
            Bucket=Bucket,
            Key=Key,
            Body=Path(Filename).read_bytes(),
            Metadata=(ExtraArgs or {}).get("Metadata"),
        )

    def delete_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        self.objects.pop((Bucket, Key), None)
        return {}


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings.from_environment(
        {
            "AWS_ACCESS_KEY_ID": "access-not-secret",
            "AWS_SECRET_ACCESS_KEY": "secret-not-secret",
            "AWS_REGION": "auto",
            "AWS_DEFAULT_REGION": "auto",
            "AWS_ENDPOINT_URL": "https://storage.example.invalid",
            "GMN_DATA_BUCKET": "bucket",
            "GMN_DATA_PREFIX": "v1",
            "GMN_PREPARATION_RUN_ID": "sequential-test",
            "GMN_OUTPUT_DIR": str(tmp_path / "out"),
        }
    )


class Sibling:
    def __init__(self, path: str, size: int) -> None:
        self.rfilename = path
        self.size = size
        self.blob_id = f"blob-{path}"
        self.lfs = None
        self.xet = None


class Info:
    sha = PINNED_DATASET_REVISION
    siblings = [
        Sibling("sample/10BT/010_00000.parquet", 10),
        Sibling("README.md", 1),
        Sibling("sample/100BT/001_00000.parquet", 20),
        Sibling("sample/10BT/002_00000.parquet", 30),
    ]


class Api:
    def dataset_info(self, *_: Any, **kwargs: Any) -> Info:
        assert kwargs == {
            "revision": PINNED_DATASET_REVISION,
            "files_metadata": True,
        }
        return Info()


def test_source_manifest_is_pinned_filtered_and_naturally_sorted() -> None:
    manifest = build_source_manifest(api=Api())
    assert manifest["pinned_revision"] == PINNED_DATASET_REVISION
    assert [item["repository_path"] for item in manifest["files"]] == [
        "sample/10BT/002_00000.parquet",
        "sample/10BT/010_00000.parquet",
    ]
    assert [item["ordinal"] for item in manifest["files"]] == [0, 1]
    assert all(
        PINNED_DATASET_REVISION in item["download_url"]
        for item in manifest["files"]
    )
    assert natural_key("file10") > natural_key("file2")


class Response:
    def __init__(self, body: bytes, status_code: int) -> None:
        self.body = body
        self.status_code = status_code

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, _: int):
        yield self.body


class Session:
    def __init__(self, responses: list[Response | Exception]) -> None:
        self.responses = responses
        self.headers: list[dict[str, str]] = []

    def get(self, _: str, *, headers: dict[str, str], **__: Any) -> Response:
        self.headers.append(headers)
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def test_part_download_resumes_with_range(tmp_path: Path) -> None:
    destination = tmp_path / "source.parquet"
    partial = destination.with_suffix(".parquet.part")
    partial.write_bytes(b"abc")
    session = Session([Response(b"def", 206)])
    source = {
        "download_url": "https://example.invalid/source",
        "expected_bytes": 6,
        "checksum": hashlib.sha256(b"abcdef").hexdigest(),
    }
    download_with_resume(source, destination, session=session)
    assert session.headers == [{"Range": "bytes=3-"}]
    assert destination.read_bytes() == b"abcdef"
    assert not partial.exists()


def test_download_retries_are_bounded(tmp_path: Path) -> None:
    session = Session([OSError("one"), OSError("two"), Response(b"ok", 200)])
    delays: list[float] = []
    download_with_resume(
        {"download_url": "https://example.invalid", "expected_bytes": 2},
        tmp_path / "source.parquet",
        session=session,
        attempts=3,
        sleep=delays.append,
    )
    assert delays == [2.0, 4.0]


def test_incremental_parquet_rows_preserve_order(tmp_path: Path) -> None:
    path = tmp_path / "ordered.parquet"
    pq.write_table(pa.table({"text": ["zero", "one", "two", "three"]}), path)
    assert list(iter_parquet_rows(path, start_row=1, batch_size=2)) == [
        (1, {"text": "one"}),
        (2, {"text": "two"}),
        (3, {"text": "three"}),
    ]


def source_manifest(paths: list[str]) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "source_dataset": "HuggingFaceFW/fineweb-edu",
        "dataset_configuration": "sample-10BT",
        "source_subdirectory": "sample/10BT",
        "requested_revision": "main",
        "pinned_revision": "b" * 40,
        "file_count": len(paths),
        "files": [
            {
                "ordinal": ordinal,
                "repository_path": path,
                "download_url": f"https://example.invalid/{path}",
                "expected_bytes": None,
                "checksum": None,
                "hub_object_id": None,
                "pinned_revision": "b" * 40,
            }
            for ordinal, path in enumerate(paths)
        ],
    }
    manifest["manifest_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    return manifest


def fake_downloader(files: Mapping[str, list[str]], downloaded: list[str]):
    def download(source: Mapping[str, Any], destination: Path) -> Path:
        name = Path(source["repository_path"]).name
        downloaded.append(name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("\n".join(files[name]))
        return destination

    return download


def fake_reader(path: Path, *, start_row: int = 0):
    for index, text in enumerate(path.read_text().splitlines()):
        if index >= start_row:
            yield index, {"text": text}


def number_tokenizer(document: dict[str, Any]) -> np.ndarray:
    return np.array(
        [int(value) for value in document["text"].split(",") if value],
        dtype=np.uint16,
    )


def uploader(records: list[dict[str, Any]], local_presence: list[bool]):
    def upload(path: Path, split: str, index: int, _: bool) -> dict[str, Any]:
        local_presence.append(path.exists())
        tokens = np.load(path, allow_pickle=False)
        record = {
            "filename": path.name,
            "split": split,
            "index": index,
            "token_count": int(tokens.size),
            "local_bytes": path.stat().st_size,
            "remote_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "etag": "fake",
            "remote_key": f"v1/shards/{path.name}",
            "upload_timestamp": "now",
        }
        records.append(record)
        return record

    return upload


def test_smoke_shard_spans_files_and_checkpoint_resumes_exactly(
    settings: Settings,
    tmp_path: Path,
) -> None:
    client = FakeS3()
    manifest = source_manifest(
        ["sample/10BT/000.parquet", "sample/10BT/001.parquet"]
    )
    files = {
        "000.parquet": ["1,2"],
        "001.parquet": ["3,4,5", "6,7"],
    }
    downloaded: list[str] = []
    records: list[dict[str, Any]] = []
    local_presence: list[bool] = []
    first = process_sequential_parquet(
        client,
        settings,
        tmp_path / "work",
        source_git_sha="c" * 40,
        shard_size=4,
        vocabulary_size=100,
        upload_shard=uploader(records, local_presence),
        resolver=lambda: manifest,
        downloader=fake_downloader(files, downloaded),
        row_reader=fake_reader,
        tokenizer_fn=number_tokenizer,
        stop_after_verified_shards=1,
        checkpoint_documents=1,
    )
    assert first["status"] == "smoke_complete"
    assert first["verified_shards"][0]["filename"] == "edufineweb_val_000000.npy"
    assert first["verified_shards"][0]["token_count"] == 4
    assert first["remaining_partial_tokens"] == 1
    assert local_presence == [True]
    assert not (tmp_path / "work" / "edufineweb_val_000000.npy").exists()
    assert (settings.bucket, settings.key("status", "COMPLETE")) not in client.objects

    resumed_records: list[dict[str, Any]] = []
    second = process_sequential_parquet(
        client,
        settings,
        tmp_path / "work-2",
        source_git_sha="c" * 40,
        shard_size=4,
        vocabulary_size=100,
        upload_shard=uploader(resumed_records, []),
        resolver=lambda: pytest.fail("stored manifest should be reused"),
        downloader=fake_downloader(files, downloaded),
        row_reader=fake_reader,
        tokenizer_fn=number_tokenizer,
        checkpoint_documents=1,
    )
    assert second["status"] == "source_complete"
    assert second["processed_tokens"] == 7
    assert [item["token_count"] for item in second["verified_shards"]] == [4, 3]
    assert resumed_records[0]["filename"] == "edufineweb_train_000001.npy"


def test_multiple_shards_from_one_source_and_final_partial(
    settings: Settings,
    tmp_path: Path,
) -> None:
    client = FakeS3()
    manifest = source_manifest(["sample/10BT/000.parquet"])
    records: list[dict[str, Any]] = []
    result = process_sequential_parquet(
        client,
        settings,
        tmp_path / "work",
        source_git_sha=None,
        shard_size=3,
        vocabulary_size=100,
        upload_shard=uploader(records, []),
        resolver=lambda: manifest,
        downloader=fake_downloader({"000.parquet": ["1,2,3,4,5,6,7"]}, []),
        row_reader=fake_reader,
        tokenizer_fn=number_tokenizer,
    )
    assert [item["token_count"] for item in records] == [3, 3, 1]
    assert [item["filename"] for item in records] == [
        "edufineweb_val_000000.npy",
        "edufineweb_train_000001.npy",
        "edufineweb_train_000002.npy",
    ]
    assert result["remaining_partial_tokens"] == 0


def test_corrupt_partial_checkpoint_is_rejected(
    settings: Settings,
    tmp_path: Path,
) -> None:
    client = FakeS3()
    manifest = source_manifest(["sample/10BT/000.parquet"])
    process_sequential_parquet(
        client,
        settings,
        tmp_path / "work",
        source_git_sha=None,
        shard_size=2,
        vocabulary_size=100,
        upload_shard=uploader([], []),
        resolver=lambda: manifest,
        downloader=fake_downloader({"000.parquet": ["1,2,3"]}, []),
        row_reader=fake_reader,
        tokenizer_fn=number_tokenizer,
        stop_after_verified_shards=1,
    )
    latest = json.loads(
        client.objects[
            (settings.bucket, settings.key("resume", "LATEST.json"))
        ]["Body"]
    )
    client.objects[(settings.bucket, latest["partial_buffer_key"])]["Body"] += b"x"
    with pytest.raises(RuntimeError, match="partial buffer object is corrupt"):
        load_checkpoint(client, settings, source_manifest=manifest)


class FailingLatestS3(FakeS3):
    def put_object(self, *, Key: str, **kwargs: Any) -> dict[str, Any]:
        if Key.endswith("resume/LATEST.json"):
            raise RuntimeError("checkpoint pointer failure")
        return super().put_object(Key=Key, **kwargs)


def test_source_deleted_only_after_checkpoint_persistence(
    settings: Settings,
    tmp_path: Path,
) -> None:
    client = FailingLatestS3()
    manifest = source_manifest(["sample/10BT/000.parquet"])
    work = tmp_path / "work"
    with pytest.raises(RuntimeError, match="checkpoint pointer failure"):
        process_sequential_parquet(
            client,
            settings,
            work,
            source_git_sha=None,
            shard_size=100,
            vocabulary_size=100,
            upload_shard=uploader([], []),
            resolver=lambda: manifest,
            downloader=fake_downloader({"000.parquet": ["1,2"]}, []),
            row_reader=fake_reader,
            tokenizer_fn=number_tokenizer,
        )
    assert (work / "source" / "000.parquet").exists()


def test_credentials_do_not_enter_source_manifest(settings: Settings) -> None:
    manifest = source_manifest(["sample/10BT/000.parquet"])
    encoded = canonical_json_bytes(manifest)
    assert settings.access_key_id.encode() not in encoded
    assert settings.secret_access_key.encode() not in encoded
    assert SOURCE_MANIFEST_KEY == "source/source_manifest.json"
