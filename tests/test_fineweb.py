from pathlib import Path

import numpy as np
import pytest

from fineweb import prepare_dataset, validate_shard_filenames


def shard(name: str) -> Path:
    return Path(name)


def test_validate_shard_filenames_orders_numerically() -> None:
    shards = validate_shard_filenames(
        [shard("edufineweb_val_000000.npy")]
        + [shard(f"edufineweb_train_{index:06d}.npy") for index in range(99, 0, -1)]
    )

    assert [(item.split, item.index) for item in shards] == [("val", 0)] + [
        ("train", index) for index in range(1, 100)
    ]


def test_validate_shard_filenames_rejects_missing_training_shard() -> None:
    with pytest.raises(ValueError, match=r"missing training shard indices: \[2\]"):
        validate_shard_filenames(
            [
                shard("edufineweb_val_000000.npy"),
                shard("edufineweb_train_000001.npy"),
                shard("edufineweb_train_000003.npy"),
            ]
        )


def test_validate_shard_filenames_rejects_duplicate_training_shard() -> None:
    with pytest.raises(ValueError, match=r"duplicate training shard indices: \[1\]"):
        validate_shard_filenames(
            [
                shard("edufineweb_val_000000.npy"),
                shard("edufineweb_train_000001.npy"),
                shard("another/edufineweb_train_000001.npy"),
            ]
        )


@pytest.mark.parametrize(
    ("filename", "message"),
    [
        ("edufineweb_train_1.npy", "malformed shard filename"),
        ("edufineweb_test_000001.npy", "invalid shard split"),
        ("notes.txt", "unexpected file in shard directory"),
    ],
)
def test_validate_shard_filenames_rejects_bad_names(
    filename: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_shard_filenames(
            [
                shard("edufineweb_val_000000.npy"),
                shard("edufineweb_train_000001.npy"),
                shard(filename),
            ]
        )


def test_validate_shard_filenames_rejects_duplicate_validation_shard() -> None:
    with pytest.raises(ValueError, match="expected exactly one validation shard, found 2"):
        validate_shard_filenames(
            [
                shard("edufineweb_val_000000.npy"),
                shard("duplicate/edufineweb_val_000000.npy"),
                shard("edufineweb_train_000001.npy"),
            ]
        )


def test_prepare_dataset_calls_shard_hook_and_stops_after_full_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = [
        {"tokens": np.array([1, 2, 3], dtype=np.uint16)},
        {"tokens": np.array([4, 5, 6], dtype=np.uint16)},
    ]

    class Pool:
        def __init__(self, _: int) -> None:
            pass

        def __enter__(self) -> "Pool":
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def imap(self, function, values, *, chunksize: int):
            assert chunksize == 16
            return map(function, values)

    monkeypatch.setattr("fineweb.load_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr("fineweb.mp.Pool", Pool)
    monkeypatch.setattr("fineweb.tokenize", lambda item: item["tokens"])
    seen: list[tuple[str, str, int, int, np.ndarray]] = []

    def callback(
        path: Path,
        split: str,
        index: int,
        token_count: int,
    ) -> bool:
        seen.append(
            (
                path.name,
                split,
                index,
                token_count,
                np.load(path, allow_pickle=False),
            )
        )
        return True

    prepare_dataset(
        output_dir=tmp_path,
        shard_size=4,
        workers=32,
        shard_callback=callback,
    )

    assert len(seen) == 1
    assert seen[0][:4] == ("edufineweb_val_000000.npy", "val", 0, 4)
    np.testing.assert_array_equal(seen[0][4], np.array([1, 2, 3, 4]))
    assert not (tmp_path / "edufineweb_train_000001.npy").exists()
