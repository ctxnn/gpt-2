from pathlib import Path

import pytest

from fineweb import validate_shard_filenames


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
