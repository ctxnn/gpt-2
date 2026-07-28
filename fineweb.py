"""Download and tokenize FineWeb-Edu into NumPy shards.

This is a separate, explicit preparation command. Importing this module never
downloads data, and the training script never invokes it.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

DATASET_ID = "HuggingFaceFW/fineweb-edu"
DEFAULT_DATASET_CONFIG = "sample-10BT"
DEFAULT_SHARD_SIZE = 100_000_000

_tokenizer = None


def tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding("gpt2")
    return _tokenizer


def tokenize(document: dict[str, Any]) -> np.ndarray:
    encoding = tokenizer()
    eot = encoding._special_tokens["<|endoftext|>"]
    tokens = np.array([eot, *encoding.encode_ordinary(document["text"])])
    if not ((0 <= tokens).all() and (tokens < 2**16).all()):
        raise ValueError("token ID does not fit uint16")
    return tokens.astype(np.uint16)


def write_datafile(filename: str | Path, tokens: np.ndarray) -> None:
    np.save(filename, tokens)


def prepare_dataset(
    *,
    output_dir: str | Path,
    dataset_config: str = DEFAULT_DATASET_CONFIG,
    shard_size: int = DEFAULT_SHARD_SIZE,
    workers: int | None = None,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(DATASET_ID, name=dataset_config, split="train")
    worker_count = workers or max(1, os.cpu_count() or 1)
    with mp.Pool(worker_count) as pool:
        shard_index = 0
        buffer = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress = None
        for tokens in pool.imap(tokenize, dataset, chunksize=16):
            consumed = 0
            while consumed < len(tokens):
                if progress is None:
                    progress = tqdm(
                        total=shard_size,
                        unit="tokens",
                        desc=f"Shard {shard_index}",
                    )
                amount = min(shard_size - token_count, len(tokens) - consumed)
                buffer[token_count : token_count + amount] = tokens[
                    consumed : consumed + amount
                ]
                token_count += amount
                consumed += amount
                progress.update(amount)
                if token_count == shard_size:
                    split = "val" if shard_index == 0 else "train"
                    write_datafile(
                        output / f"edufineweb_{split}_{shard_index:06d}.npy",
                        buffer,
                    )
                    progress.close()
                    progress = None
                    shard_index += 1
                    token_count = 0
        if token_count:
            split = "val" if shard_index == 0 else "train"
            write_datafile(
                output / f"edufineweb_{split}_{shard_index:06d}.npy",
                buffer[:token_count],
            )
            if progress is not None:
                progress.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("edu_fineweb10B"))
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    prepare_dataset(
        output_dir=args.output_dir,
        dataset_config=args.dataset_config,
        shard_size=args.shard_size,
        workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
