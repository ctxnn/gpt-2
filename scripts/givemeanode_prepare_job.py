#!/usr/bin/env python3
"""Bootstrap the original FineWeb preparation path inside a batch attempt."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPOSITORY = "https://github.com/ctxnn/gpt-2.git"
WORKSPACE = Path("/workspace")
CHECKOUT = WORKSPACE / "gpt-2"
DEPENDENCIES = (
    "numpy==2.4.6",
    "tiktoken==0.13.0",
    "datasets==3.6.0",
    "tqdm==4.70.0",
    "requests==2.34.2",
    "pyarrow==25.0.0",
    "huggingface-hub==1.25.1",
    "boto3==1.43.57",
)


def run(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    source_sha = os.environ.get("GMN_SOURCE_GIT_SHA", "").strip()
    if len(source_sha) != 40 or any(
        character not in "0123456789abcdefABCDEF" for character in source_sha
    ):
        raise ValueError("GMN_SOURCE_GIT_SHA must be a hexadecimal commit SHA")
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(WORKSPACE)
    print(
        "DISK_FREE_BYTES",
        usage.free,
        "DISK_FREE_GIB",
        round(usage.free / (1024**3), 3),
        flush=True,
    )
    run(["apt-get", "update"])
    run(
        [
            "apt-get",
            "install",
            "-y",
            "--no-install-recommends",
            "git",
            "ca-certificates",
        ]
    )
    run([sys.executable, "-m", "pip", "install", "--no-cache-dir", *DEPENDENCIES])
    run(["git", "clone", REPOSITORY, str(CHECKOUT)])
    run(["git", "checkout", "--detach", source_sha], cwd=CHECKOUT)
    checked_out = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=CHECKOUT,
        text=True,
    ).strip()
    if checked_out != source_sha.lower():
        raise RuntimeError("checked-out source SHA does not match the request")
    print(f"SOURCE_SHA_VERIFIED {checked_out}", flush=True)
    run(
        [
            sys.executable,
            "-u",
            "-m",
            "scripts.prepare_fineweb_original_to_s3",
        ],
        cwd=CHECKOUT,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
