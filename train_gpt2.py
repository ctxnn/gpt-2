"""Train GPT-2 from random initialization on pre-tokenized FineWeb-Edu shards.

The module is intentionally import-safe: importing it never initializes DDP,
downloads data, creates a model, or starts training.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import inspect
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, TextIO

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.distributed import destroy_process_group, init_process_group
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

CHECKPOINT_FORMAT_VERSION = 1
DATASET_IDENTIFIER = "HuggingFaceFW/fineweb-edu:sample-10BT"
CSV_FIELDS = [
    "event",
    "train_step",
    "train_loss",
    "learning_rate",
    "gradient_norm",
    "step_time_seconds",
    "tokens_per_second",
    "tokens_seen",
    "elapsed_hours",
    "estimated_training_cost",
    "validation_loss",
    "hellaswag_accuracy",
    "hellaswag_correct",
    "hellaswag_total",
    "sample_prompt",
    "generated_continuation",
]


class ConfigurationError(ValueError):
    """Raised when resolved configuration is unsafe or inconsistent."""


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is corrupt or incompatible."""


class CausalSelfAttention(nn.Module):
    def __init__(self, config: "GPTConfig") -> None:
        super().__init__()
        if config.n_embd % config.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence, channels = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        shape = (batch, sequence, self.n_head, channels // self.n_head)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(batch, sequence, channels)
        return self.c_proj(y)


# Backward-compatible alias for the original misspelling.
CasualSelfAttention = CausalSelfAttention


class MLP(nn.Module):
    def __init__(self, config: "GPTConfig") -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config: "GPTConfig") -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                "ln_f": nn.LayerNorm(config.n_embd),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, sequence = idx.shape
        if sequence > self.config.block_size:
            raise ValueError(
                f"sequence length {sequence} exceeds block size {self.config.block_size}"
            )
        positions = torch.arange(sequence, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(positions)
        for block in self.transformer.h:
            x = block(x)
        logits = self.lm_head(self.transformer.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss

    def configure_optimizers(
        self, weight_decay: float, learning_rate: float, device_type: str
    ) -> torch.optim.Optimizer:
        params = {name: value for name, value in self.named_parameters() if value.requires_grad}
        decay = [value for value in params.values() if value.dim() >= 2]
        no_decay = [value for value in params.values() if value.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        return torch.optim.AdamW(
            groups,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=use_fused,
        )


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def get_learning_rate(
    step: int,
    *,
    warmup_steps: int,
    max_steps: int,
    max_learning_rate: float,
    min_learning_rate: float,
) -> float:
    """Return LR for a zero-based optimizer step."""
    if step < 0:
        raise ValueError("step must be non-negative")
    if warmup_steps < 0 or max_steps <= warmup_steps:
        raise ValueError("require 0 <= warmup_steps < max_steps")
    if step < warmup_steps:
        return max_learning_rate * float(step + 1) / float(warmup_steps)
    if step >= max_steps:
        return min_learning_rate
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_learning_rate + coefficient * (
        max_learning_rate - min_learning_rate
    )


def get_lr(step: int) -> float:
    """Compatibility wrapper using the production schedule."""
    return get_learning_rate(
        step,
        warmup_steps=715,
        max_steps=19073,
        max_learning_rate=6e-4,
        min_learning_rate=6e-5,
    )


def load_tokens(filename: str | Path) -> torch.Tensor:
    array = np.load(filename).astype(np.int32)
    return torch.tensor(array, dtype=torch.long)


class DataLoaderLite:
    """Deterministic DDP-aware sequential loader over token shards."""

    def __init__(
        self,
        B: int,
        T: int,
        process_rank: int,
        num_processes: int,
        split: str,
        data_root: str | Path = "edu_fineweb10B",
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        if B <= 0 or T <= 0 or num_processes <= 0:
            raise ValueError("B, T, and num_processes must be positive")
        if not 0 <= process_rank < num_processes:
            raise ValueError("process_rank must be within world size")
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split
        self.data_root = Path(data_root)
        self.shards = sorted(self.data_root.glob(f"*{split}*.npy"))
        if not self.shards:
            raise FileNotFoundError(
                f"no {split} .npy shards found under {self.data_root.resolve()}"
            )
        self.current_shard = 0
        self.tokens = torch.empty(0, dtype=torch.long)
        self.current_position = 0
        self.reset()

    @property
    def rank_offset(self) -> int:
        return self.B * self.T * self.process_rank

    @property
    def global_stride(self) -> int:
        return self.B * self.T * self.num_processes

    def _load_current_shard(self) -> None:
        self.tokens = load_tokens(self.shards[self.current_shard])

    def _has_complete_batch(self) -> bool:
        # All ranks transition together. The final incomplete global range is
        # skipped instead of letting ranks drift onto different shards.
        rank_zero_position = self.current_position - self.rank_offset
        return rank_zero_position + self.global_stride + 1 <= len(self.tokens)

    def _advance_shard(self) -> None:
        for _ in range(len(self.shards)):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self._load_current_shard()
            self.current_position = self.rank_offset
            if self._has_complete_batch():
                return
        raise RuntimeError(
                        "no shard is large enough for one rank-local batch; "
            f"need at least {self.global_stride + 1} tokens"
        )

    def reset(self) -> None:
        self.current_shard = 0
        self._load_current_shard()
        self.current_position = self.rank_offset
        if not self._has_complete_batch():
            self._advance_shard()

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._has_complete_batch():
            self._advance_shard()
        end = self.current_position + self.B * self.T + 1
        buffer = self.tokens[self.current_position:end]
        x = buffer[:-1].view(self.B, self.T)
        y = buffer[1:].view(self.B, self.T)
        self.current_position += self.global_stride
        return x, y

    def state_dict(self) -> dict[str, int]:
        return {
            "current_shard": self.current_shard,
            "current_position": self.current_position,
            "rank_zero_position": self.current_position - self.rank_offset,
            "batch_size": self.B,
            "sequence_length": self.T,
            "num_processes": self.num_processes,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        try:
            shard = int(state["current_shard"])
            rank_zero_position = int(
                state.get("rank_zero_position", state["current_position"])
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointError("invalid data-loader state") from exc
        expected = {
            "batch_size": self.B,
            "sequence_length": self.T,
            "num_processes": self.num_processes,
        }
        incompatible = {
            key: (state.get(key), value)
            for key, value in expected.items()
            if key in state and int(state[key]) != value
        }
        if incompatible:
            raise CheckpointError(
                f"checkpoint data-loader configuration is incompatible: {incompatible}"
            )
        if not 0 <= shard < len(self.shards):
            raise CheckpointError(f"checkpoint shard index {shard} is out of range")
        if rank_zero_position < 0:
            raise CheckpointError("checkpoint data-loader position is negative")
        self.current_shard = shard
        self._load_current_shard()
        self.current_position = rank_zero_position + self.rank_offset
        if self.current_position > len(self.tokens):
            raise CheckpointError(
                "checkpoint data-loader position lies beyond the shard"
            )


def _deep_merge(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def validate_config(config: Mapping[str, Any]) -> None:
    required_sections = {
        "model",
        "training",
        "evaluation",
        "checkpointing",
        "logging",
    }
    missing = required_sections - set(config)
    if missing:
        raise ConfigurationError(f"missing configuration sections: {sorted(missing)}")
    model = config["model"]
    training = config["training"]
    if model["block_size"] < training["sequence_length"]:
        raise ConfigurationError("model.block_size must cover sequence_length")
    if not 0 <= training["warmup_steps"] < training["max_steps"]:
        raise ConfigurationError("require 0 <= warmup_steps < max_steps")
    if training["min_learning_rate"] > training["max_learning_rate"]:
        raise ConfigurationError("min_learning_rate cannot exceed max_learning_rate")
    if training["precision"] not in {"float32", "bfloat16"}:
        raise ConfigurationError("precision must be float32 or bfloat16")
    positive_fields = [
        ("training.max_steps", training["max_steps"]),
        ("training.total_batch_size_tokens", training["total_batch_size_tokens"]),
        ("training.micro_batch_size", training["micro_batch_size"]),
        ("training.sequence_length", training["sequence_length"]),
        ("evaluation.validation_interval", config["evaluation"]["validation_interval"]),
        ("evaluation.validation_batches", config["evaluation"]["validation_batches"]),
        ("evaluation.sample_interval", config["evaluation"]["sample_interval"]),
        (
            "evaluation.full_hellaswag_interval",
            config["evaluation"]["full_hellaswag_interval"],
        ),
        ("checkpointing.interval", config["checkpointing"]["interval"]),
        ("logging.scalar_interval", config["logging"]["scalar_interval"]),
    ]
    invalid = [name for name, value in positive_fields if int(value) <= 0]
    if invalid:
        raise ConfigurationError(f"configuration values must be positive: {invalid}")
    denominator = training["micro_batch_size"] * training["sequence_length"]
    if training["total_batch_size_tokens"] % denominator:
        raise ConfigurationError(
            "total_batch_size_tokens must be divisible by micro_batch_size * sequence_length"
        )
    if int(config["checkpointing"]["keep_last"]) < 0:
        raise ConfigurationError("checkpointing.keep_last cannot be negative")
    if config["logging"].get("histogram_mode", "disabled") not in {
        "disabled",
        "debug_gradients",
    }:
        raise ConfigurationError(
            "logging.histogram_mode must be disabled or debug_gradients"
        )


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigurationError("configuration root must be a mapping")
    if overrides:
        _deep_merge(config, overrides)
    validate_config(config)
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/gpt2_124m_fineweb10b.yaml", type=Path
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--compile", action="store_true", dest="compile_model")
    parser.add_argument(
        "--wandb-mode", choices=("disabled", "offline", "online")
    )
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--benchmark-steps", type=int)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--hellaswag-interval", type=int)
    parser.add_argument(
        "--max-runtime-seconds",
        type=int,
        help="gracefully checkpoint and pause after this much training-loop wall time",
    )
    return parser


def cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}

    def put(section: str, key: str, value: Any) -> None:
        if value is not None:
            overrides.setdefault(section, {})[key] = value

    put("paths", "data_root", str(args.data_root) if args.data_root else None)
    put("paths", "output_dir", str(args.output_dir) if args.output_dir else None)
    put("training", "max_steps", args.max_steps)
    put("training", "seed", args.seed)
    if args.compile_model:
        put("training", "compile", True)
    put("logging", "wandb_mode", args.wandb_mode)
    put("logging", "wandb_project", args.wandb_project)
    put("logging", "wandb_entity", args.wandb_entity)
    put("logging", "wandb_run_name", args.wandb_run_name)
    put("checkpointing", "interval", args.checkpoint_interval)
    put("evaluation", "full_hellaswag_interval", args.hellaswag_interval)
    return overrides


def parse_config(argv: list[str] | None = None) -> tuple[argparse.Namespace, dict[str, Any]]:
    args = build_parser().parse_args(argv)
    config_path = (
        Path("configs/smoke_test.yaml") if args.smoke_test else args.config
    )
    config = load_config(config_path, cli_overrides(args))
    return args, config


def atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["cpu"].cpu())
        if torch.cuda.is_available() and state.get("cuda"):
            torch.cuda.set_rng_state_all(
                [device_state.cpu() for device_state in state["cuda"]]
            )
    except (KeyError, TypeError, RuntimeError, ValueError) as exc:
        raise CheckpointError("checkpoint contains invalid RNG state") from exc


def git_commit_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return os.environ.get("GMN_SOURCE_GIT_SHA", "unknown")


def checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    completed_step: int,
    config: Mapping[str, Any],
    train_loss: float,
    validation_loss: float | None,
    train_loader: DataLoaderLite,
    tokens_processed: int,
    elapsed_wall_time: float,
    wandb_run_id: str | None,
    git_sha: str,
    per_rank_states: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "completed_step": completed_step,
        "config": dict(config),
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "data_loader": train_loader.state_dict(),
        "current_shard": train_loader.current_shard,
        "current_position": train_loader.current_position,
        "rng_state": capture_rng_state(),
        "tokens_processed": tokens_processed,
        "elapsed_wall_time": elapsed_wall_time,
        "wandb_run_id": wandb_run_id,
        "git_commit_sha": git_sha,
        "model_parameter_count": count_parameters(model),
        "per_rank_states": per_rank_states,
    }


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoaderLite,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
    rank: int = 0,
) -> dict[str, Any]:
    checkpoint_path = Path(path)
    try:
        try:
            payload = torch.load(
                checkpoint_path, map_location=map_location, weights_only=False
            )
        except TypeError:
            payload = torch.load(checkpoint_path, map_location=map_location)
    except Exception as exc:
        raise CheckpointError(f"cannot read checkpoint {checkpoint_path}: {exc}") from exc
    required = {
        "checkpoint_format_version",
        "model",
        "optimizer",
        "completed_step",
        "config",
        "data_loader",
        "rng_state",
    }
    if not isinstance(payload, dict):
        raise CheckpointError("checkpoint root must be a mapping")
    missing = required - set(payload)
    if missing:
        raise CheckpointError(f"checkpoint is missing required fields: {sorted(missing)}")
    if payload["checkpoint_format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointError(
            "unsupported checkpoint format version "
            f"{payload['checkpoint_format_version']}; expected {CHECKPOINT_FORMAT_VERSION}"
        )
    try:
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        rank_states = payload.get("per_rank_states")
        selected_state = (
            rank_states[rank]
            if rank_states and 0 <= rank < len(rank_states)
            else {
                "data_loader": payload["data_loader"],
                "rng_state": payload["rng_state"],
            }
        )
        train_loader.load_state_dict(selected_state["data_loader"])
    except Exception as exc:
        raise CheckpointError(f"checkpoint is incompatible: {exc}") from exc
    if restore_rng:
        restore_rng_state(selected_state["rng_state"])
    payload["next_step"] = int(payload["completed_step"]) + 1
    return payload


def prune_rolling_checkpoints(
    directory: str | Path, *, keep_last: int, protected_steps: set[int]
) -> None:
    paths = sorted(Path(directory).glob("checkpoint_step_*.pt"))
    rolling = []
    for path in paths:
        try:
            step = int(path.stem.rsplit("_", 1)[1])
        except ValueError:
            continue
        if step not in protected_steps:
            rolling.append(path)
    for old_path in rolling[:-keep_last] if keep_last else rolling:
        old_path.unlink()


def collect_rank_states(
    loader: DataLoaderLite, *, ddp: bool, world_size: int
) -> list[dict[str, Any]]:
    local_state = {
        "data_loader": loader.state_dict(),
        "rng_state": capture_rng_state(),
    }
    if not ddp:
        return [local_state]
    gathered: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(gathered, local_state)
    return [state for state in gathered if state is not None]


class CSVMetricLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **metrics: Any) -> None:
        exists = self.path.exists() and self.path.stat().st_size > 0
        row = {field: "" for field in CSV_FIELDS}
        row["event"] = event
        for key, value in metrics.items():
            if key in row:
                row[key] = value
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerow(row)
            handle.flush()


class Tee:
    def __init__(self, stream: TextIO, log: TextIO) -> None:
        self.stream = stream
        self.log = log

    def write(self, text: str) -> int:
        self.stream.write(text)
        written = self.log.write(text)
        self.flush()
        return written

    def flush(self) -> None:
        self.stream.flush()
        self.log.flush()

    def isatty(self) -> bool:
        return self.stream.isatty()


@contextlib.contextmanager
def tee_output(path: str | Path) -> Iterator[None]:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(old_stdout, log), Tee(old_stderr, log)
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


class WandbLogger:
    """Failure-tolerant W&B adapter; every metric is logged locally separately."""

    def __init__(self, mode: str, master_process: bool) -> None:
        self.mode = mode
        self.master_process = master_process
        self.run: Any = None
        self.run_id: str | None = None
        self._failed = False

    def initialize(
        self,
        *,
        project: str,
        entity: str | None,
        name: str | None,
        run_id: str,
        config: Mapping[str, Any],
        model: nn.Module | None = None,
        histogram_mode: str = "disabled",
    ) -> None:
        self.run_id = run_id
        if not self.master_process or self.mode == "disabled":
            return
        try:
            import wandb

            self.run = wandb.init(
                project=project,
                entity=entity,
                name=name,
                id=run_id,
                resume="allow",
                mode=self.mode,
                config=dict(config),
            )
            if histogram_mode == "debug_gradients" and model is not None:
                wandb.watch(model, log="gradients", log_freq=10)
        except Exception as exc:
            self._failed = True
            if self.mode == "online":
                raise RuntimeError(
                    "W&B online initialization failed; refusing to train without monitoring"
                ) from exc
            print(f"warning: W&B initialization failed; continuing locally: {exc}")

    def log(self, metrics: Mapping[str, Any], step: int) -> None:
        if self.run is None or self._failed:
            return
        try:
            self.run.log(dict(metrics), step=step)
        except Exception as exc:
            self._failed = True
            print(f"warning: W&B logging failed; continuing locally: {exc}")

    def finish(self, exit_code: int = 0) -> None:
        if self.run is not None:
            with contextlib.suppress(Exception):
                self.run.finish(exit_code=exit_code)


def autocast_context(device_type: str, precision: str) -> contextlib.AbstractContextManager:
    if device_type == "cuda" and precision == "bfloat16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def generate_text(
    model: nn.Module,
    *,
    prompt: str,
    device: torch.device,
    max_length: int,
    seed: int,
) -> str:
    import tiktoken

    tokenizer = tiktoken.get_encoding("gpt2")
    tokens = tokenizer.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    generator = torch.Generator(device=device).manual_seed(seed)
    was_training = model.training
    model.eval()
    try:
        while x.size(1) < max_length:
            logits, _ = model(x[:, -getattr(model, "config").block_size :])
            probabilities = F.softmax(logits[:, -1, :], dim=-1)
            topk_probabilities, topk_indices = torch.topk(
                probabilities, min(50, probabilities.size(-1)), dim=-1
            )
            chosen = torch.multinomial(topk_probabilities, 1, generator=generator)
            x = torch.cat((x, torch.gather(topk_indices, -1, chosen)), dim=1)
    finally:
        model.train(was_training)
    return tokenizer.decode(x[0].tolist())


def write_sample(path: str | Path, step: int, prompt: str, continuation: str) -> None:
    sample_path = Path(path)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    with sample_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## Step {step}\n\n**Prompt:** {prompt}\n\n{continuation}\n")


def write_run_status(path: str | Path, **values: Any) -> None:
    status_path = Path(path)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = status_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(values, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, status_path)


def prepare_smoke_data(data_root: str | Path) -> None:
    """Create deterministic, tiny local shards; never touches the network."""
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    train_path = root / "synthetic_train_000000.npy"
    val_path = root / "synthetic_val_000000.npy"
    if not train_path.exists():
        np.save(train_path, np.arange(1024, dtype=np.uint16) % 64)
    if not val_path.exists():
        np.save(val_path, np.arange(256, dtype=np.uint16) % 64)


def _setup_distributed() -> tuple[bool, int, int, int, torch.device]:
    ddp = int(os.environ.get("RANK", "-1")) >= 0
    if ddp:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP training requires CUDA")
        init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        return True, rank, local_rank, world_size, device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return False, 0, 0, 1, device


def _validation_loss(
    model: nn.Module,
    loader: DataLoaderLite,
    *,
    batches: int,
    device: torch.device,
    device_type: str,
    precision: str,
    ddp: bool,
) -> float:
    model.eval()
    loader.reset()
    accumulated = torch.zeros((), device=device)
    with torch.no_grad():
        for _ in range(batches):
            x, y = loader.next_batch()
            with autocast_context(device_type, precision):
                _, loss = model(x.to(device), y.to(device))
            assert loss is not None
            accumulated += loss.detach() / batches
    if ddp:
        dist.all_reduce(accumulated, op=dist.ReduceOp.AVG)
    return float(accumulated.item())


def _hellaswag(
    model: nn.Module,
    *,
    data_root: Path,
    device: torch.device,
    device_type: str,
    precision: str,
    rank: int,
    world_size: int,
    ddp: bool,
) -> tuple[float, int, int]:
    from hellaswag import iterate_examples, render_example

    model.eval()
    correct = 0
    total = 0
    for index, example in enumerate(
        iterate_examples("val", data_root=data_root, allow_download=False)
    ):
        if index % world_size != rank:
            continue
        _, tokens, mask, label = render_example(example)
        tokens, mask = tokens.to(device), mask.to(device)
        with torch.no_grad(), autocast_context(device_type, precision):
            logits, _ = model(tokens)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_tokens = tokens[..., 1:].contiguous()
        losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_tokens.view(-1),
            reduction="none",
        ).view(tokens.size(0), -1)
        shifted_mask = mask[..., 1:].contiguous()
        average = (losses * shifted_mask).sum(1) / shifted_mask.sum(1)
        correct += int(average.argmin().item() == label)
        total += 1
    if ddp:
        values = torch.tensor([correct, total], dtype=torch.long, device=device)
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        correct, total = map(int, values.tolist())
    return correct / total, correct, total


def train(args: argparse.Namespace, config: dict[str, Any]) -> None:
    paths = config.get("paths", {})
    data_root = Path(paths.get("data_root", "edu_fineweb10B"))
    output_dir = Path(paths.get("output_dir", "outputs/gpt2_124m_fineweb10b"))
    training = config["training"]
    evaluation = config["evaluation"]
    checkpointing = config["checkpointing"]
    logging_config = config["logging"]
    if (
        logging_config.get("wandb_mode") == "online"
        and not os.environ.get("WANDB_API_KEY")
    ):
        raise ConfigurationError(
            "WANDB_API_KEY is required when logging.wandb_mode is online"
        )
    if args.max_runtime_seconds is not None and args.max_runtime_seconds <= 0:
        raise ConfigurationError("--max-runtime-seconds must be positive")
    for split in ("train", "val"):
        if not any(data_root.glob(f"*{split}*.npy")):
            raise FileNotFoundError(
                f"no {split} .npy shards found under {data_root.resolve()}"
            )
    hellaswag_root = Path(paths.get("hellaswag_root", "hellaswag"))
    if (
        args.benchmark_steps is None
        and (
            evaluation["full_hellaswag_at_end"]
            or training["max_steps"] >= evaluation["full_hellaswag_interval"]
        )
        and not (hellaswag_root / "hellaswag_val.jsonl").is_file()
    ):
        raise FileNotFoundError(
            f"HellaSwag validation data is missing under {hellaswag_root.resolve()}; "
            "download it before allocating paid training compute"
        )
    ddp, rank, local_rank, world_size, device = _setup_distributed()
    master = rank == 0
    device_type = "cuda" if device.type == "cuda" else "cpu"
    logs_dir = output_dir / "logs"
    results_dir = output_dir / "results"
    checkpoints_dir = output_dir / "checkpoints"
    log_path = logs_dir / "train.log"
    history = CSVMetricLogger(results_dir / "training_history.csv")
    status_path = results_dir / "run_status.json"
    samples_path = results_dir / "generated_samples.md"
    seed = int(training["seed"]) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_float32_matmul_precision("high")

    denominator = (
        training["micro_batch_size"] * training["sequence_length"] * world_size
    )
    if training["total_batch_size_tokens"] % denominator:
        raise ConfigurationError(
            "total_batch_size_tokens must be divisible by micro batch tokens * DDP world size"
        )
    accumulation_steps = training["total_batch_size_tokens"] // denominator
    train_loader = DataLoaderLite(
        B=training["micro_batch_size"],
        T=training["sequence_length"],
        process_rank=rank,
        num_processes=world_size,
        split="train",
        data_root=data_root,
    )
    val_loader = DataLoaderLite(
        B=training["micro_batch_size"],
        T=training["sequence_length"],
        process_rank=rank,
        num_processes=world_size,
        split="val",
        data_root=data_root,
    )
    raw_model = GPT(GPTConfig(**config["model"])).to(device)
    optimizer = raw_model.configure_optimizers(
        training["weight_decay"], training["max_learning_rate"], device_type
    )
    start_step = 1
    tokens_seen = 0
    elapsed_before = 0.0
    latest_validation: float | None = None
    run_id = uuid.uuid4().hex
    original_git_sha = git_commit_sha()
    if args.resume:
        resumed = load_checkpoint(
            args.resume,
            model=raw_model,
            optimizer=optimizer,
            train_loader=train_loader,
            map_location=device,
            rank=rank,
        )
        start_step = resumed["next_step"]
        tokens_seen = int(resumed.get("tokens_processed", 0))
        elapsed_before = float(resumed.get("elapsed_wall_time", 0.0))
        latest_validation = resumed.get("validation_loss")
        run_id = resumed.get("wandb_run_id") or run_id
        original_git_sha = resumed.get("git_commit_sha", original_git_sha)
    if training.get("compile", False):
        raw_model_for_checkpoint = raw_model
        training_model: nn.Module = torch.compile(raw_model)
    else:
        raw_model_for_checkpoint = raw_model
        training_model = raw_model
    if ddp:
        training_model = DDP(training_model, device_ids=[local_rank])

    metadata = {
        "resolved_config": config,
        "dataset_identifier": DATASET_IDENTIFIER,
        "original_git_sha": original_git_sha,
        "training_git_sha": git_commit_sha(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_model": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "ddp_world_size": world_size,
        "precision": training["precision"],
        "givemeanode_job_id": os.environ.get("GIVEMEANODE_JOB_ID"),
        "planned_spending_limit": os.environ.get("PLANNED_SPENDING_LIMIT"),
    }
    wandb_logger = WandbLogger(logging_config.get("wandb_mode", "disabled"), master)
    started = time.monotonic()
    runtime_deadline = (
        started + args.max_runtime_seconds
        if args.max_runtime_seconds is not None
        else None
    )
    exit_code = 1
    state = "running"
    paused = False
    final_step = start_step - 1
    benchmark_steps = args.benchmark_steps
    stop_step = training["max_steps"]
    if benchmark_steps is not None:
        if benchmark_steps <= 0:
            raise ConfigurationError("--benchmark-steps must be positive")
        stop_step = min(stop_step, start_step + benchmark_steps - 1)
    if start_step > stop_step:
        raise ConfigurationError(
            f"resume would start at step {start_step}, beyond configured stop step {stop_step}"
        )
    last_validation_step: int | None = None
    last_hellaswag_step: int | None = None
    if master:
        write_run_status(
            status_path,
            status=state,
            started_at=dt.datetime.now(dt.timezone.utc).isoformat(),
            start_step=start_step,
            planned_final_step=stop_step,
            wandb_run_id=run_id,
        )
    try:
        with tee_output(log_path) if master else contextlib.nullcontext():
            wandb_logger.initialize(
                project=logging_config["wandb_project"],
                entity=logging_config.get("wandb_entity"),
                name=logging_config.get("wandb_run_name"),
                run_id=run_id,
                config=metadata,
                model=raw_model_for_checkpoint,
                histogram_mode=logging_config.get("histogram_mode", "disabled"),
            )
            print(
                f"device={device} world_size={world_size} accumulation_steps={accumulation_steps} "
                f"parameters={count_parameters(raw_model_for_checkpoint):,}"
            )
            for step in range(start_step, stop_step + 1):
                step_started = time.monotonic()
                final_step = step
                training_model.train()
                optimizer.zero_grad(set_to_none=True)
                accumulated_loss = torch.zeros((), device=device)
                for micro_step in range(accumulation_steps):
                    x, y = train_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    if ddp:
                        training_model.require_backward_grad_sync = (
                            micro_step == accumulation_steps - 1
                        )
                    with autocast_context(device_type, training["precision"]):
                        _, loss = training_model(x, y)
                    assert loss is not None
                    scaled_loss = loss / accumulation_steps
                    accumulated_loss += scaled_loss.detach()
                    scaled_loss.backward()
                if ddp:
                    dist.all_reduce(accumulated_loss, op=dist.ReduceOp.AVG)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    training_model.parameters(), training["gradient_clip"]
                )
                learning_rate = get_learning_rate(
                    step - 1,
                    warmup_steps=training["warmup_steps"],
                    max_steps=training["max_steps"],
                    max_learning_rate=training["max_learning_rate"],
                    min_learning_rate=training["min_learning_rate"],
                )
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                optimizer.step()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                step_seconds = time.monotonic() - step_started
                tokens_seen += training["total_batch_size_tokens"]
                elapsed = elapsed_before + time.monotonic() - started
                hourly_rate = os.environ.get("GPU_HOURLY_RATE")
                estimated_cost = (
                    elapsed / 3600 * float(hourly_rate) if hourly_rate else None
                )
                metrics = {
                    "train_step": step,
                    "train_loss": float(accumulated_loss.item()),
                    "learning_rate": learning_rate,
                    "gradient_norm": float(gradient_norm),
                    "step_time_seconds": step_seconds,
                    "tokens_per_second": training["total_batch_size_tokens"] / step_seconds,
                    "tokens_seen": tokens_seen,
                    "elapsed_hours": elapsed / 3600,
                    "estimated_training_cost": estimated_cost,
                }
                if master and step % logging_config["scalar_interval"] == 0:
                    print(
                        f"step {step:5d} loss={metrics['train_loss']:.6f} "
                        f"lr={learning_rate:.4e} norm={float(gradient_norm):.4f} "
                        f"tok/s={metrics['tokens_per_second']:.0f}"
                    )
                    history.log("train", **metrics)
                    wandb_logger.log(metrics, step)

                validation_due = step % evaluation["validation_interval"] == 0
                if validation_due:
                    latest_validation = _validation_loss(
                        training_model,
                        val_loader,
                        batches=evaluation["validation_batches"],
                        device=device,
                        device_type=device_type,
                        precision=training["precision"],
                        ddp=ddp,
                    )
                    if master:
                        history.log(
                            "validation",
                            train_step=step,
                            validation_loss=latest_validation,
                        )
                        wandb_logger.log(
                            {"validation_loss": latest_validation}, step
                        )
                    last_validation_step = step

                if (
                    benchmark_steps is None
                    and step % evaluation["full_hellaswag_interval"] == 0
                ):
                    accuracy, correct, total = _hellaswag(
                        training_model,
                        data_root=Path(paths.get("hellaswag_root", "hellaswag")),
                        device=device,
                        device_type=device_type,
                        precision=training["precision"],
                        rank=rank,
                        world_size=world_size,
                        ddp=ddp,
                    )
                    if master:
                        hella_metrics = {
                            "hellaswag_accuracy": accuracy,
                            "hellaswag_correct": correct,
                            "hellaswag_total": total,
                        }
                        history.log("hellaswag", train_step=step, **hella_metrics)
                        wandb_logger.log(hella_metrics, step)
                    last_hellaswag_step = step

                if (
                    benchmark_steps is None
                    and step % evaluation["sample_interval"] == 0
                    and master
                ):
                    prompt = logging_config.get(
                        "sample_prompt", "Hello, I'm a language model,"
                    )
                    continuation = generate_text(
                        raw_model_for_checkpoint,
                        prompt=prompt,
                        device=device,
                        max_length=logging_config.get("sample_max_length", 64),
                        seed=training["seed"] + step,
                    )
                    write_sample(samples_path, step, prompt, continuation)
                    history.log(
                        "sample",
                        train_step=step,
                        sample_prompt=prompt,
                        generated_continuation=continuation,
                    )
                    wandb_logger.log(
                        {"sample_step": step, "prompt": prompt, "generated_continuation": continuation},
                        step,
                    )

                checkpoint_due = step % checkpointing["interval"] == 0
                if checkpoint_due:
                    rank_states = collect_rank_states(
                        train_loader, ddp=ddp, world_size=world_size
                    )
                    if master:
                        payload = checkpoint_payload(
                            model=raw_model_for_checkpoint,
                            optimizer=optimizer,
                            completed_step=step,
                            config=config,
                            train_loss=float(accumulated_loss.item()),
                            validation_loss=latest_validation,
                            train_loader=train_loader,
                            tokens_processed=tokens_seen,
                            elapsed_wall_time=elapsed,
                            wandb_run_id=run_id,
                            git_sha=original_git_sha,
                            per_rank_states=rank_states,
                        )
                        atomic_torch_save(
                            payload, checkpoints_dir / f"checkpoint_step_{step:06d}.pt"
                        )
                        prune_rolling_checkpoints(
                            checkpoints_dir,
                            keep_last=checkpointing["keep_last"],
                            protected_steps=set(checkpointing["milestone_steps"]),
                        )
                if runtime_deadline is not None and time.monotonic() >= runtime_deadline:
                    paused = True
                    if master:
                        print(
                            f"training runtime budget reached after step {step}; "
                            "writing a resumable checkpoint"
                        )
                    break

            # Benchmarks intentionally skip expensive final evaluation.
            if benchmark_steps is None and not paused:
                if last_validation_step != final_step:
                    latest_validation = _validation_loss(
                        training_model,
                        val_loader,
                        batches=evaluation["validation_batches"],
                        device=device,
                        device_type=device_type,
                        precision=training["precision"],
                        ddp=ddp,
                    )
                    if master:
                        history.log(
                            "validation",
                            train_step=final_step,
                            validation_loss=latest_validation,
                        )
                        wandb_logger.log(
                            {"validation_loss": latest_validation}, final_step
                        )
                if (
                    evaluation["full_hellaswag_at_end"]
                    and last_hellaswag_step != final_step
                ):
                    accuracy, correct, total = _hellaswag(
                        training_model,
                        data_root=Path(paths.get("hellaswag_root", "hellaswag")),
                        device=device,
                        device_type=device_type,
                        precision=training["precision"],
                        rank=rank,
                        world_size=world_size,
                        ddp=ddp,
                    )
                    if master:
                        final_hella_metrics = {
                            "hellaswag_accuracy": accuracy,
                            "hellaswag_correct": correct,
                            "hellaswag_total": total,
                        }
                        history.log(
                            "hellaswag",
                            train_step=final_step,
                            **final_hella_metrics,
                        )
                        wandb_logger.log(final_hella_metrics, final_step)
            final_rank_states = collect_rank_states(
                train_loader, ddp=ddp, world_size=world_size
            )
            if master:
                elapsed = elapsed_before + time.monotonic() - started
                payload = checkpoint_payload(
                    model=raw_model_for_checkpoint,
                    optimizer=optimizer,
                    completed_step=final_step,
                    config=config,
                    train_loss=float(accumulated_loss.item()),
                    validation_loss=latest_validation,
                    train_loader=train_loader,
                    tokens_processed=tokens_seen,
                    elapsed_wall_time=elapsed,
                    wandb_run_id=run_id,
                    git_sha=original_git_sha,
                    per_rank_states=final_rank_states,
                )
                final_path = checkpoints_dir / f"final_step_{final_step:06d}.pt"
                if final_path.exists():
                    raise FileExistsError(f"refusing to overwrite final checkpoint {final_path}")
                atomic_torch_save(payload, final_path)
            state = "paused" if paused else "completed"
            exit_code = 0
    except BaseException:
        state = "error"
        raise
    finally:
        if master:
            write_run_status(
                status_path,
                status=state,
                final_step=final_step,
                tokens_seen=tokens_seen,
                elapsed_wall_time=elapsed_before + time.monotonic() - started,
                finished_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                wandb_run_id=run_id,
            )
        wandb_logger.finish(exit_code)
        if ddp:
            destroy_process_group()


def main(argv: list[str] | None = None) -> int:
    args, config = parse_config(argv)
    output_dir = Path(
        config.get("paths", {}).get("output_dir", "outputs/gpt2_124m_fineweb10b")
    )
    try:
        if args.smoke_test:
            prepare_smoke_data(config["paths"]["data_root"])
        train(args, config)
        return 0
    except Exception as exc:
        log_path = output_dir / "logs" / "train.log"
        with tee_output(log_path):
            traceback.print_exc()
        write_run_status(
            output_dir / "results" / "run_status.json",
            status="error",
            error_type=type(exc).__name__,
            error_message=str(exc),
            finished_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
