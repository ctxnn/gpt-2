from __future__ import annotations

import csv
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import train_gpt2
from train_gpt2 import (
    CSVMetricLogger,
    CheckpointError,
    DataLoaderLite,
    GPT,
    GPTConfig,
    WandbLogger,
    atomic_torch_save,
    capture_rng_state,
    checkpoint_payload,
    cli_overrides,
    generate_text,
    get_learning_rate,
    load_checkpoint,
    load_config,
    prune_rolling_checkpoints,
    restore_rng_state,
)


def tiny_model(vocab_size: int = 32) -> GPT:
    return GPT(
        GPTConfig(
            block_size=8,
            vocab_size=vocab_size,
            n_layer=1,
            n_head=2,
            n_embd=8,
        )
    )


def make_shards(root: Path, size: int = 65) -> None:
    root.mkdir()
    np.save(root / "synthetic_train_000000.npy", np.arange(size, dtype=np.uint16))
    np.save(root / "synthetic_train_000001.npy", np.arange(size, dtype=np.uint16) + 100)
    np.save(root / "synthetic_val_000000.npy", np.arange(size, dtype=np.uint16))


@pytest.mark.parametrize(
    ("step", "expected"),
    [(0, 0.2), (1, 0.4), (2, 0.6)],
)
def test_linear_warmup(step: int, expected: float) -> None:
    actual = get_learning_rate(
        step,
        warmup_steps=3,
        max_steps=9,
        max_learning_rate=0.6,
        min_learning_rate=0.06,
    )
    assert actual == pytest.approx(expected)


def test_cosine_decay_and_boundaries() -> None:
    kwargs = dict(
        warmup_steps=2,
        max_steps=6,
        max_learning_rate=1.0,
        min_learning_rate=0.1,
    )
    assert get_learning_rate(2, **kwargs) == pytest.approx(1.0)
    assert get_learning_rate(4, **kwargs) == pytest.approx(0.55)
    assert get_learning_rate(6, **kwargs) == pytest.approx(0.1)
    assert get_learning_rate(100, **kwargs) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        get_learning_rate(-1, **kwargs)


def test_configuration_parsing_and_cli_overrides() -> None:
    config = load_config("configs/gpt2_124m_fineweb10b.yaml")
    assert config["model"]["n_layer"] == 12
    args = SimpleNamespace(
        data_root=Path("/tmp/data"),
        output_dir=Path("/tmp/output"),
        max_steps=7,
        seed=123,
        compile_model=True,
        wandb_mode="disabled",
        wandb_project="project",
        wandb_entity=None,
        wandb_run_name=None,
        checkpoint_interval=2,
        hellaswag_interval=5,
    )
    overrides = cli_overrides(args)
    assert overrides["training"] == {"max_steps": 7, "seed": 123, "compile": True}
    assert overrides["paths"]["data_root"] == "/tmp/data"
    assert overrides["checkpointing"]["interval"] == 2


def test_data_loader_rank_offsets_transitions_and_restore(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    make_shards(data_root)
    rank_zero = DataLoaderLite(2, 4, 0, 2, "train", data_root)
    rank_one = DataLoaderLite(2, 4, 1, 2, "train", data_root)
    x0, _ = rank_zero.next_batch()
    x1, _ = rank_one.next_batch()
    assert x0.flatten().tolist() == list(range(8))
    assert x1.flatten().tolist() == list(range(8, 16))
    state = rank_zero.state_dict()
    expected, _ = rank_zero.next_batch()
    restored = DataLoaderLite(2, 4, 0, 2, "train", data_root)
    restored.load_state_dict(state)
    actual, _ = restored.next_batch()
    assert torch.equal(actual, expected)
    rank_one.next_batch()
    for _ in range(3):
        rank_zero.next_batch()
        rank_one.next_batch()
    assert rank_zero.current_shard == rank_one.current_shard


def test_loader_rejects_final_partial_global_range(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    make_shards(data_root, size=18)
    loader = DataLoaderLite(2, 4, 0, 2, "train", data_root)
    loader.next_batch()
    loader.next_batch()
    assert loader.current_shard == 1


def test_rank_zero_loader_state_reconstructs_other_rank_offset(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    make_shards(data_root)
    rank_zero = DataLoaderLite(2, 4, 0, 2, "train", data_root)
    rank_zero.next_batch()
    rank_one = DataLoaderLite(2, 4, 1, 2, "train", data_root)
    rank_one.load_state_dict(rank_zero.state_dict())
    x, _ = rank_one.next_batch()
    assert x.flatten().tolist() == list(range(24, 32))


def test_loader_restore_rejects_incompatible_batch_shape(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    make_shards(data_root)
    original = DataLoaderLite(2, 4, 0, 1, "train", data_root)
    incompatible = DataLoaderLite(1, 4, 0, 1, "train", data_root)
    with pytest.raises(CheckpointError, match="configuration is incompatible"):
        incompatible.load_state_dict(original.state_dict())


def test_checkpoint_restores_model_optimizer_loader_and_step(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    make_shards(data_root)
    loader = DataLoaderLite(2, 4, 0, 1, "train", data_root)
    model = tiny_model()
    optimizer = model.configure_optimizers(0.1, 1e-3, "cpu")
    x, y = loader.next_batch()
    _, loss = model(x % 32, y % 32)
    assert loss is not None
    loss.backward()
    optimizer.step()
    payload = checkpoint_payload(
        model=model,
        optimizer=optimizer,
        completed_step=7,
        config={"model": {"vocab_size": 32}},
        train_loss=float(loss.detach()),
        validation_loss=2.0,
        train_loader=loader,
        tokens_processed=64,
        elapsed_wall_time=5.0,
        wandb_run_id="stable-id",
        git_sha="abc",
    )
    path = tmp_path / "checkpoint.pt"
    atomic_torch_save(payload, path)
    restored_model = tiny_model()
    restored_optimizer = restored_model.configure_optimizers(0.1, 1e-3, "cpu")
    restored_loader = DataLoaderLite(2, 4, 0, 1, "train", data_root)
    loaded = load_checkpoint(
        path,
        model=restored_model,
        optimizer=restored_optimizer,
        train_loader=restored_loader,
    )
    assert loaded["next_step"] == 8
    assert restored_loader.state_dict() == loader.state_dict()
    assert restored_optimizer.state_dict()["state"]
    for original, restored in zip(model.parameters(), restored_model.parameters()):
        assert torch.equal(original, restored)


def test_rng_restoration() -> None:
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    state = capture_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(2))
    restore_rng_state(state)
    actual = (random.random(), np.random.rand(), torch.rand(2))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_atomic_checkpoint_never_replaces_good_file_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "checkpoint.pt"
    atomic_torch_save({"value": 1}, path)

    def fail_save(*args, **kwargs):
        raise OSError("simulated failure")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="simulated"):
        atomic_torch_save({"value": 2}, path)
    assert torch.load(path, weights_only=False)["value"] == 1


def test_corrupt_checkpoint_has_clear_error(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    make_shards(data_root)
    path = tmp_path / "bad.pt"
    path.write_bytes(b"not a checkpoint")
    model = tiny_model()
    optimizer = model.configure_optimizers(0.1, 1e-3, "cpu")
    loader = DataLoaderLite(2, 4, 0, 1, "train", data_root)
    with pytest.raises(CheckpointError, match="cannot read checkpoint"):
        load_checkpoint(path, model=model, optimizer=optimizer, train_loader=loader)


def test_checkpoint_retention_preserves_milestones(tmp_path: Path) -> None:
    for step in range(1, 6):
        (tmp_path / f"checkpoint_step_{step:06d}.pt").touch()
    prune_rolling_checkpoints(tmp_path, keep_last=2, protected_steps={2})
    remaining = sorted(path.name for path in tmp_path.glob("*.pt"))
    assert remaining == [
        "checkpoint_step_000002.pt",
        "checkpoint_step_000004.pt",
        "checkpoint_step_000005.pt",
    ]


def test_csv_logging_appends(tmp_path: Path) -> None:
    path = tmp_path / "history.csv"
    logger = CSVMetricLogger(path)
    logger.log("train", train_step=1, train_loss=3.0)
    logger.log("validation", train_step=1, validation_loss=3.1)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["train_loss"] == "3.0"
    assert rows[1]["validation_loss"] == "3.1"


def test_wandb_disabled_does_not_import(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "wandb", None)
    logger = WandbLogger("disabled", master_process=True)
    logger.initialize(
        project="test", entity=None, name=None, run_id="id", config={}
    )
    logger.log({"loss": 1.0}, 1)
    assert logger.run is None


def test_wandb_offline_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}

    class FakeRun:
        def log(self, values, step):
            calls["log"] = (values, step)

        def finish(self, exit_code):
            calls["finish"] = exit_code

    def init(**kwargs):
        calls["init"] = kwargs
        return FakeRun()

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    logger = WandbLogger("offline", master_process=True)
    logger.initialize(
        project="test", entity=None, name=None, run_id="id", config={}
    )
    logger.log({"loss": 1.0}, 1)
    logger.finish(0)
    assert calls["init"]["mode"] == "offline"
    assert calls["init"]["resume"] == "allow"
    assert calls["log"][1] == 1


def test_wandb_online_initialization_failure_is_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_init(**kwargs):
        del kwargs
        raise RuntimeError("authentication failed")

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=fail_init))
    logger = WandbLogger("online", master_process=True)
    with pytest.raises(RuntimeError, match="refusing to train without monitoring"):
        logger.initialize(
            project="test", entity=None, name=None, run_id="id", config={}
        )


def test_tiny_cpu_forward_and_backward() -> None:
    model = tiny_model()
    x = torch.randint(0, 32, (2, 8))
    logits, loss = model(x, x)
    assert logits.shape == (2, 8, 32)
    assert loss is not None
    loss.backward()
    assert model.transformer.wte.weight.grad is not None
    assert model.lm_head.weight is model.transformer.wte.weight


def test_tiny_model_generation() -> None:
    model = tiny_model(vocab_size=50257)
    text = generate_text(
        model,
        prompt="Hello",
        device=torch.device("cpu"),
        max_length=3,
        seed=42,
    )
    assert isinstance(text, str)
    assert text


@pytest.mark.parametrize(
    "artifact",
    [
        "edu_fineweb10B/shard.npy",
        "checkpoints/model.pt",
        "outputs/run/model.safetensors",
        "wandb/run.bin",
        ".env",
        "logs/train.log",
    ],
)
def test_gitignore_protects_large_or_secret_artifacts(artifact: str) -> None:
    result = subprocess.run(
        ["git", "check-ignore", "-q", artifact],
        check=False,
        cwd=Path(__file__).parents[1],
    )
    assert result.returncode == 0, artifact
