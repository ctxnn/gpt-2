from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from transformers import GPT2LMHeadModel

from scripts import export_hf_model as export
from train_gpt2 import GPT, GPTConfig


def tiny_models() -> tuple[GPTConfig, GPT, GPT2LMHeadModel]:
    config = GPTConfig(
        block_size=32,
        vocab_size=50_304,
        n_layer=1,
        n_head=1,
        n_embd=16,
    )
    torch.manual_seed(7)
    native = GPT(config)
    huggingface = GPT2LMHeadModel(export.hf_config_from_native(config))
    return config, native, huggingface


def test_parameter_mapping_transposes_conv1d_and_trims_tied_embeddings() -> None:
    config, native, huggingface = tiny_models()
    report = export.map_native_state(native.state_dict(), huggingface, config)

    assert report["missing_parameters"] == []
    assert report["unexpected_parameters"] == []
    assert report["trimmed_embedding_rows"] == 47
    assert torch.equal(
        huggingface.transformer.h[0].attn.c_attn.weight,
        native.transformer.h[0].attn.c_attn.weight.t(),
    )
    assert torch.equal(
        huggingface.transformer.h[0].mlp.c_proj.weight,
        native.transformer.h[0].mlp.c_proj.weight.t(),
    )
    assert huggingface.transformer.wte.weight.shape[0] == 50_257
    assert huggingface.transformer.wte.weight.data_ptr() == huggingface.lm_head.weight.data_ptr()


def test_native_and_huggingface_logits_and_loss_match() -> None:
    config, native, huggingface = tiny_models()
    export.map_native_state(native.state_dict(), huggingface, config)

    result = export.validate_equivalence(native, huggingface, seed=123)

    assert result["status"] == "passed"
    assert result["max_absolute_logit_difference"] <= result["absolute_tolerance"]
    assert result["absolute_loss_difference"] <= 1e-5


def test_all_exported_parameters_are_finite() -> None:
    config, native, huggingface = tiny_models()
    export.map_native_state(native.state_dict(), huggingface, config)
    assert export.validate_finite(huggingface)["nonfinite_tensors"] == []

    with torch.no_grad():
        huggingface.transformer.wte.weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        export.validate_finite(huggingface)


def test_source_records_require_complete_step_size_hash_and_metrics(tmp_path: Path) -> None:
    checkpoint = tmp_path / "final.pt"
    checkpoint.write_bytes(b"verified-checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    complete = tmp_path / "COMPLETE.json"
    complete.write_text(json.dumps({"status": "completed", "final_step": 19_073}))
    checkpoint_record = tmp_path / "final_checkpoint.json"
    checkpoint_record.write_text(
        json.dumps(
            {
                "completed_step": 19_073,
                "size_bytes": checkpoint.stat().st_size,
                "sha256": digest,
            }
        )
    )
    metrics = tmp_path / "final_metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "final_step": 19_073,
                "checkpoint_sha256": digest,
                "validation_loss": 3.0,
                "hellaswag_results": {"step": 19_073, "accuracy": 0.3},
            }
        )
    )

    values = export.validate_source_records(
        checkpoint_path=checkpoint,
        complete_path=complete,
        checkpoint_record_path=checkpoint_record,
        metrics_path=metrics,
    )
    assert values[0]["status"] == "completed"

    checkpoint.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="size mismatch"):
        export.validate_source_records(
            checkpoint_path=checkpoint,
            complete_path=complete,
            checkpoint_record_path=checkpoint_record,
            metrics_path=metrics,
        )


def test_upload_tree_is_exact_and_rejects_private_files(tmp_path: Path) -> None:
    for filename in export.UPLOAD_FILES:
        (tmp_path / filename).write_text("safe")
    assert set(export.validate_upload_tree(tmp_path)) == export.UPLOAD_FILES

    (tmp_path / ".env").write_text("not-a-real-secret")
    with pytest.raises(ValueError, match="unexpected"):
        export.validate_upload_tree(tmp_path)


def test_publication_requires_protected_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="protected environment"):
        export.require_hf_token()
