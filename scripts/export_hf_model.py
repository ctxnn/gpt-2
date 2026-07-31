"""Convert the verified native GPT-2 checkpoint to Hugging Face format.

The converter is deliberately strict. It validates the source completion
records, maps every native parameter, compares native and Hugging Face logits,
checks clean-process loading/generation, and only permits a small upload
whitelist. ``HF_TOKEN`` is read only when ``--push-to-hub`` is requested.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import tiktoken
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    GPT2Config,
    GPT2LMHeadModel,
)

from train_gpt2 import GPT, GPTConfig


EXPORT_VOCAB_SIZE = 50_257
EXPECTED_FINAL_STEP = 19_073
EXPECTED_ARCHITECTURE = {
    "n_layer": 12,
    "n_head": 12,
    "n_embd": 768,
    "block_size": 1024,
    "vocab_size": 50_304,
}
UPLOAD_FILES = {
    "README.md",
    "config.json",
    "export_metadata.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def validate_source_records(
    *,
    checkpoint_path: Path,
    complete_path: Path,
    checkpoint_record_path: Path,
    metrics_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    complete = load_json(complete_path)
    checkpoint_record = load_json(checkpoint_record_path)
    metrics = load_json(metrics_path)

    if complete.get("status") != "completed":
        raise ValueError("training COMPLETE does not report completed status")
    for source, value in (
        ("COMPLETE", complete.get("final_step")),
        ("checkpoint record", checkpoint_record.get("completed_step")),
        ("final metrics", metrics.get("final_step")),
    ):
        if int(value) != EXPECTED_FINAL_STEP:
            raise ValueError(f"{source} step is {value}, expected {EXPECTED_FINAL_STEP}")

    expected_size = int(checkpoint_record["size_bytes"])
    actual_size = checkpoint_path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"checkpoint size mismatch: {actual_size} != {expected_size}")
    actual_sha256 = sha256_file(checkpoint_path)
    expected_sha256 = str(checkpoint_record["sha256"])
    if actual_sha256 != expected_sha256:
        raise ValueError("checkpoint SHA-256 does not match final_checkpoint.json")
    if actual_sha256 != metrics.get("checkpoint_sha256"):
        raise ValueError("checkpoint SHA-256 does not match final_metrics.json")

    if not math.isfinite(float(metrics["validation_loss"])):
        raise ValueError("final validation loss is missing or non-finite")
    hellaswag = metrics.get("hellaswag_results")
    if not isinstance(hellaswag, dict) or int(hellaswag.get("step", -1)) != EXPECTED_FINAL_STEP:
        raise ValueError("final HellaSwag metrics are missing or have the wrong step")
    if not math.isfinite(float(hellaswag["accuracy"])):
        raise ValueError("final HellaSwag accuracy is non-finite")
    return complete, checkpoint_record, metrics


def native_config_from_payload(payload: Mapping[str, Any]) -> GPTConfig:
    model_config = payload.get("config", {}).get("model", {})
    values = {
        key: int(model_config.get(key, expected))
        for key, expected in EXPECTED_ARCHITECTURE.items()
    }
    if values != EXPECTED_ARCHITECTURE:
        raise ValueError(f"unexpected native architecture: {values}")
    return GPTConfig(**values)


def hf_config_from_native(config: GPTConfig) -> GPT2Config:
    hf_config = GPT2Config(
        vocab_size=EXPORT_VOCAB_SIZE,
        n_positions=config.block_size,
        n_ctx=config.block_size,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
        activation_function="gelu_new",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        bos_token_id=50_256,
        eos_token_id=50_256,
        tie_word_embeddings=True,
        use_cache=True,
    )
    hf_config.architectures = ["GPT2LMHeadModel"]
    hf_config._attn_implementation = "sdpa"
    return hf_config


def expected_native_keys(config: GPTConfig) -> set[str]:
    keys = {"transformer.wte.weight", "transformer.wpe.weight", "transformer.ln_f.weight", "transformer.ln_f.bias", "lm_head.weight"}
    suffixes = (
        "ln_1.weight",
        "ln_1.bias",
        "attn.c_attn.weight",
        "attn.c_attn.bias",
        "attn.c_proj.weight",
        "attn.c_proj.bias",
        "ln_2.weight",
        "ln_2.bias",
        "mlp.c_fc.weight",
        "mlp.c_fc.bias",
        "mlp.c_proj.weight",
        "mlp.c_proj.bias",
    )
    for layer in range(config.n_layer):
        keys.update(f"transformer.h.{layer}.{suffix}" for suffix in suffixes)
    return keys


def map_native_state(
    native_state: Mapping[str, torch.Tensor], hf_model: GPT2LMHeadModel, native_config: GPTConfig
) -> dict[str, Any]:
    source_keys = set(native_state)
    expected_source = expected_native_keys(native_config)
    if source_keys != expected_source:
        missing = sorted(expected_source - source_keys)
        unexpected = sorted(source_keys - expected_source)
        raise ValueError(f"native parameter mismatch; missing={missing}, unexpected={unexpected}")
    if not torch.equal(native_state["lm_head.weight"], native_state["transformer.wte.weight"]):
        raise ValueError("native input and output embeddings are not tied")

    mapped: dict[str, torch.Tensor] = {
        "transformer.wte.weight": native_state["transformer.wte.weight"][:EXPORT_VOCAB_SIZE].contiguous(),
        "transformer.wpe.weight": native_state["transformer.wpe.weight"],
        "transformer.ln_f.weight": native_state["transformer.ln_f.weight"],
        "transformer.ln_f.bias": native_state["transformer.ln_f.bias"],
        "lm_head.weight": native_state["transformer.wte.weight"][:EXPORT_VOCAB_SIZE].contiguous(),
    }
    transposed: list[str] = []
    for layer in range(native_config.n_layer):
        prefix = f"transformer.h.{layer}"
        for suffix in ("ln_1.weight", "ln_1.bias", "ln_2.weight", "ln_2.bias"):
            mapped[f"{prefix}.{suffix}"] = native_state[f"{prefix}.{suffix}"]
        for module in ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"):
            weight = f"{prefix}.{module}.weight"
            bias = f"{prefix}.{module}.bias"
            mapped[weight] = native_state[weight].t().contiguous()
            mapped[bias] = native_state[bias]
            transposed.append(weight)

    target_keys = set(hf_model.state_dict())
    if set(mapped) != target_keys:
        missing = sorted(target_keys - set(mapped))
        unexpected = sorted(set(mapped) - target_keys)
        raise ValueError(f"Hugging Face parameter mismatch; missing={missing}, unexpected={unexpected}")
    incompatible = hf_model.load_state_dict(mapped, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"strict load failed: {incompatible}")
    hf_model.tie_weights()
    if hf_model.transformer.wte.weight.data_ptr() != hf_model.lm_head.weight.data_ptr():
        raise ValueError("Hugging Face input and output embeddings are not tied")
    return {
        "native_parameter_tensors": len(source_keys),
        "huggingface_parameter_tensors": len(target_keys),
        "transposed_weight_tensors": len(transposed),
        "transposed_weight_names": transposed,
        "trimmed_embedding_rows": native_config.vocab_size - EXPORT_VOCAB_SIZE,
        "missing_parameters": [],
        "unexpected_parameters": [],
    }


def validate_finite(model: GPT2LMHeadModel) -> dict[str, Any]:
    nonfinite: list[str] = []
    checked = 0
    for name, tensor in model.state_dict().items():
        checked += tensor.numel()
        if not torch.isfinite(tensor).all():
            nonfinite.append(name)
    if nonfinite:
        raise ValueError(f"non-finite tensors found: {nonfinite}")
    return {"status": "passed", "values_checked": checked, "nonfinite_tensors": []}


@torch.no_grad()
def validate_equivalence(
    native_model: GPT, hf_model: GPT2LMHeadModel, *, seed: int = EXPECTED_FINAL_STEP
) -> dict[str, Any]:
    torch.manual_seed(seed)
    input_ids = torch.randint(0, EXPORT_VOCAB_SIZE, (2, 24), dtype=torch.long)
    native_model.eval()
    hf_model.eval()
    native_logits, _ = native_model(input_ids)
    native_trimmed = native_logits[..., :EXPORT_VOCAB_SIZE]
    hf_logits = hf_model(input_ids, use_cache=False).logits
    difference = (native_trimmed - hf_logits).abs()
    max_abs = float(difference.max())
    mean_abs = float(difference.mean())
    atol = 5e-5
    rtol = 5e-5
    if not torch.allclose(native_trimmed, hf_logits, atol=atol, rtol=rtol):
        raise ValueError(f"logit equivalence failed: max_abs={max_abs}")

    labels = input_ids[:, 1:]
    native_loss = F.cross_entropy(
        native_trimmed[:, :-1].reshape(-1, EXPORT_VOCAB_SIZE), labels.reshape(-1)
    )
    hf_loss = hf_model(input_ids, labels=input_ids, use_cache=False).loss
    loss_abs = abs(float(native_loss) - float(hf_loss))
    if loss_abs > 1e-5:
        raise ValueError(f"next-token loss equivalence failed: delta={loss_abs}")
    return {
        "status": "passed",
        "seed": seed,
        "input_shape": list(input_ids.shape),
        "comparison_vocabulary_size": EXPORT_VOCAB_SIZE,
        "max_absolute_logit_difference": max_abs,
        "mean_absolute_logit_difference": mean_abs,
        "absolute_tolerance": atol,
        "relative_tolerance": rtol,
        "native_trimmed_next_token_loss": float(native_loss),
        "huggingface_next_token_loss": float(hf_loss),
        "absolute_loss_difference": loss_abs,
        "loss_note": "Native logits are trimmed to the exported 50,257-token vocabulary before cross-entropy; the 47 training-only padding rows are intentionally excluded.",
    }


def validate_tokenizer(tokenizer: Any) -> dict[str, Any]:
    reference = tiktoken.get_encoding("gpt2")
    prompts = [
        "Hello, I'm a language model,",
        "The future of artificial intelligence is",
        "Unicode check: café — π",
    ]
    for prompt in prompts:
        expected = reference.encode(prompt)
        actual = tokenizer.encode(prompt, add_special_tokens=False)
        if actual != expected:
            raise ValueError(f"tokenizer mismatch for prompt {prompt!r}")
    if len(tokenizer) != EXPORT_VOCAB_SIZE or tokenizer.eos_token_id != 50_256:
        raise ValueError("tokenizer vocabulary or EOS token does not match GPT-2")
    return {"status": "passed", "prompts_checked": len(prompts), "vocab_size": len(tokenizer)}


def clean_process_validation(export_dir: Path) -> dict[str, Any]:
    code = f"""
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
path = {str(export_dir)!r}
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True)
inputs = tokenizer('The future of artificial intelligence is', return_tensors='pt')
outputs = model.generate(**inputs, max_new_tokens=12, do_sample=False, pad_token_id=tokenizer.eos_token_id)
print(json.dumps({{'model_class': type(model).__name__, 'tokenizer_class': type(tokenizer).__name__, 'max_generated_id': int(outputs.max()), 'generated_tokens': int(outputs.shape[1])}}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    if int(result["max_generated_id"]) >= EXPORT_VOCAB_SIZE:
        raise ValueError("clean-process generation produced a padded token ID")
    return {"status": "passed", **result}


def render_model_card(
    *, metrics: Mapping[str, Any], checkpoint_record: Mapping[str, Any], license_id: str | None
) -> str:
    license_value = license_id or "LICENSE_PENDING"
    prefix = "---\n" + f"license: {license_value}\n" + "language:\n- en\nlibrary_name: transformers\npipeline_tag: text-generation\ntags:\n- gpt2\n- fineweb-edu\n- pretrained\n---\n\n"
    return prefix + f"""# GPT-2 124M FineWeb-Edu 10B

This is a GPT-2 124M **base text-completion model trained from scratch** on the FineWeb-Edu `sample-10BT` dataset. It is not instruction-tuned and is not a chatbot.

## Model description

- Architecture: GPT-2 decoder-only Transformer
- Parameters: 124,439,808 after export-vocabulary trimming
- Layers / heads / hidden size: 12 / 12 / 768
- Context length: 1,024 tokens
- Export vocabulary: 50,257 GPT-2 tokens
- Training tokens: {metrics['total_tokens_processed']:,}
- Hardware: one NVIDIA H100

The native trainer padded its embedding/output matrix from 50,257 to 50,304 rows for efficient kernels. The final 47 rows were never tokenizer-addressable. This export keeps rows 0–50,256, sets `config.vocab_size=50257`, and preserves tied input/output embeddings, so generation cannot emit a padded ID.

## Final evaluation

| Metric | Value |
|---|---:|
| Training step | {metrics['final_step']:,} |
| Train loss | {metrics['train_loss']:.6f} |
| Validation loss | {metrics['validation_loss']:.6f} |
| Validation perplexity | {metrics['validation_perplexity']:.6f} |
| HellaSwag accuracy | {metrics['hellaswag_results']['accuracy_percent']:.4f}% ({metrics['hellaswag_results']['correct']:,}/{metrics['hellaswag_results']['total']:,}) |

## Intended use

The model is intended for research, education, reproducibility studies, and experiments with small pretrained language models. It performs ordinary next-token completion.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "ctxnn1/gpt2-124m-fineweb-edu-10b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
inputs = tokenizer("The future of artificial intelligence is", return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=64, do_sample=True, top_k=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Limitations, risks, and biases

- This is a 124M-parameter base model and is not competitive with modern large language models.
- It is not instruction-tuned, preference-aligned, safety-tuned, or suitable as a chatbot.
- Outputs can be inaccurate, incoherent, biased, offensive, unsafe, or memorized from pretraining data.
- HellaSwag accuracy is only modestly above the 25% random-choice baseline.
- The run did not include comprehensive safety, fairness, memorization, or downstream-task evaluation.
- Users must evaluate outputs and suitability for their own domain before deployment.

## Training and conversion provenance

- Source code: https://github.com/ctxnn/gpt-2
- W&B: {metrics['wandb_run_url']}
- Native checkpoint SHA-256: `{checkpoint_record['sha256']}`
- Training Git SHA: `{metrics['git_commit_sha']}`
- Cloud execution Git SHA: `{metrics['execution_source_git_sha']}`

The native Linear weights for attention and MLP projections were transposed into Hugging Face GPT-2 `Conv1D` orientation. Positional embeddings, LayerNorm parameters, attention/MLP projections, and tied token embeddings were preserved and validated with native-versus-Hugging-Face logit and loss comparisons.
"""


def validate_upload_tree(export_dir: Path) -> list[str]:
    files = sorted(path.name for path in export_dir.iterdir() if path.is_file())
    unexpected = sorted(set(files) - UPLOAD_FILES)
    missing = sorted(UPLOAD_FILES - set(files))
    if unexpected or missing:
        raise ValueError(f"upload tree mismatch; missing={missing}, unexpected={unexpected}")
    for path in export_dir.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"unsupported upload entry: {path.name}")
    return files


def require_hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required in the protected environment for upload")
    return token


def publish_and_validate(export_dir: Path, repo_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
    token = require_hf_token()
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=export_dir,
        commit_message="Publish verified GPT-2 124M FineWeb-Edu checkpoint",
    )
    revision = commit.oid
    with tempfile.TemporaryDirectory(prefix="hf-gpt2-reload-") as directory:
        snapshot = Path(
            snapshot_download(repo_id=repo_id, repo_type="model", revision=revision, token=token, local_dir=directory)
        )
        uploaded_sha256 = sha256_file(snapshot / "model.safetensors")
        if uploaded_sha256 != metadata["model_safetensors_sha256"]:
            raise ValueError("uploaded model.safetensors checksum mismatch")
        tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=revision, token=token)
        model = AutoModelForCausalLM.from_pretrained(repo_id, revision=revision, token=token)
        inputs = tokenizer("The future of artificial intelligence is", return_tensors="pt")
        output = model.generate(**inputs, max_new_tokens=12, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        max_id = int(output.max())
        if max_id >= EXPORT_VOCAB_SIZE:
            raise ValueError("uploaded model generated an invalid padded token ID")

    publication = {
        "status": "passed",
        "repo_id": repo_id,
        "url": f"https://huggingface.co/{repo_id}",
        "model_revision": revision,
        "downloaded_model_safetensors_sha256": uploaded_sha256,
        "generated_max_token_id": max_id,
    }
    metadata["huggingface_validation"] = publication
    metadata["publication_ready"] = True
    (export_dir / "export_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    final_commit = api.upload_file(
        repo_id=repo_id,
        repo_type="model",
        path_or_fileobj=export_dir / "export_metadata.json",
        path_in_repo="export_metadata.json",
        commit_message="Record post-upload validation",
    )
    remote_file = Path(
        hf_hub_download(repo_id=repo_id, filename="model.safetensors", revision=final_commit.oid, token=token)
    )
    if sha256_file(remote_file) != metadata["model_safetensors_sha256"]:
        raise ValueError("final uploaded checksum verification failed")
    publication["metadata_revision"] = final_commit.oid
    return publication


def convert(args: argparse.Namespace) -> dict[str, Any]:
    complete, checkpoint_record, metrics = validate_source_records(
        checkpoint_path=args.checkpoint,
        complete_path=args.complete,
        checkpoint_record_path=args.checkpoint_record,
        metrics_path=args.metrics,
    )
    try:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(payload, dict) or int(payload.get("completed_step", -1)) != EXPECTED_FINAL_STEP:
        raise ValueError("torch.load succeeded but checkpoint payload has the wrong final step")
    native_config = native_config_from_payload(payload)
    native_model = GPT(native_config)
    native_model.load_state_dict(payload["model"], strict=True)
    checkpoint_git_sha = payload.get("git_commit_sha")
    del payload
    gc.collect()

    hf_config = hf_config_from_native(native_config)
    hf_model = GPT2LMHeadModel(hf_config)
    mapping = map_native_state(native_model.state_dict(), hf_model, native_config)
    finite = validate_finite(hf_model)
    equivalence = validate_equivalence(native_model, hf_model)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    hf_model.save_pretrained(args.output_dir, safe_serialization=True, max_shard_size="10GB")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_source)
    tokenizer_validation = validate_tokenizer(tokenizer)
    tokenizer.save_pretrained(args.output_dir)
    generation_config = GenerationConfig.from_model_config(hf_config)
    generation_config.pad_token_id = 50_256
    generation_config.eos_token_id = 50_256
    generation_config.bos_token_id = 50_256
    generation_config.save_pretrained(args.output_dir)

    model_path = args.output_dir / "model.safetensors"
    if not model_path.is_file():
        raise ValueError("save_pretrained did not create a single model.safetensors file")
    model_sha256 = sha256_file(model_path)
    parameter_count = sum(parameter.numel() for parameter in hf_model.parameters())
    expected_count = int(checkpoint_record["model_parameter_count"]) - (
        native_config.vocab_size - EXPORT_VOCAB_SIZE
    ) * native_config.n_embd
    if parameter_count != expected_count:
        raise ValueError(f"export parameter count mismatch: {parameter_count} != {expected_count}")

    clean_process = clean_process_validation(args.output_dir)
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "source": {
            "training_run_id": metrics["training_run_id"],
            "checkpoint_filename": args.checkpoint.name,
            "checkpoint_step": EXPECTED_FINAL_STEP,
            "checkpoint_size_bytes": args.checkpoint.stat().st_size,
            "checkpoint_sha256": checkpoint_record["sha256"],
            "checkpoint_git_sha": checkpoint_git_sha,
            "complete_verified": complete.get("status") == "completed",
        },
        "architecture": {
            "model_type": "GPT2LMHeadModel",
            "n_layer": native_config.n_layer,
            "n_head": native_config.n_head,
            "n_embd": native_config.n_embd,
            "n_positions": native_config.block_size,
            "native_vocab_size": native_config.vocab_size,
            "export_vocab_size": EXPORT_VOCAB_SIZE,
            "parameter_count": parameter_count,
            "tied_embeddings": hf_model.transformer.wte.weight.data_ptr() == hf_model.lm_head.weight.data_ptr(),
        },
        "mapping": mapping,
        "validation": {
            "finite_parameters": finite,
            "native_huggingface_equivalence": equivalence,
            "tokenizer": tokenizer_validation,
            "clean_process": clean_process,
            "generated_ids_below_vocab_size": clean_process["max_generated_id"] < EXPORT_VOCAB_SIZE,
        },
        "model_safetensors_size_bytes": model_path.stat().st_size,
        "model_safetensors_sha256": model_sha256,
        "license": args.license,
        "publication_ready": bool(args.license),
        "target_repo_id": args.repo_id,
    }
    card = render_model_card(metrics=metrics, checkpoint_record=checkpoint_record, license_id=args.license)
    card_name = "README.md" if args.license else "README.DRAFT.md"
    (args.output_dir / card_name).write_text(card)
    (args.output_dir / "export_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    if args.push_to_hub:
        if not args.license:
            raise RuntimeError("an approved license is required before public upload")
        validate_upload_tree(args.output_dir)
        metadata["huggingface_validation"] = publish_and_validate(
            args.output_dir, args.repo_id, metadata
        )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--complete", type=Path, required=True)
    parser.add_argument("--checkpoint-record", type=Path, default=Path("results/final_checkpoint.json"))
    parser.add_argument("--metrics", type=Path, default=Path("results/final_metrics.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-source", default="openai-community/gpt2")
    parser.add_argument("--repo-id", default="ctxnn1/gpt2-124m-fineweb-edu-10b")
    parser.add_argument("--license", help="Approved Hugging Face license identifier")
    parser.add_argument("--push-to-hub", action="store_true")
    return parser.parse_args()


def main() -> None:
    metadata = convert(parse_args())
    safe_summary = {
        "checkpoint_step": metadata["source"]["checkpoint_step"],
        "source_checkpoint_sha256": metadata["source"]["checkpoint_sha256"],
        "model_safetensors_sha256": metadata["model_safetensors_sha256"],
        "parameter_count": metadata["architecture"]["parameter_count"],
        "publication_ready": metadata["publication_ready"],
    }
    print(json.dumps(safe_summary, indent=2))


if __name__ == "__main__":
    main()
