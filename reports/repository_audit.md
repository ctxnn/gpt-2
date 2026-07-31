# Repository audit

## Audit snapshot (before preparation changes)

- Absolute path: `/Users/chiragtaneja/Codes/repos/gpt-2`
- Branch: `main`
- HEAD: `285596522f4eb39693e1e642395de3564afc07ce`
- Remote: `origin https://github.com/ctxnn/gpt-2.git` (fetch and push)
- Working tree: **not clean before this task**. Existing state was
  `playground_and_notes.ipynb` modified, with `.DS_Store`,
  `train_gpt2_og.py`, and `understanding.md` untracked. The graph index created
  during this audit also produced an untracked `.codebase-memory/` directory.
- Tracked project files: `README.md`, `fineweb.py`, `hellaswag.py`, `input.txt`,
  `playground_and_notes.ipynb`, `requirements.txt`, `samples.txt`,
  `terminal_loger.py`, and `train_gpt2.py`.
- No tests, configuration directory, Docker files, or `.gitignore` existed.
- No `.pt`, `.pth`, `.bin`, `.safetensors`, checkpoint directory, FineWeb shard
  directory, `.env`, private key, or obvious API-key value was tracked.
  `playground_and_notes.ipynb` (about 2.2 MB) and `input.txt` (about 1.1 MB)
  were the largest tracked files; neither is a model checkpoint or FineWeb
  dataset shard.

This snapshot distinguishes pre-existing user changes from the preparation
work and should be retained when reviewing the final diff.

## Architecture and parameter count

The model is a decoder-only, pre-layer-normalized Transformer. Token and
learned positional embeddings are summed, then passed through 12 blocks. Each
block contains:

1. LayerNorm, 12-head causal self-attention, and a residual connection.
2. LayerNorm, a 4x-expansion GELU MLP, and a residual connection.

Attention uses PyTorch scaled-dot-product attention with `is_causal=True`.
The output projection and MLP residual projections use GPT-2-style scaled
initialization. The language-model head has no bias and is tied to the token
embedding weights.

The original training construction used block size 1,024, vocabulary 50,304,
12 layers, 12 heads, and embedding width 768. Its exact trainable parameter
count is **124,475,904**:

- Token embedding / tied LM head: 38,633,472
- Position embedding: 786,432
- 12 Transformer blocks: 85,054,464
- Final LayerNorm: 1,536

The padded vocabulary adds 47 rows beyond the GPT-2 tokenizer's 50,257 tokens
for more hardware-friendly matrix dimensions. This does not change the
intended GPT-2-small architecture.

## Dataset preparation and tokenizer

`fineweb.py` selected `HuggingFaceFW/fineweb-edu`, configuration
`sample-10BT`, and the train split. It used the `tiktoken` GPT-2 byte-pair
encoding, prepended token 50256 (`<|endoftext|>`) to every document, stored
token IDs as `uint16`, and wrote nominal 100M-token NumPy shards. Shard zero
was validation and all later shards were training data.

Originally, importing `fineweb.py` immediately created a directory, downloaded
the dataset, and started multiprocessing. It had no `if __name__ == "__main__"`
guard or configurable path. That behavior was unsafe in tests and could trigger
a major download at the wrong stage. Dataset preparation is now a separate,
explicit command and the training module never imports or invokes it.

## Batch and token calculations

The intended effective global batch is:

`micro batch 16 × sequence 1,024 × gradient accumulation × DDP world size`

Gradient accumulation is therefore:

`524,288 / (16 × 1,024 × world size) = 32 / world size`

For one H100, this is 32 micro-steps per optimizer step. The divisor must be
integral. The planned token count is:

`19,073 optimizer steps × 524,288 tokens = 9,999,745,024 tokens`

That is approximately one pass over sample-10BT. The old README called 5,000
training steps “5,000 epochs”; that was incorrect. An epoch is a pass over the
dataset, while a step is one optimizer update.

## Original schedule, validation, HellaSwag, and generation

The intended schedule was 715 linear warmup steps from `6e-4 / 715` to `6e-4`,
then cosine decay toward `6e-5`. The implementation used a second independent
`if step > warmup_steps: return min_lr`, so every step after 715 jumped
immediately to `6e-5`; cosine decay never ran. It also had awkward boundary
semantics and no tests.

Validation reset the validation loader and averaged 20 batches every 250
training-loop indices and on the final index. Resetting makes comparisons
deterministic, but the frequency and batch count were hard-coded.

Full HellaSwag evaluated all 10,042 validation examples every 250 steps and at
the end. At the first evaluation it could download HellaSwag from GitHub during
the paid training process. It partitioned examples across DDP ranks and reduced
counts correctly, but the cadence was unnecessarily expensive. Generation also
ran every 250 steps, produced four top-k samples to length 32, and ran on every
rank with a rank-specific fixed seed.

The prepared defaults are validation every 250 steps, samples every 1,000, and
full HellaSwag only at 5,000-step intervals plus the end. Short benchmarks skip
full HellaSwag and final full evaluation. Missing local HellaSwag data is now a
clear preflight failure instead of a training-stage download.

## Original checkpointing, resume, and logging

Original checkpoints were ordinary `torch.save` writes under `log/` at steps
5,000 and final, and only when validation ran. They contained model weights,
the model dataclass, loop index, and validation loss. The stored loop index
described the evaluation before that numbered optimizer update, which created
an off-by-one ambiguity.

They omitted optimizer state, RNG state, data-loader position, tokens
processed, elapsed time, W&B identity, Git SHA, parameter count, resolved
configuration, and a format version. There was no resume path, compatibility
check, atomic write, rolling retention, milestone protection, or final-file
overwrite protection.

The original script opened `log/log.txt` with mode `w` on every start, clearing
history. `terminal_loger.py` also replaced `sys.stdout` at import time and
created a timestamped file, a surprising global side effect. W&B appeared in
`requirements.txt` but was not initialized or used. Training metrics only
recorded step and loss locally; validation and HellaSwag used an ad hoc text
format; samples were stdout-only.

## Correctness bugs

1. The post-warmup scheduler returned minimum LR immediately and bypassed
   cosine decay.
2. `DataLoaderLite.next_batch()` referenced global `B` and `T` instead of
   `self.B` and `self.T`, making instances silently depend on module globals.
3. The shard-boundary test added an extra global stride after already advancing
   position, discarding more data than necessary.
4. Rank-local positions could not be restored because loader state did not
   exist.
5. Setting `model.module.require_backward_grad_sync` changed the wrapped model,
   not the DDP wrapper flag; gradient accumulation could synchronize every
   micro-step and waste communication.
6. Checkpoints labeled with `step` were written before that step's optimizer
   update, making resume semantics ambiguous.
7. Autocast used `device_type="cpu"` for non-CUDA devices, including MPS,
   despite tensors being on MPS.
8. Training, filesystem mutation, DDP initialization, and model construction
   all happened at module import, preventing safe unit testing.

## Reliability and reproducibility risks

- The startup log was always truncated and stderr was not captured.
- Checkpoint writes were non-atomic and a partial file could look valid.
- There was no known-good rolling set or corruption/incompatibility message.
- Model/optimizer/loader/RNG state could not be resumed exactly.
- Seeds were not comprehensively captured; there was no Git or environment
  metadata.
- Paths, hyperparameters, intervals, prompt, and compile behavior were
  hard-coded.
- `torch.compile` was controlled only by editing source and had no isolated
  benchmark path.
- Dataset and evaluation downloads were not separated from paid compute.
- The HellaSwag downloader had no timeout, HTTP status check, or atomic
  replacement, so an interrupted/error response could leave a partial JSONL.
- W&B had no rank guard, stable run ID, local mirror, or failure isolation
  because it was not implemented.
- There was no `try/finally` finalization or machine-readable run status.
- The original FineWeb multiprocessing code did not robustly loop over a
  document larger than the remaining shard capacity.

## Potential unnecessary GPU costs

- Full HellaSwag every 250 steps means roughly 76 complete passes over 10,042
  examples in a 19,073-step run, rather than four planned evaluations.
- Generation every 250 steps on every rank adds repeated inference and logging.
- Incorrect DDP no-sync placement can communicate gradients on every
  accumulation micro-step.
- Downloading HellaSwag after GPU allocation leaves paid hardware waiting on
  network and disk I/O.
- A broken LR schedule can waste the entire run by training under an unintended
  learning rate.
- Non-resumable or corrupt checkpoints can force expensive work to be repeated.
- Silent compile behavior without a controlled benchmark can increase startup
  time or fail after GPU allocation.

## Git and GitHub artifact risks

There was no `.gitignore`, so FineWeb shards, HellaSwag downloads, W&B runs,
logs, environment files, and model checkpoints could be accidentally staged.
The Docker context likewise had no exclusions and could include local datasets,
weights, notebooks, or secrets. The new ignore rules protect data/model/secret
artifacts while allowing intentional small CSV, JSON, Markdown, and PNG reports.

The tracked notebook is about 2.2 MB and `input.txt` about 1.1 MB. They are not
large-file-hosting violations, but reviewers should decide whether they belong
in the long-term source repository. The audit did not alter or remove them.

## Prepared design and remaining operational verification

The preparation introduces YAML configuration and safe CLI overrides, corrected
LR boundaries, synchronized DDP shard transitions, explicit precision/device
handling, opt-in compile, atomic complete checkpoints, rolling/milestone/final
retention, exact `completed_step + 1` resume, append-only local logs, optional
failure-tolerant master-only W&B, CSV/sample/status artifacts, and local tests.

Local CPU tests can verify the software contracts. Before paid training, an
operator must still verify the exact remote filesystem paths, available disk,
H100 bfloat16/SDPA behavior, actual micro-batch memory headroom, W&B protected
secret injection, local FineWeb/HellaSwag completeness, container CUDA
compatibility, checkpoint storage durability, and measured `torch.compile`
benchmark results. Those checks require the later remote stage and were
deliberately not performed in this local-only task.
