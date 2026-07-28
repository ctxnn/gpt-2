# GPT-2 from scratch

This repository trains the GPT-2-small architecture from random initialization
on the FineWeb-Edu `sample-10BT` dataset. The production configuration uses 12
layers, 12 attention heads, a 768-wide embedding, a 1,024-token context, and a
padded 50,304-token vocabulary (124,475,904 trainable parameters).

The planned run is **19,073 optimizer steps**, not 19,073 epochs. With an
effective batch of 524,288 tokens per optimizer step, that is 9,999,745,024
training tokens—approximately one pass over the 10B-token sample.

## Local preparation

Install the base and development dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

W&B is optional:

```bash
python -m pip install -r requirements-wandb.txt
```

Prepare FineWeb-Edu separately from paid training:

```bash
python fineweb.py --output-dir edu_fineweb10B
```

HellaSwag must also be present before paid training. The trainer deliberately
sets `allow_download=False` and fails clearly if its local validation JSONL is
missing.

Run the tiny, network-free CPU smoke test:

```bash
python train_gpt2.py --smoke-test --benchmark-steps 1
```

Run the production configuration:

```bash
python train_gpt2.py \
  --config configs/gpt2_124m_fineweb10b.yaml \
  --data-root /data/edu_fineweb10B \
  --output-dir /outputs/gpt2-124m
```

Resume from a known-good checkpoint:

```bash
python train_gpt2.py \
  --config configs/gpt2_124m_fineweb10b.yaml \
  --data-root /data/edu_fineweb10B \
  --output-dir /outputs/gpt2-124m \
  --resume /outputs/gpt2-124m/checkpoints/checkpoint_step_010000.pt
```

`torch.compile` is never enabled silently. Opt in with `--compile`, and first
measure it with `--benchmark-steps N`. Benchmarks skip full HellaSwag and final
full evaluation.

## W&B and local artifacts

`WANDB_API_KEY` is read only by W&B from the environment or the platform's
protected secret store. Do not put it in YAML, source, logs, or shell history.
Use `--wandb-mode disabled`, `offline`, or `online`. Only DDP rank 0 initializes
W&B, with a stable run ID and `resume="allow"`. A W&B outage does not stop
training.

The output directory contains:

- `logs/train.log` — stdout and stderr, appended on resume.
- `results/training_history.csv` — local copy of scalar/evaluation metrics.
- `results/generated_samples.md` — prompts and continuations.
- `results/run_status.json` — running/completed/error state.
- `checkpoints/` — atomic rolling, milestone, and final checkpoints.

Checkpoint files contain model, optimizer, loader and RNG state plus resolved
configuration and reproducibility metadata. Training resumes at checkpoint
`completed_step + 1`. The latest three rolling checkpoints are retained;
milestones and the final checkpoint are protected.

## Default evaluation cadence

- Local/W&B scalar metrics every 10 optimizer steps.
- Validation every 250 steps using 20 deterministic batches.
- Samples every 1,000 steps.
- Full HellaSwag at steps 5,000, 10,000, 15,000, and at the end.
- One final validation and one final HellaSwag evaluation.

See [reports/repository_audit.md](reports/repository_audit.md) for the
pre-change audit, correctness findings, and reliability/cost analysis.
