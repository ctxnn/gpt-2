# GPT-2 124M FineWeb-Edu pretraining report

## Outcome

Pretraining completed and is independently verified at step **19,073** after processing **9,999,745,024 tokens**. The final checkpoint exists in GiveMeANode object storage, its 1,493,919,114-byte size and SHA-256 match the completion pointer and checksum sidecar, and a local CPU `torch.load` succeeded. Final validation and HellaSwag evaluations are present at step 19,073. The W&B run is finished at history step 19,073, and GiveMeANode reports no active GPU nodes.

## Final metrics

| Metric | Final value |
|---|---:|
| Final step | 19,073 |
| Tokens processed | 9,999,745,024 |
| Final train loss | 3.103327 |
| Final validation loss | 3.030832 |
| Validation perplexity | 20.714451 |
| HellaSwag accuracy | 30.0339% (3,016 / 10,042) |
| Cumulative training time | 33,846.821 s (9.402 h) |
| Aggregate throughput | 295,441 tokens/s |
| Mean logged step throughput | 299,864 tokens/s |
| Estimated metered compute | $25.39 at $0.045/H100-minute |
| Platform-reported July MTD spend | $0.00 |

The train loss is the final value stored in the step-19,073 checkpoint. Validation perplexity is `exp(validation_loss)`. Aggregate throughput is total processed tokens divided by the checkpoint’s cumulative training wall time.

## Evaluation evidence

- Final validation: step 19,073, loss `3.0308315753936768`.
- Final HellaSwag: step 19,073, accuracy `0.30033857797251545`, with 3,016 correct answers out of 10,042.
- W&B: run `65e78f54c14046ef99e04e12e7b3e810` is `finished`; its last history step is 19,073. The final regularly logged training row is step 19,070 because training metrics were logged every ten steps, while final validation and HellaSwag were logged at step 19,073.
- W&B run: https://wandb.ai/ctxnn-thapar-university/gpt2-from-scratch/runs/65e78f54c14046ef99e04e12e7b3e810

## Checkpoint verification

- Object: `s3://gpt2-fineweb10b/training/gpt2-124m-fineweb10b-20260729t103500z-36bfc9e/checkpoints/final_step_019073.pt`
- Size: `1,493,919,114` bytes.
- SHA-256: `e519d993d20c98c841ef061f76a1dec3e6ee24d5e55162bdea2a3e2da280fd40`.
- `LATEST.json`, the checksum sidecar, the downloaded byte count, and the independently calculated local SHA-256 all agree.
- CPU `torch.load(..., map_location="cpu", weights_only=False)` passed with PyTorch 2.13.0.
- Loaded checkpoint metadata reports format version 1, completed step 19,073, 9,999,745,024 processed tokens, 124,475,904 model parameters, optimizer state, data-loader state, and RNG state.
- The checkpoint records training Git SHA `36bfc9edd044eb828e118d49c79532eef8440a2a`. The final cloud continuation wrapper ran source SHA `9862792ab4024f9ebec758be73ebe7e75419d09b`; those later changes were artifact/runtime reliability corrections and did not change the preserved training configuration.

## Completion and resource checks

- `training/gpt2-124m-fineweb10b-20260729t103500z-36bfc9e/status/COMPLETE` exists and records final step 19,073.
- Terminal GiveMeANode job `job-znutk` succeeded with zero restarts and zero preemptions.
- GiveMeANode’s live node inventory is empty, confirming the H100 was released.
- A binary/text scan of the downloaded checkpoint, log, history, completion/status files, checksum pointer, and generated-sample artifact found no credential-shaped values.

## Generated samples

Four fixed-seed, top-50 samples were generated from the independently loaded final checkpoint using prompts spanning artificial intelligence, science, narrative text, and problem solving. See `results/generated_samples.md` for prompts, outputs, and reproduction settings.

## Cost methodology and caveats

The $25.39 estimate applies the known batch rate of $0.045 per minute to the checkpoint’s cumulative 33,846.821 seconds of training wall time. GiveMeANode currently reports $0.00 month-to-date spend, so the platform ledger does not corroborate the metered estimate. The estimate does not attempt to reconstruct incidental setup time from failed jobs that did not contribute to the checkpoint’s cumulative training timer.

No checkpoint was deleted. No model was uploaded to Hugging Face, and no instruction tuning was started.
