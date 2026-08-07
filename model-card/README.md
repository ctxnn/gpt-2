# Interactive GPT-2 model card

This directory is a dependency-free static website for the `ctxnn/gpt-2` training project.

## Preview

From the repository root:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000/model-card/`.

## Refresh the evidence bundle

The charts and headline metrics are generated from the authoritative files in `../results/`:

```bash
node model-card/build-evidence.mjs
```

The command refreshes `data/evidence.js`. The original JSON, CSV, checkpoint record, and sample report are also copied into `data/` so the published page remains self-contained.

Run the dependency-free integrity checks with:

```bash
node model-card/test.mjs
```

## Publish

The site contains no build step. After this directory is committed, enable GitHub Pages from the repository root and open:

`https://ctxnn.github.io/gpt-2/`
