# Development verification — 8 November 2026

Environment: macOS 15.5 ARM64, Python 3.11.12, PyTorch 2.6.0,
Transformers 4.57.1, uv 0.8.4. No W&B credential was available in the development
process; W&B behavior was checked using controlled API doubles, not a live run.

```text
.venv/bin/python -m pytest -q
93 passed, 1 skipped in 24.36s

bash -n scripts/env.sh scripts/setup.sh scripts/submit.sh slurm/run.sh
passed

uv lock --check --offline --cache-dir runtime/uv-cache
Resolved 97 packages
passed
```

The skipped integration test requires Linux, modern bubblewrap, and a SLURM
compute allocation. No generated code executed on the development machine or a
login node. The actual pretrained 0.5B model's GPU preflight, four full-size
verification updates, live W&B authentication/upload, and full training remain
cluster checks. The submission wrapper gates training on the relevant checks;
this document does not assert that those checks have passed.

Local coverage includes:

- Literal upstream FP32 and BF16 sampling equivalence, block/step/history
  reconstruction, independent 4/8 continuations, and single-token exclusion.
- Entropy, remasked reconstruction probability, embedding-to-mask CE derivative,
  no future-token input leakage, full candidate evaluation, and empty targets.
- Leave-one-out advantages, clipping, enumerable sampler likelihood mismatch,
  stable KL targets and analytical derivatives, including confident tokens.
- Synthetic optimizer updates in all four arms, frozen weights, finite nonzero
  adapter updates, valid-state loss averaging, and three-epoch tail batches.
- Atomic checkpoints, optimizer/RNG/partial-gradient recovery, bitwise resumed
  training in all arms, and safe interruption during best-checkpoint evaluation.
- Evaluation pass@k, per-problem recovery, cache provenance, held-out separation,
  scheduler duration/permission checks, strict online W&B, resume IDs, artifacts,
  and credential redaction.
