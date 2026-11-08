# Upstream attribution

The MDLM Gumbel-Max/confidence sampler and denoising likelihood convention in
`src/hindsight_grpo/sampler.py` and `src/hindsight_grpo/model.py` are adapted from
[dLLM](https://github.com/ZHZisZZ/dllm), commit
`ca176752fbceec49c6b4777a2c18ae88e4eb10ed`, distributed under Apache License 2.0.
The upstream license is reproduced in `licenses/dllm-Apache-2.0.txt`.

This adaptation adds exact commitment history, independent continuation streams,
fixed-position interventions, deterministic scoring seeds, and invariant checks.
It does not imply endorsement by the upstream authors. The pinned model and
dataset are downloaded separately and retain their own repository licenses.
