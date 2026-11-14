# Upstream attribution

The new `src/d1_exclusion/d1_objective.py` implements the diffu-GRPO conventions
audited from [d1](https://github.com/dllm-reasoning/d1) at commit
`faf85d6dcea40477c722c467023c73b08a46cb82`, distributed under Apache License 2.0.
The standard Apache 2.0 license text is reproduced in
`licenses/dllm-Apache-2.0.txt`. Source-file hashes, references and architectural
adaptations are recorded in `docs/d1_reference.md`. This adaptation does not
imply endorsement by the d1 authors.

The MDLM Gumbel-Max/confidence sampler and denoising likelihood convention in
`src/hindsight_grpo/sampler.py` and `src/hindsight_grpo/model.py` are adapted from
[dLLM](https://github.com/ZHZisZZ/dllm), commit
`ca176752fbceec49c6b4777a2c18ae88e4eb10ed`, distributed under Apache License 2.0.
The upstream license is reproduced in `licenses/dllm-Apache-2.0.txt`.

This adaptation adds exact commitment history, independent continuation streams,
fixed-position interventions, deterministic scoring seeds, and invariant checks.
It does not imply endorsement by the upstream authors. The pinned model and
dataset are downloaded separately and retain their own repository licenses.

`tests/fixtures/hf_mdlm_modeling_qwen2.py` and `hf_mdlm_config.json` reproduce the
published [checkpoint source](https://huggingface.co/dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1/blob/a284e895a6248256baf2f60502e54aba61b24c1a/modeling_qwen2.py)
and configuration at the immutable model revision. The Python fixture is
unchanged, with SHA-256
`708c3f667f7b4eac3fd504ace32e1ee7171cfbfbbc6b6d9d95eda0353a779710`,
and is used to check real model loading without weights, network access, or the
standalone demonstration's optional packages. Its architecture originates from
the same Apache-2.0 dLLM source identified above.
