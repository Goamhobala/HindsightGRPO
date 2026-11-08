# Pinned architecture fixtures

`hf_mdlm_modeling_qwen2.py` and `hf_mdlm_config.json` are unchanged small files
from the actual diffusion checkpoint at revision
`a284e895a6248256baf2f60502e54aba61b24c1a`:

- https://huggingface.co/dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1/blob/a284e895a6248256baf2f60502e54aba61b24c1a/modeling_qwen2.py
- https://huggingface.co/dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1/blob/a284e895a6248256baf2f60502e54aba61b24c1a/config.json

Model source SHA-256:
`708c3f667f7b4eac3fd504ace32e1ee7171cfbfbbc6b6d9d95eda0353a779710`.
This A2D code is distributed under the upstream Apache 2.0 license; see
`licenses/dllm-Apache-2.0.txt` and `THIRD_PARTY_NOTICES.md`.

The tests construct small, random CPU weights from these classes. They never
download or substitute another pretrained architecture. The source includes
an `import dllm` in its guarded `__main__` conversion example, which is not
needed when importing the model classes normally.
