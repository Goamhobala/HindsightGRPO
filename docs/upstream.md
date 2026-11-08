# Inspected upstream and likelihood convention

The source inspection pins dLLM commit
`ca176752fbceec49c6b4777a2c18ae88e4eb10ed`:

- [MDLM example](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/examples/a2d/mdlm/sample.py).
- [A2D diffusion GRPO example](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/examples/rl/grpo/a2d/mdlm/train.py).
- [MDLM sampler](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/core/samplers/mdlm.py).
- [Sampler utility functions](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/core/samplers/utils.py).
- [Diffusion GRPO implementation](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/dllm/pipelines/rl/grpo/trainer.py).

The model is
`dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1`, immutable revision
`a284e895a6248256baf2f60502e54aba61b24c1a`. Its custom architecture is
`A2DQwen2LMHeadModel`, model type `a2d-qwen2`, with padding-only bidirectional
attention and no causal logit shift. Its mask token is `<|mask|>` (151665).
The loader uses `AutoModelForMaskedLM` with pinned trusted custom code and
explicit eager attention. It never loads the original autoregressive model.

The recorded sampler reproduces the upstream fixed schedule: four sequential
64-token blocks, 256 total iterations, one commitment per iteration, temperature
0.8, no CFG, no logit shift. At each step, independent float64 Gumbel-Max
provisional token predictions are sampled over the entire canvas. The masked
position in the current block whose sampled token has the greatest **untempered**
predictive probability is committed. This is distinct from selecting the position
whose most probable token has greatest probability. CPU tests compare every
commitment with an independent literal upstream loop using identical randomness.
If the model samples the MASK token as an actual commitment, the sampler raises
an explicit error: suppressing this token would change the requested ordinary
sampling distribution. The token is not silently banned in alternatives either.

Every forward pass receives the full prompt plus the 256-token canvas. Future
blocks are present as masks and participate in bidirectional attention, even
while commitments are restricted to the current block. Accordingly, gradient CE
targets cover every eligible implementation token still masked in this actual
input, including future blocks. Completed-reference values appear only as CE
labels; inputs are reconstructed from historical commitments and masks.

## Root likelihood surrogate

Upstream `DiffuGRPOTrainer._get_per_token_logps` masks the entire completion and
masks prompt tokens with Bernoulli probability 0.3. One bidirectional forward pass
provides untempered log probabilities for completion targets. This repository
uses the same denoising surrogate, with a saved deterministic per-trajectory mask
seed shared by old/current policies. Clipped ratios use these token log
probabilities. There is no reward standard-deviation normalisation. Root
advantages use the requested leave-one-out mean rather than upstream's inclusive
group mean. Post-EOS tokens are excluded from this surrogate objective.

These token probabilities are not exact likelihoods of the confidence-based
decoder. An enumerable two-position test demonstrates the mismatch: provisional
probabilities `(0.9,0.1)` at A and `(0.6,0.4)` at B yield transition probability
0.06 for committing B's first token, while that token's local probability is 0.6.
Position selection depends on predictions at the other position.

## Auxiliary interpretation

Keep and alternative continuations share one historical state. Alternatives are
fixed-position interventions drawn from the temperature-scaled distribution with
the historical token excluded only at that intervention. The suffix sampler then
returns to the ordinary confidence-based policy. The reward contrast is a
conditional intervention contrast. The binary KL projection is not an unbiased
policy gradient for the complete decoder. Hindsight selectors depend on the
completed reference; the gradient score is a first-order directional derivative,
not a correctness or exact causal-importance score.

LoRA uses rank 32, alpha 64, dropout zero, seven upstream attention/MLP projection
modules, and no bias adaptation. Evaluation mode is retained during training to
disable dropout while gradients still update LoRA parameters. The frozen rollout
policy is an independent parameter snapshot; gradient scoring only differentiates
the actual historical input embeddings via `torch.autograd.grad`.
