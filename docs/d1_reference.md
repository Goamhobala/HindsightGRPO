# d1 implementation audit

This experiment adapts **diffu-GRPO**, not the original LLaDA model, SFT stage,
reward mixture, datasets, or reported benchmark scores. The source of truth for
the loss is the executable official implementation. No TRL, DeepSpeed, 4-bit
quantization, or additional environment is needed by this adaptation.

## Inspected immutable sources

The official repository was inspected at commit
`faf85d6dcea40477c722c467023c73b08a46cb82` (2026-08-30), retrieved on
2026-11-14. Links below use that immutable revision, rather than a moving main.
SHA-256 values are computed from downloaded source bytes.

| Source | SHA-256 |
| --- | --- |
| [diffu_grpo_trainer.py](https://github.com/dllm-reasoning/d1/blob/faf85d6dcea40477c722c467023c73b08a46cb82/diffu-grpo/diffu_grpo_trainer.py) | `b4c006d773af7f89cc7a6a333465ba2acd580d5767b9e56dd45d35eda3c534dc` |
| [diffu_grpo_config.py](https://github.com/dllm-reasoning/d1/blob/faf85d6dcea40477c722c467023c73b08a46cb82/diffu-grpo/diffu_grpo_config.py) | `22b9b6969802c4a5fd9898cfb89ea4e3474fbb0a3882970b0b3c6215748da5f4` |
| [diffu_grpo_train.py](https://github.com/dllm-reasoning/d1/blob/faf85d6dcea40477c722c467023c73b08a46cb82/diffu-grpo/diffu_grpo_train.py) | `1859bbc9141ab50e0357282d0e18b1df88a4077c7d2d03ed765ca8b962d7e112` |
| [slurm_scripts/train.yaml](https://github.com/dllm-reasoning/d1/blob/faf85d6dcea40477c722c467023c73b08a46cb82/diffu-grpo/slurm_scripts/train.yaml) | `3a92a5cd3424ce4a852de9c605cd38271d44ff2ab1b6cb36bd543c7f5b245913` |
| [run.sh](https://github.com/dllm-reasoning/d1/blob/faf85d6dcea40477c722c467023c73b08a46cb82/diffu-grpo/run.sh) | `6a658129c7f633023411efa2c69c435d69314962653defc50d3d11945739f1df` |
| [env.yml](https://github.com/dllm-reasoning/d1/blob/faf85d6dcea40477c722c467023c73b08a46cb82/env.yml) | `24831b3af45f5ba443edddc371bfe941196d03c2143627c28bae605f7f54f169` |

The [paper, arXiv:2504.12216v1](https://arxiv.org/html/2504.12216v1),
especially equations 2–4, section 3.1 and Algorithm 1, was also inspected.
The official environment pins TRL to
[`0f88c179e30b3439467942a08c3190f624d5c423`](https://github.com/huggingface/trl/blob/0f88c179e30b3439467942a08c3190f624d5c423/trl/trainer/grpo_trainer.py);
that trainer was inspected to resolve reference-policy lifecycle behavior
(SHA-256 `6a6dc510dd5659271886b3a3398ab221d72569486e421bc9386a16c4feff5868`).

## Conventions reproduced

* `forward_process`, lines 242–268, independently masks prompt tokens and
  masks **every response token**. `_get_per_token_logps`, lines 306–362,
  predicts the original response at its own positions with cross entropy.
  No causal shift, response prefix, or historical commitment context is used.
  This is an untempered one-step mean-field surrogate, separate from the
  temperature-0.8 sampling distribution and local auxiliary objective.
* The advantages are rewards minus the **whole group mean**, with no
  standard-deviation division. The standard deviation computed by the official
  trainer is a diagnostic; its nearby normalization comment does not describe
  an actual division. Leave-one-out centering from the previous experiment is
  not used.
* `compute_loss`, lines 115–122, uses a separate ratio for each sampled token:
  `exp(current_logp - old_logp)`, clipped to `[0.5,1.5]`. Both signs of the
  advantage pass through the standard minimum before negation. There is no
  sequence-level ratio and no clamp of log ratios in this implementation.
  Nonfinite losses fail before an optimizer step.
* The reference penalty is the sampled-token approximation
  `exp(ref_logp-current_logp) - (ref_logp-current_logp) - 1`, coefficient 0.04.
  This is the official approximate KL estimator, rather than a full-vocabulary
  KL or a trajectory KL. The implementation evaluates the equivalent
  `expm1(d)-d` to reduce cancellation near zero.
* The first EOS token contributes to scoring, while later response tokens do
  not (official lines 445–450). The generated diffusion window is still fully
  masked and retains its original length in the model input.
* The launch YAML, rather than dataclass defaults, establishes prompt masking
  0.15, epsilon 0.5 and 12 inner iterations. Dataclass defaults such as epsilon
  0.2 or one inner iteration do not describe the published launch recipe.
* Its optimizer uses AdamW, learning rate `3e-6`, betas `(0.9,0.99)`, weight
  decay `0.1`, gradient clipping `0.2`, and `constant_with_warmup` scheduling
  with `warmup_ratio=0.0001`. Following Transformers' ceiling convention,
  this experiment's 456 planned optimizer iterations have one warmup iteration:
  the first scheduled learning rate is zero, then it is `3e-6`. Scheduler and
  optimizer states are restored together; resumption does not restart warmup.

## Paper/code differences and architectural adaptations

**Token aggregation.** Paper equation 4 displays an equal mean over responses,
each divided by its response length. Official `compute_loss` instead divides
the sum of all masked token losses by the total number of valid tokens in its
batch (line 122). This code follows the latter convention across the complete
outer collection. Streaming one response at a time therefore accumulates each
response's token-loss **sum divided by the same collection-wide denominator**.
It does not average response or prompt means. This matters when EOS lengths
differ and ensures correct normalization of the final two-prompt collection.

**Clipping diagnostic.** Official line 130 records
`unclipped_surrogate < clipped_surrogate` under the name `is_clipped`.
That predicate selects cases where the minimum uses the *unclipped* term.
The optimized loss is unchanged here, but `root/clipped_fraction` measures
actual clipping activation (`unclipped_surrogate > clipped_surrogate`).
`per_token_terms` also exposes the original predicate as `official_is_clipped`
for direct audit comparisons. Equality contributes to neither predicate.

**Three policy roles.** The frozen old policy includes the LoRA adapter at the
start of an outer collection. Its responses and rewards stay fixed during all
12 optimizer iterations. The current model has trainable LoRA weights and a
frozen base. The reference for this PEFT implementation is the original base
MDLM **with adapters disabled**. Official custom trainer lines 479–483 score
that reference explicitly; it does not read a moving `ref_model`. Although
the official YAML sets `sync_ref_model: True`, the pinned TRL parent sets
`ref_model=None` for PEFT and its reference synchronization setting does not
change the adapter-disabled scores used here. This experiment keeps that
effective fixed-base reference. A frozen reference toy model can be supplied
for CPU tests; it is not confused with a snapshot of the evolving adapter.

**Repeated corruption.** Official lines 454–483 draw one masking seed for each
inner iteration and precompute old/reference scores for all those corruptions
before any optimizer update. Current scoring reuses the iteration's seed.
This adaptation can recompute matching old/reference scores on demand using
the frozen old snapshot and fixed base reference; with deterministic inference
this is the same estimator. Seeds are deterministically derived from training
seed, collection, inner iteration, task and root index, and a local generator
does not reset global training randomness. Each new inner iteration uses a new
prompt corruption while all sampled outputs, rewards, advantages and auxiliary
targets remain frozen. The original implementation's small random seed range
and global `set_seed` side effects are not needed for algorithmic equivalence.

**MDLM attention and stops.** The existing pinned Qwen diffusion model is
bidirectional and accepts attention masks. Training scores one unpadded response
at a time; every position in its original diffusion window attends normally.
The generic corruption helper also supports explicit prompt padding with
attention zero. Unlike d1's tokenizer setup (PAD assigned to EOS), this model
has a distinct PAD ID. A generated distinct PAD and its suffix are excluded
from scoring, matching the shared MBPP decoder's termination rule; EOS remains
inclusive. Post-stop output tokens are never visible to the likelihood model.
This stop adaptation is explicit rather than silently treating padding as code.

**Generation and training scope.** Existing upstream-compatible confidence
sampling is reused: four sequential 64-token blocks, 256 denoising steps,
one commitment per step, temperature 0.8, low-confidence remasking and CFG 0.
It is not replaced by d1's original LLaDA generation implementation. LoRA rank
32, eight roots, four prompts per collection, 150 distinct MBPP tasks and the
new branch interventions are this experiment's controlled adaptations. The
root objective has no continuation-level GRPO or branch-distillation term.

**Outcome-stratified historical references.** The amended auxiliary procedure
selects one root uniformly when all eight outcomes are identical. In a mixed
group it first chooses successful versus failed references with equal
probability, then selects uniformly within that class. The same maximum
commitment-time entropy selector and first-half eligible-progress restriction
apply after either outcome class is selected. This is an adaptation of the
auxiliary selected-state procedure, not a change to d1's root objective or its
adapter-disabled reference model for KL regularization. These two uses of
"reference" have separate roles.

The selected trajectory's reward is selection metadata only. Four fresh keep
continuations and eight fresh proposals estimate conditional values without
reusing that terminal reward. The signed alternative-minus-keep contrast alone
determines the unchanged Bernoulli target and loss: successful references can
have their selected token discouraged, and failed references can have it
reinforced. Improvement strength 4, auxiliary coefficient 0.25, and all twelve
inner updates remain unchanged. The protocol identifier
`outcome_stratified_v1` prevents silently resuming a uniform-reference collection
as if it had been sampled with this rule.

Stratification intentionally selects a different distribution of historical
states for auxiliary learning. It does not give an unbiased ordinary
prompt-level policy-gradient estimator, and it does not establish an advantage
over uniform reference selection without a separate ablation. No extra arm or
ablation is included. Diagnostics report successful and failed references and
all-success, all-failure and mixed root groups, including nonzero conditional
contrasts from homogeneous groups whose d1 advantages are zero.

## CPU verification

`tests/d1_exclusion/test_d1_objective.py` compares the loss against the literal
official algebra, detects token-mean versus response-mean differences, and
checks streaming loss/gradient parity for both four-prompt collections and
the two-prompt tail. It verifies group centering, old/reference detachment,
ratio clipping, reference gradient directions, exact EOS/PAD masks, prompt
padding attention, reproducible corruption without global RNG mutation, and
the absence of response-token leakage or causal shifts. Tests use tiny CPU
models and require no model download, generated-code execution, GPU allocation
or cluster submission.
