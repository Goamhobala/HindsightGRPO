# Four-arm d1 token-exclusion experiment

This extension uses the existing diffusion MDLM, MBPP prompts, code extraction,
isolated reward workers, confidence sampler, LoRA loader, atomic checkpoint
primitives and metric logger. It has its own configuration, package and output
namespace. Earlier experiment code, checkpoints and downloaded results remain
available. No new environment or dependencies are needed; use the existing
`.venv` and `uv.lock`.

The root reinforcement-learning objective is d1 diffu-GRPO, shared by all arms.
The earlier leave-one-out root objective is not used. The source audit, immutable
official d1 references and architectural adaptations are documented in
[d1_reference.md](d1_reference.md).

| Configuration | Arm | Intervention strength |
|---|---|---|
| `configs/d1_exclusion/root.json` | `d1_root` | No conditional branches or auxiliary loss |
| `configs/d1_exclusion/no_ban.json` | `entropy_no_ban` | 0 |
| `configs/d1_exclusion/soft_ban.json` | `entropy_soft_ban` | 0.5 |
| `configs/d1_exclusion/hard_ban.json` | `entropy_hard_ban` | 1 |

The three counterfactual configurations inherit identical settings from
`common.json`; only their arm identifier and alpha differ. JSON follows the
repository's existing configuration approach. The shared protocol records
`reference_selection: outcome_stratified_v1` for the amended reference-selection
rule. The root-only arm's training algorithm is unchanged.

## Fixed protocol

The pinned model is
`dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1` at revision
`a284e895a6248256baf2f60502e54aba61b24c1a`, with the existing audited dLLM sampler
revision `ca176752fbceec49c6b4777a2c18ae88e4eb10ed`. It is the bidirectional MDLM,
not the autoregressive Qwen base. Generation commits 256 tokens in 256 denoising
steps, with four sequential blocks of 64, one confidence-selected position at a
time. Temperature is 0.8 and classifier-free guidance is zero. Confidence ranking
retains the upstream model-logit dtype, provisional Gumbel sampling and tie
behavior. Positions within a block are not forced into textual order.

Official MBPP `full` is pinned at
`4bb6404fdc6cacfda99d4ac4205087b89d32030c`. Loading verifies 374 training, 90
validation, 500 test and 10 prompt-split examples, unique task IDs and disjoint
splits. A permutation of the training task IDs seeded with `20261108` selects the
first 150 distinct tasks without screening. Each arm saves its dataset manifest,
including the complete permutation, selected IDs and dataset fingerprints, and
checks it on resume. Public assertions appear in prompts; reference solutions
never do. Binary reward means every supplied public assertion passed. These
results do not establish correctness on hidden tests.

The pinned dataset was also loaded during data-only local verification on
2026-11-14, confirming those actual split counts. The saved reference
[task manifest](../configs/d1_exclusion/task_manifest.json) includes all 374
permuted training IDs, the selected 150, fingerprints and a training-content
SHA-256. Dataset loading did not execute generated programs or load model weights.

Each outer collection contains four prompts and eight ordinary root trajectories
per prompt. The final collection contains two prompts without duplication or
padding: 37 four-prompt collections plus one two-prompt collection, for 38
collections and 456 planned optimizer iterations. Following the official d1
implementation, root losses aggregate over all active response tokens in the
collection; padding and tokens after the first EOS are excluded. The two-prompt
tail uses its actual active-token count. Auxiliary losses average only over
valid selected states in that collection.

All arms use rank-32 LoRA, AdamW with learning rate `3e-6`, betas `(0.9, 0.99)`,
epsilon `1e-8`, weight decay `0.1`, and gradient clipping at `0.2`. The official
d1 constant-with-warmup scheduler uses warmup ratio `0.0001`, yielding one warmup
step for the 456-step schedule. CUDA model loading uses BF16 on L40S.

## d1 and the conditional intervention

Advantages are reward minus the mean of all eight rewards, without leave-one-out
correction or standard-deviation normalization. Likelihood estimation masks the
entire response and independently masks prompt tokens with probability 0.15.
Completed response tokens are targets only; they are never visible response
context in this estimator. Token ratios use current and frozen old scores on
the identical corruption. Clipping epsilon is 0.5. The sampled reference-KL
penalty has coefficient 0.04. The rollout policy is frozen separately from the
trainable adapters; the reference policy is the pretrained model with adapters
disabled, not the latest rollout policy.

The 12 optimizer iterations reuse the exact same responses, rewards, advantages,
conditional branches and detached auxiliary targets. Prompt corruptions follow
the audited d1 convention; old and reference scores match each corruption. Only
current policy weights change during the collection.

For each counterfactual prompt, exactly one reference is selected after scoring
the eight roots. If all eight fail, selection is uniform among the failures; if
all eight succeed, selection is uniform among the successes. For mixed groups,
a seeded draw first chooses success or failure with probability 0.5 each, then
chooses uniformly within that outcome class. This deliberately represents
successful references even when successes are rare. All three counterfactual
arms use the same outcome-stratified rule; neither the highest-reward trajectory
nor failed trajectories are given deterministic priority.

The selected reference then receives the same entropy selector regardless of
its reward: choose the highest recorded, untempered commitment-time entropy
among eligible implementation tokens in the first half of eligible commitment
progress. Progress is historical commitment order, not textual position. No
eligible token means a logged skip and ordinary d1 training continues.

At the selected historical position, let `p` be the old temperature-scaled token
distribution and `a` the historical token. Proposal sampling uses exactly
`q_alpha = (1-alpha)*p + alpha*q_exclude`, where `q_exclude(a)=0`. The historical
token has proposal probability `(1-alpha)*p(a)`. Hard exclusion applies to this
one commitment only. Even at alpha zero, the first historical position is fixed
and its token is resampled; normal confidence-based sampling then resumes.

Four independent keep continuations restore the original post-state. Eight
independent proposal continuations restore the pre-state, fix that position and
draw from the proposal distribution. All twelve continuations are freshly
sampled. The selected reference's own terminal reward never enters the keep
estimate: its outcome influenced reference selection and would bias that
estimate. Proposal repeats of the historical token
restore exactly the keep post-state and are pooled with the four keep rewards:

```
Q_keep = (sum(keep_rewards) + sum(repeated_proposal_rewards)) / (4+n_repeat)
Q_alt  = sum(changed_proposal_rewards) / n_change
Q_proposal = sum(all_proposal_rewards) / 8
```

When `n_change=0`, `Q_alt` is unobserved: all outputs and rewards are retained,
`no_alternative_observed` is logged, and the auxiliary update is skipped without
drawing extra samples. Conditional on changing, proposal tokens follow the same
`q_exclude` distribution at every alpha. The population relationship is

```
Q_proposal - Q_keep = [1-(1-alpha)*p(a)] * (Q_alt-Q_keep).
```

Finite-sample reward estimates need not satisfy this equality exactly. Compare
observed change frequencies with the logged theoretical change probabilities;
do not replace `Q_alt` with the mixture value `Q_proposal`.

The detached target is
`sigmoid(logit(p_old_keep) - 4*(Q_alt-Q_keep))`. The auxiliary loss is
`KL(Bern(target) || Bern(current_keep))`, weighted by 0.25. The target shift depends
only on the independently estimated contrast: a positive `Q_alt-Q_keep` lowers the
historical token's target probability; a negative value raises it. This applies
to successful and failed references alike. Reference success does not certify
the selected token or add an outcome-specific loss, penalty or weight. A zero
contrast supplies no directional counterfactual reward evidence; an unobserved
alternative skips the auxiliary update. The d1 objective and its own 0.04
reference-model KL coefficient remain unchanged.

Outcome stratification intentionally changes which historical states receive
auxiliary training. This is a local selected-state policy-improvement procedure,
not an unbiased estimator of the ordinary prompt-level policy gradient or an
exact policy gradient for the entire confidence-based sampler. The amendment
does not establish superiority to uniform reference selection; that would need
a separate ablation, which is outside these four arms. Zero observed contrasts
are empirical ties. Only one training seed is planned, so independent-run
variability remains unmeasured.

## Reference-outcome diagnostics

Every counterfactual record identifies the root group as `all_failed`, `mixed`
or `all_successful`, the number of successful roots, selected reference index
and binary reward, the class chosen for a mixed group, and the reproducible
reference-selection seed. These fields remain available when no eligible
commitment exists. Selected states also record the original keep probability,
entropy-selected commitment and progress, changed/repeated proposal counts,
conditional reward estimates, signed contrast, validity or skip reason, and
target probability and signed target shift. An unavailable state or alternative
is recorded as unobserved rather than assigned a reward.

Training metrics and saved-data comparisons separately summarize successful
references, failed references, all-success groups, all-failure groups and mixed
groups. The summaries count valid and skipped auxiliary states and signed
contrasts. In particular, they count all-success and all-failure groups with
nonzero counterfactual contrasts even though those roots have zero
group-centred advantages. Such groups can still provide a conditional training
signal. The outcome strata overlap: each example belongs to one reference
outcome stratum and one root-group stratum, so their counts must not be added
as if they were five disjoint groups.

Training scalars describe the current frozen collection and repeat across its
12 inner iterations. Do not sum those repetitions as distinct examples.
`compare.py` aggregates each saved collection's examples once, both over all
available collections and over the common completed training prefix.

## Prepare and submit on the cluster

Start from the existing checkout and environment. Preserve the intentional
relative path exactly:

```bash
cd ~
cd ../../scratch/yhxjin001/HindsightGRPO
# Reuse .venv; use the existing setup only if model/data/sandbox preparation is missing.
export WANDB_PROJECT=HindsightGRPO
# Export your existing WANDB_API_KEY in this shell; never put it in a file.
bash scripts/d1_exclusion/submit_all.sh --dry-run --group d1-exclusion-my-run
bash scripts/d1_exclusion/submit_all.sh --group d1-exclusion-my-run
```

The dry run prints the four configurations and destinations without cluster
commands, environment installation or filesystem writes. Omit `--group` to
generate a unique name. An existing group is rejected before any submission.
If submission stops after queuing only some arms, inspect `submissions.tsv` and
the queue before proceeding; the script deliberately does not resubmit the
whole group automatically.

When explicitly executed by the user, submission queues four independent jobs
with partition `l40s`, account and QoS `l40sfree`, one GPU, eight CPUs and a
24-hour limit. All logs and outputs are arm-specific. The cluster must permit
this requested duration; adjust an operational wall-time request if its current
QoS policy requires it, while preserving scientific parameters. Setup/tests do
not invoke the submission script. No jobs or GPU evaluations were run during
development.

The existing pinned model/data cache and reward sandbox must already be
prepared. Batch jobs verify sandbox execution inside their compute allocation
before model work. The isolated, resource-limited bubblewrap worker has no
network, home directory or training credentials. It logs syntax/extraction
failures, assertion failures, runtime exceptions and timeouts; there is no
unrestricted execution fallback. Online W&B uses the existing tracker and
inherited credentials, with durable local JSONL records.

## Checkpoints, signals and resume

Training performs no full validation or test evaluations. The 24-hour allocation
is a scheduling budget, not a claim that 150 prompts will finish. Conditional
generation and repeated d1 optimization may require more allocations. No
denoising steps, root samples, continuations or inner iterations are reduced to
fit the limit.

The batch wrapper forwards USR1, TERM and INT to Python, including SLURM's
15-minute pre-timeout signal. It waits for the child to finish its safe save and
does not queue successors. Checkpoints atomically persist adapters, optimizer,
scheduler, RNG state, training position, frozen old weights and any frozen
collection with its next inner-iteration index. Recovery resumes committed
optimizer work; a hard kill may replay only work after the last atomic save.
Separate run locks prevent simultaneous trainers targeting one arm.

Start the amended protocol in a new experiment group. Configurations and
checkpoints record `outcome_stratified_v1`; an earlier uniform-reference
checkpoint has a different protocol and is rejected rather than silently
continuing its frozen collection under a new selection rule. Existing outputs
remain intact. The root-only algorithm, sampling allocations, 150-task
manifest, and 12 inner updates are unchanged by this amendment.

After the earlier allocation has ended:

```bash
bash scripts/d1_exclusion/resume.sh d1-exclusion-my-run entropy_soft_ban
# Resume all four arms once every arm has created its first checkpoint:
bash scripts/d1_exclusion/resume.sh d1-exclusion-my-run all
```

Do not launch resume against a still-running arm. Fresh training refuses an
existing training state. A checkpoint from an earlier experiment is rejected.
All intended resume commands retain their original configuration and output
directory. Completed arms have a `completed.json` marker and are skipped by
the resume convenience script.

```
runtime/runs/d1-exclusion/<group>/<arm>/
  config.json, dataset_manifest.json, progress.json
  metrics.jsonl, collections.jsonl
  collections/0000.json ...
  steps/000001.json ...
  checkpoints/latest.pt, collection-0001.pt ..., final.pt
  completed.json
  evaluation/<checkpoint-identity>/<split>/
runtime/logs/d1-exclusion/<group>/<arm>/<job-id>.out, .err
runtime/logs/d1-exclusion/<group>/submissions.tsv
```

Individual fixed collection records preserve root/conditional outputs, rewards,
selection diagnostics, proposal tokens, repeated/changed counts, conditional
estimates and targets. Step records and `progress.json` identify applied
optimizer work. Process elapsed costs are throughput evidence, not measured GPU
busy time or scheduler-billed GPU-hours.

## Separate evaluation and comparisons

Only run evaluations when their additional allocation is intended. The helper
submits a separate job; it is never called by training:

```bash
bash scripts/d1_exclusion/evaluate.sh d1-exclusion-my-run entropy_soft_ban test final
bash scripts/d1_exclusion/evaluate.sh d1-exclusion-my-run entropy_soft_ban validation latest
bash scripts/d1_exclusion/evaluate.sh d1-exclusion-my-run d1_root test pretrained
```

These commands evaluate all 500 test or 90 validation problems with eight
ordinary samples each. Evaluation performs no token exclusion. Within a compute
allocation, the module also accepts an explicit `--task-limit`; it saves a
deterministic, performance-independent subset in a separate cache label. Paired
comparisons require identical task IDs, sample seeds and training exposure.
The pretrained checkpoint needs to be evaluated only once for the shared model,
but retain its task records and provenance rather than substituting its outcomes
for any unobserved trained arm.

Evaluation copies a checkpoint to a stable temporary snapshot and identifies it
by SHA-256, so replacing `latest.pt` cannot change which weights a cache refers
to. Each cache records checkpoint identity, task manifest, dataset fingerprint,
config digest and training progress. Checkpoint/split locks prevent concurrent
cache writers. Completed task records resume without regeneration. Test metrics
never select training checkpoints.

```bash
.venv/bin/python scripts/d1_exclusion/compare.py \
  --run-root runtime/runs/d1-exclusion/d1-exclusion-my-run \
  --output runtime/runs/d1-exclusion/d1-exclusion-my-run/comparison.json
```

Comparison reads saved files only and refuses to overwrite an existing report.
It reports the largest common completed training exposure, separate applied
exposure for partial inner loops, optimizer counts, conditional-discovery
diagnostics by reference outcome and root group, elapsed costs and available
ordinary-generation evaluations. Records from older protocols lacking outcome
metadata remain unclassified; their missing metadata is not inferred from
unobserved outcomes. The
completed exposure counts prompts only after all 12 inner iterations; it does
not infer a matched checkpoint from unequal latest checkpoints. A missing arm
means zero common exposure. Save later comparisons under new filenames.

## Local verification

```bash
.venv/bin/python -m pytest tests/d1_exclusion
.venv/bin/python -m pytest
bash -n slurm/d1_exclusion/train.sbatch
bash scripts/d1_exclusion/submit_all.sh --dry-run --group validation-only
```

Tests use synthetic CPU models and records. They cover objective arithmetic,
frozen-policy handling, intervention distributions, repeated-token pooling,
auxiliary derivatives, interruption/recovery, arm configuration equality,
submission syntax and evaluation provenance. Amendment tests cover seeded
outcome-stratified selection, approximately equal class frequencies in mixed
groups, uniform selection within a class, independent conditional rewards,
outcome-independent contrast directions, and nonzero auxiliary signals in
zero-root-advantage groups. Real L40S/BF16 training and Linux
bubblewrap execution require the cluster and are not claimed as locally
verified. Existing Linux-only integration checks retain their allocation guards.

After the outcome-stratification amendment, local verification on 2026-11-14
used the existing Python 3.11 / PyTorch 2.6.0 environment: **271 tests passed,
one skipped** in the full suite, including 115 tests for this extension.
The skipped check requires Linux bubblewrap in a
SLURM compute allocation. The tiny random-weight PEFT checkpoint fixture emitted
its expected missing-remote-config warning; no model weights were downloaded for
these tests. Shell syntax, module help, Python compilation and the submission dry
run also passed. Synthetic interruption tests compare resumed weights and AdamW
state bit for bit at prompt, inner-update and collection boundaries. Separate
tiny instances of the actual audited MDLM architecture verify adapter/reference
handling and checkpoint recovery. The actual MBPP dataset check is described
above. L40S/BF16 training and real compute-node sandbox execution remain cluster
checks; no jobs or GPU evaluations were launched.
