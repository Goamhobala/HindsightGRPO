# HindsightGRPO

Four matched LoRA experiments on the **diffusion** Qwen2.5-Coder-0.5B-Instruct MDLM: `root`, `entropy_cf`, `r_cf`, and `gradient_ce_cf`. This repository prepares and submits the full experiment; no HPC jobs have been submitted from the development machine.

## On the HPC

Start in your HPC home directory. The leading `../../` is intentional and remains the project-path convention:

```bash
cd ~
git clone <your-repository-url> ../../scratch/yhxjin001/HindsightGRPO
cd ../../scratch/yhxjin001/HindsightGRPO
# For an existing checkout, use git pull here instead.
bash scripts/setup.sh

# Export your existing key in this shell. Never commit it or put it in a config.
# export WANDB_API_KEY=...
export WANDB_PROJECT=HindsightGRPO
# Optional: export WANDB_ENTITY=your-team
bash scripts/submit.sh
```

Prerequisites are Git, `uv`, Python 3.11 (or permission for `uv` to provision it), SLURM, and Linux with working unprivileged user namespaces. Setup checks the installed **bubblewrap** capabilities. If the site's binary is missing required options such as `--disable-userns`, it builds a pinned compatible copy under `runtime/bin/` without sudo. Building needs `cc`, `make`, `ar`, `ranlib`, and `pkg-config`; the installer reports missing tools. It also builds its own static libcap, so system libcap development headers are unnecessary. The login node needs Hugging Face/GitHub/kernel.org/PyPI access during setup. Compute nodes need outbound W&B access; model and dataset access use the prefetched scratch cache. Setup runs only static/synthetic checks on the login node. Generated code executes only in isolated workers within a compute allocation.

`scripts/env.sh` anchors `../../scratch/yhxjin001/HindsightGRPO` at `$HOME`. It puts caches, Python environments, datasets, temporary files, checkpoints, W&B files, logs, and artifacts under that project's `runtime/`. Do not source an older experiment's environment, which may set `WANDB_MODE=disabled`.

Submission verifies the actual `l40s` partition, `l40sfree` account and QoS, and online W&B authentication. It queues four independent jobs, each requesting one node, one task, one GPU, and eight CPUs. Each job requests up to **48 hours**, reduced if the partition/QoS limit is stricter. An explicit `HINDSIGHT_TIME_LIMIT` can shorten this duration but cannot exceed the 48-hour ceiling; the full three-epoch schedule continues through successor jobs. Memory uses the site's default allocation. SLURM emails all supported job events to `yhxjin001@myuct.ac.za`.

Each job first probes actual reward-sandbox execution before loading model weights, then runs the real-model GPU preflight and a **full-size, one-update verification run** for its arm. Failed checks prevent that arm's full training. Verification has separate checkpoints and W&B runs; full training starts from pretrained weights. Real L40S/BF16, bubblewrap, remote checkpoint loading, and GPU verification updates must pass on the cluster; they have not been claimed as locally verified.

## Fixed protocol

The complete configuration is [configs/experiment.json](configs/experiment.json). Scientific constants are validated at load time.

| Setting | Value |
|---|---|
| Ordinary decoding | 256 tokens, 256 commitments, four sequential blocks of 64 |
| Inside each block | Confidence-based any-order; one token per step |
| Temperature / CFG | 0.8 / 0 |
| Root rollouts per prompt | 8 |
| Root advantage | Reward minus mean of the other seven rewards; no standard-deviation normalization |
| Effective update batch | 4 prompts; final epoch batch has 2 without padding |
| Training | All 374 official MBPP training problems, 3 epochs |
| Total per arm | 1,122 training prompts, **282 optimizer updates** |
| Adaptation | LoRA rank 32, alpha 64, dropout 0; attention and MLP projections |
| Optimizer | AdamW, LR 2e-5, betas (0.9, 0.999), epsilon 1e-8, decay 0.01, gradient clipping 1 |
| PPO clipping | 0.2; one update per frozen collection batch |
| Counterfactual reference | Uniformly select one of the eight roots, independently of reward |
| Candidates | Implementation tokens committed before 50% eligible commitment progress |
| Conditional samples | 4 independent keep, 8 independent alternative |
| Counterfactual target | sigmoid(logit(p_old) − 4 × (Q_alternative − Q_keep)) |
| Auxiliary loss | KL(Bern(target) ∥ Bern(current)); coefficient 0.25; average over valid selected states |
| Validation | Every 25 updates and at training completion, all 90 validation tasks |
| Evaluation budget | 8 ordinary samples per problem; pass@1,2,4,8 |
| Held-out test | All 500 tasks: pretrained baseline, final, and validation-selected best |

The prompt split has 10 examples and is kept separate. Prompts include the official public task assertions to specify the required function; they never include the dataset's reference implementation. All arms share prompting, extraction, rewards, eligibility, sampling, root scoring, and evaluation seeds. There is no screened training subset, reduced gradient candidate set, best-of-eight loss, or suffix policy loss. Epoch tails and independent random streams are documented in the configuration/code.

Pins:

| Input | Immutable revision |
|---|---|
| [Diffusion model](https://huggingface.co/dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1) | `a284e895a6248256baf2f60502e54aba61b24c1a` |
| [dLLM upstream](https://github.com/ZHZisZZ/dllm) | `ca176752fbceec49c6b4777a2c18ae88e4eb10ed` |
| [Official MBPP full](https://huggingface.co/datasets/google-research-datasets/mbpp) | `4bb6404fdc6cacfda99d4ac4205087b89d32030c` |

`uv.lock` pins transitive dependencies. Torch 2.6.0 and Transformers 4.57.1 are tested locally. Upstream specified Transformers 4.57.0, which is yanked for installation problems; this uses its patched 4.57.1 release. The loader imports the immutable checkpoint's unchanged bidirectional Python source, verifies its expected SHA-256, and calls those explicit config/model classes' Hugging Face `from_pretrained` methods. The checkpoint file's `__main__` example imports `dllm`; Transformers' Auto-class static scanner treats that example-only import as a runtime dependency. Explicit class loading avoids that false dependency without changing the model source or architecture. Upstream is separately checked out for audit; installing its optional evaluation/DeepSpeed/TRL stack is unnecessary for these single-GPU jobs.

Setup and submission also run `scripts/check_model_imports.py --offline`: a tiny CPU instance of the actual pinned architecture is loaded, saved/reloaded, and checked for forward, input-embedding gradients, and LoRA updates. This catches model-import/loader failures before jobs are queued. The report is `runtime/metadata/model_imports.json`; the full 0.5B BF16 checkpoint remains a GPU preflight check.

## W&B and outputs

Online W&B is mandatory. Missing credentials, disabled/offline modes, and authentication failure stop the run. `sbatch --export=ALL` forwards the exported `WANDB_API_KEY`; neither scripts nor logging copy its value into configuration, Git, or artifacts. `WANDB_PROJECT` and `WANDB_ENTITY` override configuration metadata.

Each arm has a stable W&B run ID, shared experiment group, and its own directories:

```text
runtime/runs/<group>/training/<arm>/
  config.json, wandb_run.json, dataset_manifest.json
  metrics.jsonl, root_trajectories.jsonl, counterfactual.jsonl
  checkpoints/latest.pt, best.pt, final.pt
  evaluation/<checkpoint-label>/<split>/<task-id>.json
  plots/*.png, *.pdf, comparison.json
runtime/runs/<group>/verification/<arm>/
runtime/logs/submissions.tsv
runtime/logs/<arm>-<job-id>.out and .err
runtime/metadata/inputs_manifest.json
runtime/preflight/<group>/<arm>.json
```

Every optimizer update logs root rewards, losses, gradient norms, LR, throughput, and cumulative wall/GPU time. Counterfactual metrics include decision progress, scores, probabilities, keep/alternative means, signed differences, target, auxiliary loss/gradient norm, validity/errors/timeouts. Gradient selection logs every candidate score, CE, full candidate counts, empty-target skips, forward/backward time, and additional peak memory. Detailed records include prompt IDs, tokens, commitment positions/order, block IDs, seeds, and reconstructible masks. JSONL records upload as versioned artifacts at validation/checkpoint publication, interruption, and finish; scalar metrics stream throughout.

Configuration, model/data provenance, checkpoint adapters/optimizer state, evaluation outputs, and plots are W&B artifacts. Standard W&B captures console and system telemetry. Local durable JSONL files remain the audit source. Recovery may replay records written immediately before an uncommitted checkpoint; downstream comparisons should deduplicate by `(arm, epoch, task_id, root_index)` or `(arm, epoch, task_id)` for counterfactual records.

The selected sandbox binary and version are included in preflight records. A local build additionally saves source checksums in `runtime/metadata/bubblewrap.json`. To inspect capabilities without running generated code:

```bash
.venv/bin/python scripts/install_bwrap.py --check
```

Selection prefers `HINDSIGHT_BWRAP` when explicitly set, then the project's `runtime/bin/bwrap`, then the system binary. An obsolete system binary does not override the compatible local copy.

To test actual namespace creation on its own **inside a SLURM compute allocation**:

```bash
.venv/bin/python scripts/check_sandbox.py
```

Failures save UID/thread counts, inherited process limits, available namespace-limit settings, and readable cgroup PID limits in `runtime/preflight/sandbox.json`. Scheduled jobs save per-arm reports under `runtime/preflight/<group>/<arm>-sandbox.json`. The generated-code process cap is applied inside the worker after namespace creation; applying it to the launcher can incorrectly count PyTorch/W&B threads across the user's jobs.

Plots show validation accuracy against updates and GPU-hours, root rewards, and counterfactual differences. Each arm refreshes comparisons from available arms at completion; the last finisher includes all arms. To refresh after synchronizing outputs:

```bash
.venv/bin/python -m hindsight_grpo.plots \
  --run-root runtime/runs/<group>/training \
  --output runtime/runs/<group>/comparison-plots
```

Training records separate collection, scoring, conditional rollouts, reward execution, backward, optimizer, evaluation, and checkpoint costs. Cumulative GPU time is active single-GPU process wall time across saved resumptions; use SLURM accounting for billing/allocation totals, including startup and any work lost to an ungraceful kill. Verification is logged in separate runs and does not count toward training prompts.

## Recovery and verification

Checkpoints use atomic replacement/fsync, save adapters, optimizer, random states, accumulated root and counterfactual gradients, epoch/cursor, and W&B identity. They are saved after each completed prompt. The last two prompts of each epoch are averaged as a two-prompt update. A signal stops at a safe prompt/evaluation-problem boundary; completed examples are not repeated or skipped. In-flight work may be replayed after a hard kill. Partial validation/test results are persisted per problem. Best-checkpoint test evaluation restores final weights before saving any interrupted training state.

At the scheduler's twenty-minute warning, the wrapper forwards a checkpoint signal and queues a successor with an `afterany` dependency, keeping the same group/run directories. Successors resume until completion. All job IDs are appended to `submissions.tsv`. Cancel a queued successor as well when intentionally stopping an experiment. A manual TERM suppresses further submissions by the updated wrapper. Previously submitted scripts retain their original warning/traps. Ordinary failures stop rather than hiding a broken mathematical or logging check.

```bash
.venv/bin/python -m pytest
squeue -u "$USER"
tail -f runtime/logs/<arm>-<job-id>.out
```

CPU tests cover literal upstream FP32/BF16 sampler agreement, block/history restoration, conditional invariants, all selectors and gradient finite differences, the likelihood counterexample, root/KL mathematics, synthetic optimizer updates for all four arms, valid-state averaging, checkpoint/RNG/partial-gradient recovery, evaluation caching, and strict W&B behavior. The Linux sandbox integration is skipped outside a SLURM Linux environment. GPU preflight verifies the actual model, mask ID, bidirectionality, frozen/current score equality, and finite nonzero LoRA updates before the four real verification updates.

See [docs/upstream.md](docs/upstream.md) and [docs/reward_sandbox.md](docs/reward_sandbox.md) for implementation details.

## Continue an existing group with a 200-problem budget

This opt-in pilot stops at **200 total training problems / 50 optimizer updates**,
including work already completed. All arms use the same first 200 problems in
the original seeded epoch-zero shuffle. The scientific config, W&B IDs, root and
counterfactual sampling, optimizer state, and partial gradients remain compatible.
Without a group budget, the original full protocol still applies.

The pilot performs one final full validation and one final test on a deterministic
100-problem held-out subset, with eight samples per problem. Its matched baseline
reuses existing pretrained problem caches; missing baseline problems are generated
only before any optimizer updates. Existing full baseline summaries stay intact.
Intermediate validation and the separate best-checkpoint test are skipped. The
selected IDs and budget are published as a protocol artifact and saved per arm in
run_budget.json. Pilot test metrics use separate `pilot-200-pretrained` and
`pilot-200-final` labels. This is a budget-limited pilot, not the full three-epoch result.

First stop the old processes gracefully. On the cluster, inspect the **current**
IDs, including automatic successors; original job IDs may already have finished:

```bash
squeue -u "$USER" -o '%.18i %.18j %.10T %.30E %R'
tail -n 20 runtime/logs/submissions.tsv
```

Cancel this group's pending successors first. Signal its running batch scripts
with TERM so the wrapper forwards the request to Python. Replace the capitalized
placeholders below with the actual IDs; do not include unrelated jobs:

```bash
scancel --state=PENDING PENDING_JOB_IDS
scancel --batch --signal=TERM RUNNING_JOB_IDS
```

Explicit signalling leaves time for the current prompt and artifact uploads to
finish. Ordinary `scancel` can escalate to SIGKILL after the site's KillWait;
USR1 invokes the old wrapper's successor-submission trap. Wait until these runs
leave the queue, then check for and cancel any successors created while stopping.
Already completed prompts remain in latest.pt; a hard kill can lose the in-flight
prompt. W&B is suitable for monitoring, while scheduler control requires cluster
access. See the official [scancel documentation](https://slurm.schedmd.com/scancel.html).

Update the cluster checkout with these changes, keeping runtime/ intact. If using
the supplied pilot200.patch instead of synchronizing Git, copy it to the cluster
repository, then run `git apply --check pilot200.patch` and `git apply pilot200.patch`.
After the old jobs are stopped, activate the group budget and submit the same group:

```bash
.venv/bin/python -m hindsight_grpo.budget --group hindsight-20261109T073222Z
export EXPERIMENT_GROUP=hindsight-20261109T073222Z
bash scripts/submit.sh
```

The default budget permits 36 further elapsed hours for training and reserves the
remainder of 48 elapsed hours for evaluation. Time starts at activation, includes
queue/startup delays, and is shared across all arms and successors. Reactivation
preserves the original deadlines. Training stops at a complete optimizer batch;
the overall deadline stops evaluation at a sample boundary, preserving completed
problems. A deadline can therefore yield fewer than 200 training problems or an
incomplete evaluation; those runs are marked explicitly. Budget expiration is not
reported as successful completion. No checkpoint beyond 200 prompts is rewound:
such a resume fails with an explanation. Check `latest_update` in W&B before
activation if the runs have continued substantially since the last snapshot.

If one arm has already passed the cutoff, its checkpoint is preserved. To finish
evaluating its current weights without collecting another prompt or taking another
optimizer step, explicitly submit that arm in evaluation-only mode:

```bash
export EXPERIMENT_GROUP=hindsight-20261109T073222Z
bash scripts/submit.sh --arm root --evaluate-only
```

This submits only root, so existing jobs for the other arms can continue. Use it
after the failed root job has exited and any duplicate queued root successor is
removed. Evaluation uses the same test subset and the existing budget deadlines.
The mode is saved in the checkpoint and forwarded to successor jobs.

Evaluation-only mode preserves accumulated gradients without applying them. Its
W&B protocol fields and complete.json report the actual optimizer update, applied
training prompts, and whether the model matches the training budget. For example,
203 processed prompts with three still accumulated means the weights have trained
on 200 prompts; 204 processed prompts after update 51 means they have trained on
204. An unmatched result uses a test label such as `pilot-200-final-update-51`.
It must not be presented as an equal-training-budget comparison. Exact matching
requires a saved update-50 checkpoint; evaluation-only mode cannot reconstruct
past weights or undo updates. Normal training continues to reject checkpoints
beyond the cutoff unless evaluation-only mode was explicitly selected.

W&B summaries now update after every saved training prompt and every evaluation
sample, including `progress/phase`, training counts/target, evaluation problem and
sample counts, rough training ETA, saved checkpoint counts, and current SLURM job
ID. Local progress.json and progress.jsonl retain the same status. A paused run
reports phase `paused`; an expired incomplete run reports `budget_exhausted`.

To obtain a small status file locally without a VPN, authenticate locally with
`.venv/bin/wandb login` if needed, then run:

```bash
.venv/bin/python scripts/wandb_status.py \
  --entity goamhobala-university-of-cape-town \
  --group hindsight-20261109T073222Z \
  --output downloads/wandb-status.json
```

The export reads run summaries only and never exports credentials. API keys belong
in the W&B login flow, not in chat or repository files. Older processes expose
their existing summary fields until they restart with the updated code.

## Scientific scope

Root optimization uses upstream's **denoising likelihood surrogate**: remask all response tokens, independently mask prompt tokens with probability 0.3 using the same frozen/current seed, and score untempered token log probabilities without causal shifting. The clipped loss averages token ratios. This is not the exact confidence-based trajectory likelihood; an enumerable counterexample is tested.

Keep/alternative comparisons are fixed-position conditional interventions. Only the original token is excluded at that intervention, using the temperature-scaled restricted distribution; later sampling is ordinary. The KL target describes a binary keep-versus-change abstraction. Retrospective selectors depend on the completed reference, and the directional gradient is a first-order reconstruction sensitivity, not a correctness score or exact causal effect. Full historical inputs retain future masks; eventual reference tokens appear only as gradient reconstruction labels. The final outcome measures ordinary unrestricted generation, without counterfactual branching at evaluation.
# HindsightGRPO
