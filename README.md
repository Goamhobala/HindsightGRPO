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

Prerequisites are Git, `uv`, Python 3.11 (or permission for `uv` to provision it), SLURM, and Linux **bubblewrap** with working unprivileged user namespaces and `--size` support. The login node needs Hugging Face/GitHub/PyPI access during setup. Compute nodes need outbound W&B access; model and dataset access use the prefetched scratch cache. Setup runs only static/synthetic checks on the login node. Generated code executes only in isolated workers within a compute allocation.

`scripts/env.sh` anchors `../../scratch/yhxjin001/HindsightGRPO` at `$HOME`. It puts caches, Python environments, datasets, temporary files, checkpoints, W&B files, logs, and artifacts under that project's `runtime/`. Do not source an older experiment's environment, which may set `WANDB_MODE=disabled`.

Submission verifies the actual `l40s` partition, `l40sfree` account and QoS, and online W&B authentication. It uses the partition/QoS duration limit and queues four independent jobs, each requesting one GPU. An explicit `HINDSIGHT_TIME_LIMIT` may shorten the scheduler maximum; it does not change the three-epoch schedule. If neither scheduler limit is finite, export a duration explicitly.

Each job first runs the real-model GPU preflight and a **full-size, one-update verification run** for its arm. Failed checks prevent that arm's full training. Verification has separate checkpoints and W&B runs; full training starts from pretrained weights. Real L40S/BF16, bubblewrap, remote checkpoint loading, and GPU verification updates must pass on the cluster; they have not been claimed as locally verified.

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

`uv.lock` pins transitive dependencies. Torch 2.6.0 and Transformers 4.57.1 are tested locally. Upstream specified Transformers 4.57.0, which is yanked for installation problems; this uses its patched 4.57.1 release. The architecture loads the immutable checkpoint's custom bidirectional code with `trust_remote_code=True`; its source hashes and dataset fingerprints are recorded and uploaded. Upstream is separately checked out for audit; its optional DeepSpeed stack is unnecessary for these single-GPU jobs.

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

Plots show validation accuracy against updates and GPU-hours, root rewards, and counterfactual differences. Each arm refreshes comparisons from available arms at completion; the last finisher includes all arms. To refresh after synchronizing outputs:

```bash
.venv/bin/python -m hindsight_grpo.plots \
  --run-root runtime/runs/<group>/training \
  --output runtime/runs/<group>/comparison-plots
```

Training records separate collection, scoring, conditional rollouts, reward execution, backward, optimizer, evaluation, and checkpoint costs. Cumulative GPU time is active single-GPU process wall time across saved resumptions; use SLURM accounting for billing/allocation totals, including startup and any work lost to an ungraceful kill. Verification is logged in separate runs and does not count toward training prompts.

## Recovery and verification

Checkpoints use atomic replacement/fsync, save adapters, optimizer, random states, accumulated root and counterfactual gradients, epoch/cursor, and W&B identity. They are saved after each completed prompt. The last two prompts of each epoch are averaged as a two-prompt update. A signal stops at a safe prompt/evaluation-problem boundary; completed examples are not repeated or skipped. In-flight work may be replayed after a hard kill. Partial validation/test results are persisted per problem. Best-checkpoint test evaluation restores final weights before saving any interrupted training state.

At the scheduler's five-minute warning, the wrapper forwards a checkpoint signal and queues a successor with an `afterany` dependency, keeping the same group/run directories. Successors resume until completion. All job IDs are appended to `submissions.tsv`. Cancel a queued successor as well when intentionally stopping an experiment. Ordinary failures stop rather than hiding a broken mathematical or logging check.

```bash
.venv/bin/python -m pytest
squeue -u "$USER"
tail -f runtime/logs/<arm>-<job-id>.out
```

CPU tests cover literal upstream FP32/BF16 sampler agreement, block/history restoration, conditional invariants, all selectors and gradient finite differences, the likelihood counterexample, root/KL mathematics, synthetic optimizer updates for all four arms, valid-state averaging, checkpoint/RNG/partial-gradient recovery, evaluation caching, and strict W&B behavior. The Linux sandbox integration is skipped outside a SLURM Linux environment. GPU preflight verifies the actual model, mask ID, bidirectionality, frozen/current score equality, and finite nonzero LoRA updates before the four real verification updates.

See [docs/upstream.md](docs/upstream.md) and [docs/reward_sandbox.md](docs/reward_sandbox.md) for implementation details.

## Scientific scope

Root optimization uses upstream's **denoising likelihood surrogate**: remask all response tokens, independently mask prompt tokens with probability 0.3 using the same frozen/current seed, and score untempered token log probabilities without causal shifting. The clipped loss averages token ratios. This is not the exact confidence-based trajectory likelihood; an enumerable counterexample is tested.

Keep/alternative comparisons are fixed-position conditional interventions. Only the original token is excluded at that intervention, using the temperature-scaled restricted distribution; later sampling is ordinary. The KL target describes a binary keep-versus-change abstraction. Retrospective selectors depend on the completed reference, and the directional gradient is a first-order reconstruction sensitivity, not a correctness score or exact causal effect. Full historical inputs retain future masks; eventual reference tokens appear only as gradient reconstruction labels. The final outcome measures ordinary unrestricted generation, without counterfactual branching at evaluation.
# HindsightGRPO
