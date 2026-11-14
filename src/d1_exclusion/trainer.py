"""Frozen outer collections, twelve transactional d1 inner optimizer steps."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from hindsight_grpo.data import (completion_text, implementation_token_mask,
                                 load_mbpp, prompt_for)
from hindsight_grpo.model import (frozen_snapshot, load_model, model_device,
                                  selected_token_logp)
from hindsight_grpo.rewards import RewardRunner
from hindsight_grpo.tracking import Tracker

from .checkpoint import (SignalHandler, atomic_json, load_checkpoint,
                          restore_policy, save_checkpoint)
from .config import load_config, stable_seed
from .counterfactual import (bernoulli_kl_from_logprobs,
                             estimate_conditional_values)
from .d1_objective import completion_mask, per_token_terms, root_advantages, token_logps
from .diagnostics import scalar_outcome_metrics
from .proposals import ExclusionSampler
from .selector import select_reference, select_entropy
from .trajectory import Trajectory


class Trainer:
    """Inject tiny models/rewards for CPU tests; production entry point is CUDA only.

    Each persisted state is before collection or after a complete prompt/optimizer
    transaction. A hard kill can replay in-flight work from the last durable state,
    without applying an optimizer update twice to restored weights.
    """
    def __init__(self, config, run_dir, model, tokenizer, reward, rows, *,
                 resume=False, tracker_factory=None, reference_model=None):
        self.config, self.run_dir = config, Path(run_dir)
        self.model, self.tokenizer, self.reward = model.eval(), tokenizer, reward
        self.rows = list(rows)
        if len(self.rows) != config["training_tasks"]:
            raise ValueError("Training row count differs from the configured distinct-task budget")
        self.task_ids = [int(row["task_id"]) for row in self.rows]
        if len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("Training task IDs must be distinct")
        self.reference_model = reference_model
        if reference_model is None and not hasattr(model, "disable_adapter"):
            raise ValueError("Provide a fixed reference model or a PEFT model with disable_adapter()")
        if reference_model is not None:
            reference_model.eval().requires_grad_(False)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.lock = (self.run_dir / ".training.lock").open("a")
        self.tracker = None
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._initialize(resume, tracker_factory)
        except BaseException:
            self.close()
            raise

    def _initialize(self, resume, tracker_factory):
        c = self.config
        latest = self.run_dir / "checkpoints/latest.pt"
        existing = self.run_dir / "config.json"
        if not resume and (existing.exists() or latest.exists()):
            raise FileExistsError("Run already exists; use --resume or choose a new run directory")
        if resume and not latest.exists():
            raise FileNotFoundError("--resume requires checkpoints/latest.pt")
        if existing.exists() and json.loads(existing.read_text()) != c:
            raise ValueError("Resume configuration differs from the saved protocol")
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self.parameters, lr=c["learning_rate"],
            betas=tuple(c["adam_betas"]), eps=c["adam_epsilon"], weight_decay=c["weight_decay"])
        total_steps = math.ceil(len(self.rows) / c["prompts_per_collection"]) * c["inner_iterations"]
        warmup_steps = math.ceil(total_steps * c["warmup_ratio"])
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: min(1.0, step / max(1, warmup_steps)))
        self.progress = {"cursor": 0, "outer_collection": 0, "inner_iteration": 0,
                         "optimizer_steps": 0, "applied_prompts": 0, "collected_prompts": 0,
                         "complete": False, "wall_seconds": 0.0, "costs": {},
                         "task_ids": self.task_ids}
        self.collection = None
        self.old = frozen_snapshot(self.model)
        if resume:
            state = load_checkpoint(latest, self.model, self.optimizer, self.scheduler)
            if state["config"] != c or state["progress"]["task_ids"] != self.task_ids:
                raise ValueError("Checkpoint protocol or data ordering differs")
            self.progress, self.collection = state["progress"], state["collection"]
            if state["old_weights"] is None:
                raise ValueError("Missing frozen rollout policy")
            restore_policy(self.old, state["old_weights"])
        else:
            atomic_json(existing, c)
        self.started, self.previous_wall = time.perf_counter(), self.progress["wall_seconds"]
        self.sampler = ExclusionSampler(self.old, self.tokenizer,
            generation_length=c["generation_length"], block_size=c["block_size"],
            temperature=c["temperature"], steps=c["denoising_steps"])
        self.signals = SignalHandler()
        if tracker_factory:
            self.tracker = tracker_factory(self.run_dir, c["arm"], c,
                                           group=self.run_dir.parent.name)
        if not resume:
            self.checkpoint()
        else:
            # A kill after checkpoint commit but before metric publication is repaired.
            if self.progress.get("last_metrics"):
                self._publish_step(self.progress["last_metrics"])
            self.publish_progress()

    def close(self, exit_code=0):
        try:
            if self.tracker is not None:
                self.tracker.finish(exit_code=exit_code)
                self.tracker = None
        finally:
            if hasattr(self, "lock") and not self.lock.closed:
                self.lock.close()

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.close(exit_code=0 if kind is None else 1)

    @contextmanager
    def timed(self, name):
        device = model_device(self.model)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        try:
            yield
        finally:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            costs = self.progress["costs"]
            costs[name] = costs.get(name, 0.0) + time.perf_counter() - start

    def publish_progress(self):
        self.progress["wall_seconds"] = self.previous_wall + time.perf_counter() - self.started
        atomic_json(self.run_dir / "progress.json", self.progress)
        if self.tracker:
            self.tracker.summary({"training/" + key: value for key, value in self.progress.items()
                                  if isinstance(value, (int, float, bool))})

    def checkpoint(self, name="latest"):
        self.progress["wall_seconds"] = self.previous_wall + time.perf_counter() - self.started
        with self.timed("checkpoint_seconds"):
            save_checkpoint(self.run_dir / f"checkpoints/{name}.pt", self.model,
                            self.optimizer, self.progress, self.config,
                            scheduler=self.scheduler, old_model=self.old,
                            collection=self.collection)
        self.publish_progress()

    def _score(self, trajectory, row):
        return self.reward.evaluate(completion_text(trajectory.generated_ids, self.tokenizer), row).to_dict()

    def collect_prompt(self, row):
        c = self.config
        task = int(row["task_id"])
        stream = lambda *parts: stable_seed(c["seed"], "d1-exclusion", task, *parts)
        prompt = self.tokenizer.encode(prompt_for(row, self.tokenizer), add_special_tokens=False)
        with self.timed("root_sampling_seconds"):
            roots = [self.sampler.sample(prompt, stream("root", i), prompt_id=str(task))
                     for i in range(c["root_trajectories"])]
        with self.timed("reward_seconds"):
            rewards = [self._score(trajectory, row) for trajectory in roots]
        advantages = root_advantages(torch.tensor([r["reward"] for r in rewards])).tolist()
        result = {"task_id": task, "roots": [t.to_dict() for t in roots],
                  "rewards": rewards, "advantages": advantages, "cf": None}
        if c["arm"] == "d1_root":
            return result
        reference_selection = select_reference([r["reward"] for r in rewards], stream("reference"))
        chosen = reference_selection["reference_index"]
        reference = roots[chosen]
        eligible = implementation_token_mask(reference.generated_ids, self.tokenizer)
        with self.timed("selector_seconds"):
            selection = select_entropy(reference, eligible)
        # Selection's terminal reward is metadata only. Conditional reward
        # estimates below use twelve fresh branches, never this root's reward.
        cf = {**reference_selection, "selected_index": selection.index,
              "selection": selection.metrics, "valid": False, "skip_reason": None,
              "original_token": None, "old_keep_logprob": None, "old_keep_probability": None,
              "selected_position": None, "selected_progress": None, "selected_entropy": None,
              "n_alt": None, "n_keep": None, "q_keep": None, "q_alt": None,
              "q_proposal": None, "delta": None, "target_logprob": None,
              "target_probability": None, "target_shift": None,
              "keep_sample_count": 0, "proposal_sample_count": 0}
        result["cf"] = cf
        if selection.index is None:
            cf["skip_reason"] = "no_eligible_implementation_commitment"
            return result
        index = selection.index
        commitment = reference.commitments[index]
        cf.update(selected_position=commitment.position,
                  selected_progress=selection.metrics["selected_progress"],
                  selected_entropy=commitment.entropy)
        with torch.no_grad():
            old_logp = float(selected_token_logp(self.old, reference, index, c["temperature"]))
        keep_seeds = [stream("keep", i) for i in range(c["keep_continuations"])]
        proposal_seeds = [stream("proposal", i) for i in range(c["proposal_continuations"])]
        with self.timed("conditional_sampling_seconds"):
            keeps, proposals = self.sampler.counterfactual_groups(reference, index,
                keep_seeds, proposal_seeds, c["alpha"])
        with self.timed("reward_seconds"):
            keep_rewards = [self._score(t, row) for t in keeps]
            proposal_rewards = [self._score(t, row) for t in proposals]
        position = reference.commitments[index].position
        tokens = [t.tokens[position] for t in proposals]
        estimates = estimate_conditional_values(
            [r["reward"] for r in keep_rewards], [r["reward"] for r in proposal_rewards],
            tokens, reference.commitments[index].token, old_logp, beta=c["cf_beta"])
        cf.update(estimates)
        cf.update({"old_keep_logprob": old_logp, "old_keep_probability": float(np.exp(old_logp)),
                   "theoretical_change_probability": 1 - (1 - c["alpha"]) * float(np.exp(old_logp)),
                   "proposal_tokens": tokens, "keep_rewards": keep_rewards,
                   "proposal_rewards": proposal_rewards,
                   "keep_seeds": keep_seeds, "proposal_seeds": proposal_seeds,
                   "keep_sample_count": len(keeps), "proposal_sample_count": len(proposals),
                   "keep_trajectories": [t.to_dict() for t in keeps],
                   "proposal_trajectories": [t.to_dict() for t in proposals]})
        if cf["valid"]:
            cf["target_probability"] = math.exp(cf["target_logprob"])
            # Preserve tiny shifts when both probabilities round to one.
            cf["target_shift"] = math.exp(old_logp) * math.expm1(cf["target_logprob"] - old_logp)
        return result

    @contextmanager
    def reference_policy(self):
        if self.reference_model is not None:
            yield self.reference_model
        else:
            # PEFT adapter-disabled immutable pretrained model is d1's reference.
            with self.model.disable_adapter():
                yield self.model

    def inner_step(self):
        c, p = self.config, self.progress
        trajectories = [(record, i, Trajectory.from_dict(value))
                        for record in self.collection for i, value in enumerate(record["roots"])]
        eos, pad = getattr(self.tokenizer, "eos_token_id", None), getattr(self.tokenizer, "pad_token_id", None)
        denominator = sum(int(completion_mask(torch.tensor(t.generated_ids), eos, pad).sum())
                          for _, _, t in trajectories)
        if denominator == 0:
            raise ValueError("Frozen collection has no valid response tokens")
        valid_cf = [r for r in self.collection if r["cf"] and r["cf"].get("valid")]
        self.optimizer.zero_grad(set_to_none=True)
        metrics = {"loss/pg": 0.0, "loss/ref_kl": 0.0, "loss/cf": 0.0,
                   "root/clipped_fraction": 0.0, "root/max_ratio": 0.0}
        with self.timed("d1_scoring_backward_seconds"):
            for record, i, trajectory in trajectories:
                if not bool(completion_mask(torch.tensor(trajectory.generated_ids), eos, pad).any()):
                    continue
                seed = stable_seed(c["seed"], "d1-corruption", p["outer_collection"],
                                   p["inner_iteration"], record["task_id"], i)
                with torch.no_grad():
                    old = token_logps(self.old, trajectory, self.tokenizer.mask_token_id, seed,
                                      c["p_mask_prompt"], eos_token_id=eos, pad_token_id=pad)
                    with self.reference_policy() as ref_model:
                        ref = token_logps(ref_model, trajectory, self.tokenizer.mask_token_id, seed,
                                          c["p_mask_prompt"], eos_token_id=eos, pad_token_id=pad)
                current = token_logps(self.model, trajectory, self.tokenizer.mask_token_id, seed,
                                      c["p_mask_prompt"], eos_token_id=eos, pad_token_id=pad)
                terms = per_token_terms(current, old, ref, record["advantages"][i],
                                        epsilon=c["clip_epsilon"], beta=c["ref_kl_beta"])
                loss = terms["loss"].sum() / denominator
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite d1 loss")
                loss.backward()
                metrics["loss/pg"] += float(terms["pg"].detach().sum()) / denominator
                metrics["loss/ref_kl"] += float(terms["kl"].detach().sum()) / denominator
                metrics["root/clipped_fraction"] += float(terms["clipped"].sum()) / denominator
                metrics["root/max_ratio"] = max(metrics["root/max_ratio"], float(terms["ratio"].detach().max()))
        with self.timed("cf_backward_seconds"):
            for record in valid_cf:
                cf = record["cf"]
                trajectory = Trajectory.from_dict(record["roots"][cf["reference_index"]])
                current = selected_token_logp(self.model, trajectory, cf["selected_index"], c["temperature"])
                target = torch.tensor(cf["target_logprob"], device=current.device, dtype=torch.float64)
                auxiliary = bernoulli_kl_from_logprobs(target, current)
                if not torch.isfinite(auxiliary):
                    raise FloatingPointError("Nonfinite auxiliary loss")
                (c["cf_coefficient"] * auxiliary / len(valid_cf)).backward()
                metrics["loss/cf"] += float(auxiliary.detach()) / len(valid_cf)
        with self.timed("optimizer_seconds"):
            norm = torch.nn.utils.clip_grad_norm_(self.parameters, c["gradient_clip"], error_if_nonfinite=True)
            used_lr = self.optimizer.param_groups[0]["lr"]
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
        p["optimizer_steps"] += 1
        p["inner_iteration"] += 1
        p["applied_prompts"] = p["cursor"] + len(self.collection)
        metrics.update({"optimizer_steps": p["optimizer_steps"], "outer_collection": p["outer_collection"],
                        "inner_iteration": p["inner_iteration"], "applied_prompts": p["applied_prompts"],
                        "prompts_in_collection": len(self.collection), "cf/valid_states": len(valid_cf),
                        "gradient_norm": float(norm), "learning_rate": used_lr,
                        "next_learning_rate": self.scheduler.get_last_lr()[0],
                        "root/reward_mean": sum(r["reward"] for record in self.collection for r in record["rewards"])
                            / len(trajectories), "valid_response_tokens": denominator})
        states = [record["cf"] for record in self.collection if record["cf"]]
        selected = [state for state in states if state.get("selected_index") is not None]
        root_rewards = [reward for record in self.collection for reward in record["rewards"]]
        metrics.update({
            "cf/no_eligible_states": sum(state.get("skip_reason") == "no_eligible_implementation_commitment" for state in states),
            "cf/no_alternative_observed": sum(state.get("skip_reason") == "no_alternative_observed" for state in states),
            "cf/selected_states": len(selected),
            "cf/changed_proposals": sum(state.get("n_alt", 0) for state in selected),
            "cf/repeated_proposals": sum(state.get("n_keep", 0) for state in selected),
            "root/mixed_reward_groups": sum(0 < sum(r["reward"] for r in record["rewards"]) < len(record["rewards"])
                                             for record in self.collection),
            "root/syntax_valid": sum(r["syntax_valid"] for r in root_rewards) / len(root_rewards),
            "root/partial": sum(r["partial"] for r in root_rewards) / len(root_rewards),
            "root/timeouts": sum(r["timeout"] for r in root_rewards) / len(root_rewards),
            "root/execution_errors": sum(bool(r["error"]) for r in root_rewards) / len(root_rewards),
        })
        if selected:
            metrics["cf/expected_change_probability"] = sum(s["theoretical_change_probability"] for s in selected) / len(selected)
            metrics["cf/observed_change_fraction"] = sum(s.get("n_alt", 0) for s in selected) / (8 * len(selected))
        if states:
            metrics.update(scalar_outcome_metrics(self.collection))
        metrics["loss/total"] = metrics["loss/pg"] + c["ref_kl_beta"] * metrics["loss/ref_kl"] + c["cf_coefficient"] * metrics["loss/cf"]
        p["last_metrics"] = metrics
        self.checkpoint()
        self._publish_step(metrics)

    def _publish_step(self, metrics):
        step = metrics["optimizer_steps"]
        atomic_json(self.run_dir / f"steps/{step:06d}.json", metrics)
        if self.tracker:
            self.tracker.log(metrics, step)

    def run(self, after_transaction=None):
        """Callbacks enable interruption tests at exactly the durable boundaries."""
        c, p = self.config, self.progress
        with self.signals:
            while p["cursor"] < len(self.rows):
                if self.signals.requested:
                    return 75
                if self.collection is None:
                    self.old.load_state_dict(self.model.state_dict())
                    self.old.eval()
                    self.collection = []
                    p["inner_iteration"] = 0
                    self.checkpoint()
                end = min(p["cursor"] + c["prompts_per_collection"], len(self.rows))
                while p["cursor"] + len(self.collection) < end:
                    row = self.rows[p["cursor"] + len(self.collection)]
                    self.collection.append(self.collect_prompt(row))
                    p["collected_prompts"] = p["cursor"] + len(self.collection)
                    self.checkpoint()
                    if after_transaction:
                        after_transaction(self, "prompt")
                    if self.signals.requested:
                        return 75
                while p["inner_iteration"] < c["inner_iterations"]:
                    self.inner_step()
                    if after_transaction:
                        after_transaction(self, "optimizer")
                    if self.signals.requested:
                        return 75
                completed = {"outer_collection": p["outer_collection"],
                             "optimizer_steps": p["optimizer_steps"], "records": self.collection,
                             "applied_prompts": p["applied_prompts"]}
                atomic_json(self.run_dir / f"collections/{p['outer_collection']:04d}.json", completed)
                p["cursor"] += len(self.collection)
                p["outer_collection"] += 1
                p["inner_iteration"] = 0
                self.collection = None
                self.checkpoint()
                self.checkpoint(f"collection-{p['outer_collection']:04d}")
                if self.tracker:
                    self.tracker.record("collections", completed, p["optimizer_steps"])
                    self.tracker.artifact(f"{c['arm']}-checkpoint", self.run_dir / "checkpoints/latest.pt", type="model")
                    self.tracker.flush_records()
                if after_transaction:
                    after_transaction(self, "collection")
            p["complete"] = True
            self.checkpoint()
            self.checkpoint("final")
            atomic_json(self.run_dir / "completed.json", self.progress)
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    c = load_config(args.config)
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Training requires a SLURM GPU allocation; do not run on a login node")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This protocol requires BF16 support on the allocated L40S")
    # Sandbox execution is checked before model loading, on allocated compute only.
    reward = RewardRunner(timeout_seconds=c["reward_timeout_seconds"], memory_mb=c["reward_memory_mb"],
                          temporary_dir=Path(args.run_dir) / "tmp")
    reward.check_available()
    random.seed(c["seed"])
    np.random.seed(c["seed"] % 2**32)
    torch.manual_seed(c["seed"])
    torch.cuda.manual_seed_all(c["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    data = load_mbpp(cache_dir=os.environ.get("HF_DATASETS_CACHE"))
    from .data import selected_training_rows
    rows = selected_training_rows(data, Path(args.run_dir) / "dataset_manifest.json",
                                   c["seed"], c["training_tasks"],
                                   reference_path=Path(__file__).resolve().parents[2] /
                                       "configs/d1_exclusion/task_manifest.json")
    bundle = load_model(c)
    with Trainer(c, args.run_dir, bundle.model, bundle.tokenizer, reward, rows,
                 resume=args.resume, tracker_factory=Tracker) as trainer:
        return trainer.run()


if __name__ == "__main__":
    raise SystemExit(main())
