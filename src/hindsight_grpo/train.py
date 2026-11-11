"""One GPU per arm; frozen collection, microbatch accumulation, exact prompt cursor."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
from dataclasses import asdict
import fcntl
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .checkpoint import SignalHandler, atomic_json, load_checkpoint, save_checkpoint
from .budget import RunBudget
from .config import ARMS, config_digest, load_config, stable_seed
from .data import completion_text, implementation_token_mask, load_mbpp, prompt_for
from .evaluation import BudgetExpired, StopRequested, evaluate
from .model import frozen_snapshot, load_model, root_logps, selected_token_logp
from .objectives import cf_loss, cf_target, clipped_root_loss, root_advantages
from .rewards import RewardRunner
from .sampler import Sampler
from .selectors import select_decision
from .tracking import Tracker


def average_metrics(records):
    keys = set().union(*(record.keys() for record in records))
    return {key: sum(float(record[key]) for record in records if key in record)
            / sum(key in record for record in records) for key in keys
            if all(isinstance(record.get(key, 0), (int, float, bool)) for record in records)}


class Experiment:
    def __init__(self, config, arm, run_dir, group, verify=False):
        self.lock = None
        self.tracker = None
        self._closed = False
        try:
            self._initialize(config, arm, run_dir, group, verify)
        except BaseException as error:
            # Initialization can fail after opening the lock or starting W&B.
            # Release both before a caller removes a temporary run directory.
            self.__exit__(type(error), error, error.__traceback__)
            raise

    def _initialize(self, config, arm, run_dir, group, verify=False):
        self.config, self.arm, self.run_dir = config, arm, Path(run_dir)
        self.verify = verify
        budget_path = self.run_dir.parent.parent / "run_budget.json"
        self.budget = RunBudget.load(budget_path) if not verify and budget_path.exists() else None
        self.phase = "initializing"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.lock = (self.run_dir / ".training.lock").open("w")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another process is already training this arm/run")
        if not torch.cuda.is_available():
            raise RuntimeError("Full experiment and model verification require a CUDA GPU")
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("Use a SLURM compute allocation; do not train on the login node")
        self.started = time.monotonic()
        random.seed(config["seed"])
        np.random.seed(config["seed"] % 2**32)
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        existing_config = self.run_dir / "config.json"
        if existing_config.exists() and json.loads(existing_config.read_text()) != config:
            raise RuntimeError("Resume requires identical scientific configuration")
        self.tracker = Tracker(self.run_dir, arm, config,
                               group=group + ("-verification" if verify else ""))
        bundle = load_model(config)
        self.model, self.tokenizer = bundle.model, bundle.tokenizer
        self.mask_id = self.tokenizer.mask_token_id
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self.parameters, lr=config["learning_rate"],
            betas=tuple(config["adam_betas"]), eps=config["adam_epsilon"],
            weight_decay=config["weight_decay"])
        self.progress = {"epoch": 0, "cursor": 0, "update": 0, "prompts_processed": 0,
                         "accumulation_count": 0, "pending_metrics": [], "cf_gradients": {},
                         "cf_valid_count": 0, "wall_seconds": 0.,
                         "costs": {}, "baseline_done": False, "best_validation": -1.,
                         "validated_update": -1, "wandb_run_id": self.tracker.run_id,
                         "final_test_done": False, "best_test_done": False,
                         "verification": verify, "complete": False}
        latest = self.run_dir / "checkpoints" / "latest.pt"
        if latest.exists():
            restored = load_checkpoint(latest, self.model, self.optimizer)
            if restored["config"] != config or restored["progress"]["verification"] != verify:
                raise RuntimeError("Resume requires identical scientific configuration and run type")
            self.progress = restored["progress"]
            if self.progress["wandb_run_id"] != self.tracker.run_id:
                raise RuntimeError("Resume W&B run identity mismatch")
        saved_budget = self.progress.get("run_budget")
        active_budget = asdict(self.budget) if self.budget else None
        if saved_budget is not None and saved_budget != active_budget:
            raise RuntimeError("Resume requires the same saved run budget and deadlines")
        if self.budget:
            if self.progress["prompts_processed"] > self.budget.training_prompts:
                raise RuntimeError("Checkpoint already exceeds the training cutoff; cannot undo updates")
            self.progress["run_budget"] = active_budget
        self.previous_wall = self.progress["wall_seconds"]
        self.rollout = frozen_snapshot(self.model)
        self.sampler = Sampler(self.rollout, self.tokenizer,
            generation_length=config["generation_length"], block_size=config["block_size"],
            temperature=config["temperature"])
        self.reward = RewardRunner(timeout_seconds=config["reward_timeout_seconds"],
            memory_mb=config["reward_memory_mb"], temporary_dir=self.run_dir / "tmp")
        self.tracker.record("preflight", self.reward.check_available(), self.progress["update"])
        self.data = load_mbpp(cache_dir=Path(os.environ.get("HF_DATASETS_CACHE", "runtime/datasets")),
                              manifest_path=self.run_dir / "dataset_manifest.json")
        if self.budget:
            if self.budget.training_prompts > len(self.data["train"]):
                raise ValueError("Training cutoff exceeds one pass through the training split")
            self.test_rows = self.budget.test_rows(self.data["test"], config["seed"])
            order = list(range(len(self.data["train"])))
            random.Random(stable_seed(config["seed"], "training-order", 0)).shuffle(order)
            budget_manifest = {"budget": active_budget,
                "training_task_ids": [self.data["train"][i]["task_id"]
                                      for i in order[:self.budget.training_prompts]],
                "test_task_ids": [row["task_id"] for row in self.test_rows],
                "validation": "full split, final checkpoint only", "best_test": False}
            manifest_path = self.run_dir / "run_budget.json"
            if manifest_path.exists() and json.loads(manifest_path.read_text()) != budget_manifest:
                raise RuntimeError("Saved budget/task selection differs from the active plan")
            atomic_json(manifest_path, budget_manifest)
            self.tracker.artifact(f"{arm}-run-budget", manifest_path, type="protocol")
        self.tracker.artifact(f"{arm}-dataset", self.run_dir / "dataset_manifest.json", type="dataset-metadata")
        project = Path(__file__).resolve().parents[2]
        runtime = project / "runtime"
        provenance = [
            runtime / "metadata" / "inputs_manifest.json",
            runtime / "metadata" / "dataset_manifest.json",
            runtime / "metadata" / "bubblewrap.json",
            runtime / "metadata" / "model_imports.json",
            runtime / "installed-requirements.txt",
            runtime / "preflight" / group / f"{arm}.json",
            runtime / "preflight" / group / f"{arm}-sandbox.json",
            runtime / "preflight" / "partition.txt",
            runtime / "preflight" / "partition-detail.txt",
            runtime / "preflight" / "association.txt",
            runtime / "preflight" / "qos.txt",
            project / "pyproject.toml", project / "uv.lock",
        ]
        provenance = [path for path in provenance if path.is_file()]
        if provenance:
            self.tracker.artifact(f"{arm}-provenance", provenance, type="provenance")
        self.signals = SignalHandler()
        self.publish_progress("ready")

    def close(self, exit_code=0):
        """Finish tracking and release the run lock, including on failed runs."""
        if self._closed:
            return
        self._closed = True
        try:
            if self.tracker is not None:
                self.tracker.finish(exit_code=exit_code)
        finally:
            if self.lock is not None and not self.lock.closed:
                self.lock.close()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        try:
            self.close(exit_code=0 if exception is None else 1)
        except Exception as cleanup_error:
            if exception is None:
                raise
            # Keep the original training error visible if tracking also fails
            # during shutdown; the lock is still closed by close()'s finally.
            exception.add_note(f"Experiment shutdown also failed: {type(cleanup_error).__name__}")
        return False

    @contextmanager
    def timed(self, name):
        torch.cuda.synchronize()
        start = time.monotonic()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            costs = self.progress["costs"]
            costs[name] = costs.get(name, 0.) + time.monotonic() - start

    def checkpoint(self, name="latest", upload=False):
        self.progress["wall_seconds"] = self.previous_wall + time.monotonic() - self.started
        path = self.run_dir / "checkpoints" / f"{name}.pt"
        with self.timed("checkpoint_seconds"):
            save_checkpoint(path, self.model, self.optimizer, self.progress, self.config)
        if upload:
            self.tracker.artifact(f"{self.arm}-checkpoint", path, type="model", aliases=[name])
            self.tracker.flush_records()
        self.publish_progress(self.phase, **{"checkpoint/prompts_processed": self.progress["prompts_processed"],
            "checkpoint/update": self.progress["update"], "checkpoint/saved_unix": time.time()})

    def publish_progress(self, phase, **extra):
        self.phase = phase
        p, c = self.progress, self.config
        total = (self.budget.training_prompts if self.budget else
                 c["prompts_per_update"] if self.verify else len(self.data["train"]) * c["epochs"])
        updates = (total // c["prompts_per_update"] if self.budget or self.verify else
                   math.ceil(len(self.data["train"]) / c["prompts_per_update"]) * c["epochs"])
        values = {"progress/phase": phase, "progress/training_prompts_completed": p["prompts_processed"],
            "progress/training_prompts_total": total, "progress/updates_completed": p["update"],
            "progress/updates_total": updates, "latest_update": p["update"],
            "progress/partial_batch_prompts": p["accumulation_count"], **extra}
        if phase == "training" and p["prompts_processed"]:
            training_seconds = sum(value for key, value in p["costs"].items()
                                   if key not in ("evaluation_seconds", "plotting_seconds"))
            values["progress/training_eta_seconds"] = (max(0, total - p["prompts_processed"])
                * training_seconds / p["prompts_processed"])
        if self.budget:
            values.update({"protocol/budget": self.budget.label,
                "protocol/test_problems": self.budget.test_problems,
                "budget/training_deadline_unix": self.budget.training_deadline_unix,
                "budget/deadline_unix": self.budget.deadline_unix})
        self.tracker.progress(values, p["update"])

    def training_cutoff_reached(self):
        if not self.budget:
            return False
        p = self.progress
        if p["prompts_processed"] >= self.budget.training_prompts:
            p["training_stop_reason"] = "prompt_cutoff"
            return True
        if p["accumulation_count"] == 0 and time.time() >= self.budget.training_deadline_unix:
            p["training_stop_reason"] = "training_deadline"
            return True
        return False

    def sync_rollout(self):
        self.rollout.load_state_dict(self.model.state_dict())
        self.rollout.eval()

    def score(self, trajectory, row):
        text = completion_text(trajectory.generated_ids, self.tokenizer)
        return self.reward.evaluate(text, row)

    def collect_prompt(self, row, denominator):
        c, p = self.config, self.progress
        epoch, task = p["epoch"], int(row["task_id"])
        stream = lambda *parts: stable_seed(c["seed"], epoch, task, *parts)
        with self.timed("root_rollout_seconds"):
            roots = [self.sampler.sample(self.tokenizer.encode(prompt_for(row, self.tokenizer),
                        add_special_tokens=False), stream("root", i), prompt_id=str(task))
                     for i in range(c["root_trajectories"])]
        with self.timed("reward_seconds"):
            rewards = [self.score(trajectory, row) for trajectory in roots]
        advantages = root_advantages(torch.tensor([r.reward for r in rewards],
            device=next(self.rollout.parameters()).device))
        metrics = {"root/reward_mean": float(np.mean([r.reward for r in rewards])),
                   "root/reward_std": float(np.std([r.reward for r in rewards])),
                   "root/reward_max": max(r.reward for r in rewards),
                   "root/partial": float(np.mean([r.partial for r in rewards])),
                   "root/syntax_valid": float(np.mean([r.syntax_valid for r in rewards])),
                   "root/timeouts": float(np.mean([r.timeout for r in rewards])),
                   "root/execution_errors": float(np.mean([bool(r.error) for r in rewards]))}
        # The identical old/current corruption seed is independent of decoding RNG.
        root_value = 0.
        with self.timed("root_backward_seconds"):
            for i, trajectory in enumerate(roots):
                seed = stream("surrogate", i)
                with torch.no_grad():
                    old_logp = root_logps(self.rollout, [trajectory], self.mask_id, seed,
                                         c["p_mask_prompt"])[0]
                current = root_logps(self.model, [trajectory], self.mask_id, seed,
                                     c["p_mask_prompt"])[0]
                if not torch.allclose(old_logp, current.detach(), atol=2e-4, rtol=2e-4):
                    # There is exactly one optimizer step per frozen collection batch.
                    raise RuntimeError("Old/current likelihood surrogate differs before update")
                loss = clipped_root_loss(current[None], old_logp[None], advantages[i:i+1],
                                         clip_eps=c["clip_epsilon"])
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite root loss")
                (loss / (len(roots) * denominator)).backward()
                root_value += float(loss.detach()) / len(roots)
        metrics["loss/root"] = root_value
        for i, (trajectory, reward) in enumerate(zip(roots, rewards)):
            self.tracker.record("root_trajectories", {"epoch": epoch, "task_id": task,
                "root_index": i, "trajectory": trajectory.to_dict(), "reward": reward.to_dict(),
                "advantage": float(advantages[i]), "surrogate_seed": stream("surrogate", i)}, p["update"])
        if self.arm != "root":
            # Uniform reference choice is independent of observed rewards.
            reference_index = random.Random(stream("reference")).randrange(len(roots))
            reference = roots[reference_index]
            eligible = implementation_token_mask(reference.generated_ids, self.tokenizer)
            with self.timed("decision_scoring_seconds"):
                selection = select_decision(self.rollout, reference, eligible, self.mask_id,
                                             self.arm, temperature=c["temperature"])
            metrics.update(selection.metrics)
            metrics["cf/valid_selection"] = float(selection.index is not None)
            if selection.index is not None:
                with self.timed("counterfactual_rollout_seconds"):
                    keep, alt = self.sampler.counterfactual_groups(reference, selection.index,
                        [stream("keep", i) for i in range(4)],
                        [stream("alternative", i) for i in range(8)])
                with self.timed("reward_seconds"):
                    kr = [self.score(t, row) for t in keep]
                    ar = [self.score(t, row) for t in alt]
                qk, qa = np.mean([r.reward for r in kr]), np.mean([r.reward for r in ar])
                delta = float(qa - qk)
                with self.timed("cf_backward_seconds"):
                    with torch.no_grad():
                        old = selected_token_logp(self.rollout, reference, selection.index, c["temperature"])
                    current = selected_token_logp(self.model, reference, selection.index, c["temperature"])
                    auxiliary = cf_loss(current, old, delta, beta=c["cf_beta"])
                    if not torch.isfinite(auxiliary):
                        raise RuntimeError("Nonfinite counterfactual loss")
                    auxiliary_grads = torch.autograd.grad(c["cf_coefficient"] * auxiliary,
                        self.parameters, allow_unused=True)
                    gradient_norm = sum(float(g.detach().float().square().sum())
                                        for g in auxiliary_grads if g is not None) ** .5
                    # Root gradients are averaged over all prompts. Auxiliary
                    # gradients must be averaged over VALID selected states only;
                    # keep them separate until the accumulation batch is complete.
                    names = [name for name, parameter in self.model.named_parameters()
                             if parameter.requires_grad]
                    for name, gradient in zip(names, auxiliary_grads):
                        if gradient is not None:
                            value = gradient.detach().float().cpu()
                            if name in p["cf_gradients"]:
                                p["cf_gradients"][name].add_(value)
                            else:
                                p["cf_gradients"][name] = value.clone()
                    p["cf_valid_count"] += 1
                commitment = reference.commitments[selection.index]
                metrics.update({"cf/selector_score": selection.score,
                    "cf/selected_commitment": selection.index, "cf/selected_position": commitment.position,
                    "cf/original_probability": float(old.exp()), "cf/keep_mean": float(qk),
                    "cf/alternative_mean": float(qa), "cf/delta": delta,
                    "cf/alternative_max": max(r.reward for r in ar),
                    "cf/delta_positive": float(delta > 0), "cf/delta_negative": float(delta < 0),
                    "cf/delta_zero": float(delta == 0),
                    "cf/target_probability": float(cf_target(old, delta, c["cf_beta"]).exp()),
                    "loss/counterfactual": float(auxiliary.detach()),
                    "cf/auxiliary_gradient_norm": gradient_norm,
                    "cf/syntax_valid": float(np.mean([r.syntax_valid for r in kr + ar])),
                    "cf/execution_errors": float(np.mean([bool(r.error) for r in kr + ar])),
                    "cf/timeouts": float(np.mean([r.timeout for r in kr + ar]))})
                self.tracker.record("counterfactual", {"epoch": epoch, "task_id": task,
                    "reference_index": reference_index, "selection": selection,
                    "old_log_probability": float(old), "target_log_probability": float(cf_target(old, delta)),
                    "q_keep": float(qk), "q_alternative": float(qa), "delta": delta,
                    "keep": [{"trajectory": t.to_dict(), "reward": r.to_dict()} for t,r in zip(keep,kr)],
                    "alternative": [{"trajectory": t.to_dict(), "reward": r.to_dict()} for t,r in zip(alt,ar)]},
                    p["update"])
            else:
                self.tracker.record("selection_skips", {"epoch": epoch, "task_id": task,
                                                         "selection": selection}, p["update"])
        metrics["loss/total"] = metrics["loss/root"] + c["cf_coefficient"] * metrics.get("loss/counterfactual", 0.)
        return metrics

    def evaluate(self, label, split, cache_source=None, checkpoint_id=None):
        # Snapshot includes the just-updated LoRA weights; all decoding is ordinary.
        self.sync_rollout()
        with self.timed("evaluation_seconds"):
            rows = self.test_rows if self.budget and split == "test" else self.data[split]
            result = evaluate(self.sampler, self.tokenizer, self.reward, rows, split,
                label, self.config, self.run_dir, self.tracker, self.progress["update"], self.signals,
                checkpoint_id=checkpoint_id, cache_source=cache_source,
                progress_callback=lambda values: self.publish_progress("evaluation", **values),
                deadline_unix=self.budget.deadline_unix if self.budget else None)
        wall = self.previous_wall + time.monotonic() - self.started
        metrics = {"compute/wall_seconds": wall, "compute/gpu_hours": wall / 3600,
            **{f"cost/{key}": value for key, value in self.progress["costs"].items()}}
        if split == "validation":
            metrics.update({"validation/accuracy": result["accuracy"],
                            "validation/pass_at_1": result["pass_at_1"]})
        self.tracker.log(metrics, self.progress["update"])
        return result

    def validate(self):
        p = self.progress
        wall = self.previous_wall + time.monotonic() - self.started
        self.tracker.log({"compute/wall_seconds": wall, "compute/gpu_hours": wall / 3600,
            **{f"cost/{key}": value for key, value in p["costs"].items()}}, p["update"])
        result = self.evaluate(f"update-{p['update']}", "validation")
        if result["accuracy"] > p["best_validation"]:
            p["best_validation"] = result["accuracy"]
            self.checkpoint("best", upload=True)
        p["validated_update"] = p["update"]
        self.checkpoint()

    def mark_verification_passed(self):
        atomic_json(self.run_dir / "verification_passed.json", {
            "arm": self.arm, "config_digest": config_digest(self.config),
            "update": self.progress["update"], "prompts": self.progress["prompts_processed"],
            "wandb_url": self.tracker.url, "model_revision": self.config["model_revision"]})
        self.tracker.flush_records()

    def run(self):
        p, c = self.progress, self.config
        with self.signals:
            try:
                # A crash after the durable update checkpoint but before writing
                # the verification marker must not consume another training row.
                if self.verify and p["update"] >= 1:
                    self.mark_verification_passed()
                    return 0
                if self.budget and not p.get("budget_baseline_done"):
                    # Existing trained arms must reuse their original baseline.
                    # Missing caches cannot be regenerated with trained weights.
                    if p["update"]:
                        missing = [row["task_id"] for row in self.test_rows
                                   if not (self.run_dir / "evaluation" / "pretrained" / "test"
                                           / f"{row['task_id']}.json").exists()]
                        if missing:
                            raise RuntimeError("Pretrained baseline cache missing for a trained arm")
                    if not p["baseline_done"]:
                        self.evaluate("pretrained", "validation")
                    self.evaluate(f"{self.budget.label}-pretrained", "test",
                                  cache_source="pretrained", checkpoint_id="pretrained")
                    p["baseline_done"] = p["budget_baseline_done"] = True
                    self.checkpoint()
                elif not self.verify and not p["baseline_done"]:
                    self.evaluate("pretrained", "validation")
                    self.evaluate("pretrained", "test")
                    p["baseline_done"] = True
                    self.checkpoint()
                for epoch in range(p["epoch"], c["epochs"]):
                    if self.training_cutoff_reached():
                        break
                    p["epoch"] = epoch
                    order = list(range(len(self.data["train"])))
                    random.Random(stable_seed(c["seed"], "training-order", epoch)).shuffle(order)
                    # Resumed checkpoint at optimizer boundary may still need validation.
                    if not self.verify and not self.budget and p["update"] and p["update"] % c["validation_every_updates"] == 0 and p["validated_update"] != p["update"]:
                        self.validate()
                    while p["cursor"] < len(order):
                        if self.budget and time.time() >= self.budget.deadline_unix:
                            raise BudgetExpired()
                        if self.training_cutoff_reached():
                            break
                        if self.signals.requested:
                            raise StopRequested()
                        self.publish_progress("training", **{"progress/task_id":
                            self.data["train"][order[p["cursor"]]]["task_id"]})
                        if p["accumulation_count"] == 0:
                            self.sync_rollout()
                            self.optimizer.zero_grad(set_to_none=True)
                        batch_start = p["cursor"] - p["accumulation_count"]
                        denominator = min(c["prompts_per_update"], len(order) - batch_start)
                        metrics = self.collect_prompt(self.data["train"][order[p["cursor"]]], denominator)
                        p["cursor"] += 1
                        p["prompts_processed"] += 1
                        p["accumulation_count"] += 1
                        p["pending_metrics"].append(metrics)
                        if p["accumulation_count"] == denominator:
                            with self.timed("optimizer_seconds"):
                                if p["cf_valid_count"]:
                                    for name, parameter in self.model.named_parameters():
                                        if name in p["cf_gradients"]:
                                            gradient = p["cf_gradients"][name].to(
                                                device=parameter.device, dtype=parameter.dtype) / p["cf_valid_count"]
                                            if parameter.grad is None:
                                                parameter.grad = gradient
                                            else:
                                                parameter.grad.add_(gradient)
                                norm = torch.nn.utils.clip_grad_norm_(self.parameters, c["max_grad_norm"], error_if_nonfinite=True)
                                self.optimizer.step()
                                self.optimizer.zero_grad(set_to_none=True)
                            p["update"] += 1
                            metrics = average_metrics(p["pending_metrics"])
                            metrics["loss/total"] = metrics["loss/root"] + c["cf_coefficient"] * metrics.get("loss/counterfactual", 0.)
                            metrics["cf/valid_states"] = p["cf_valid_count"]
                            p["pending_metrics"], p["accumulation_count"] = [], 0
                            p["cf_gradients"], p["cf_valid_count"] = {}, 0
                            wall = self.previous_wall + time.monotonic() - self.started
                            metrics.update({"train/epoch": epoch + p["cursor"] / len(order),
                                "train/prompts_processed": p["prompts_processed"],
                                "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                                "train/gradient_norm": float(norm), "compute/wall_seconds": wall,
                                "compute/gpu_hours": wall / 3600,
                                "train/prompts_per_second": p["prompts_processed"] / max(wall, 1e-8),
                                "compute/peak_memory_bytes": torch.cuda.max_memory_allocated(),
                                **{f"cost/{key}": value for key,value in p["costs"].items()}})
                            self.tracker.log(metrics, p["update"])
                            self.tracker.summary({"latest_update": p["update"],
                                "latest_root_loss": metrics["loss/root"],
                                "latest_counterfactual_loss": metrics.get("loss/counterfactual", 0.),
                                "latest_total_loss": metrics["loss/total"],
                                "gpu_hours": wall / 3600})
                        self.checkpoint()
                        if self.verify and p["update"] >= 1:
                            self.mark_verification_passed()
                            return 0
                        if not self.verify and not self.budget and p["accumulation_count"] == 0 and p["update"] % c["validation_every_updates"] == 0:
                            self.validate()
                    if self.training_cutoff_reached():
                        break
                    p["epoch"], p["cursor"] = epoch + 1, 0
                    self.checkpoint()
                if not self.verify:
                    if p["validated_update"] != p["update"]:
                        self.validate()
                    self.checkpoint("final", upload=True)
                    if not p["final_test_done"]:
                        self.evaluate(f"{self.budget.label}-final" if self.budget else "final", "test")
                        p["final_test_done"] = True
                        self.checkpoint()
                    if not self.budget and not p["best_test_done"]:
                        # Keep final/optimizer resume state separate from best evaluation.
                        try:
                            load_checkpoint(self.run_dir / "checkpoints" / "best.pt", self.model,
                                            restore_randomness=False)
                            self.evaluate("best", "test")
                            p["best_test_done"] = True
                        finally:
                            # A signal during best-model evaluation must never
                            # checkpoint best weights with final optimizer state.
                            load_checkpoint(self.run_dir / "checkpoints" / "latest.pt", self.model,
                                            self.optimizer, restore_randomness=False)
                    from .plots import make_plots
                    with self.timed("plotting_seconds"):
                        paths = make_plots(self.run_dir.parent, self.run_dir / "plots")
                    self.tracker.artifact(f"{self.arm}-comparison-plots", paths, type="plots")
                    p["complete"] = True
                    self.checkpoint(upload=True)
                    wall = self.previous_wall + time.monotonic() - self.started
                    final_costs = {"compute/wall_seconds": wall, "compute/gpu_hours": wall / 3600,
                        **{f"cost/{key}": value for key, value in p["costs"].items()}}
                    self.tracker.log(final_costs, p["update"])
                    self.tracker.summary({"complete": True, "prompts_processed": p["prompts_processed"],
                        "optimizer_updates": p["update"], "best_validation": p["best_validation"],
                        **final_costs})
                    atomic_json(self.run_dir / "complete.json", {"arm": self.arm, "updates": p["update"],
                                "prompts": p["prompts_processed"], "wandb_url": self.tracker.url,
                                "budget": asdict(self.budget) if self.budget else None,
                                "training_stop_reason": p.get("training_stop_reason", "full_schedule")})
                    self.publish_progress("complete")
                return 0
            except BudgetExpired:
                p["budget_exhausted"] = True
                self.checkpoint(upload=True)
                self.publish_progress("budget_exhausted")
                self.tracker.summary({"complete": False, "budget_exhausted": True})
                return 0
            except StopRequested:
                self.checkpoint(upload=True)
                self.publish_progress("paused")
                return 75
            except Exception as error:
                message = str(error)
                for field in ("WANDB_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
                    secret = os.environ.get(field)
                    if secret:
                        message = message.replace(secret, "[redacted]")
                self.tracker.record("failure", {"exception_type": type(error).__name__,
                    "message": message, "epoch": p["epoch"], "cursor": p["cursor"],
                    "accumulation_count": p["accumulation_count"]}, p["update"])
                self.tracker.flush_records()
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--run-root", default="runtime/runs")
    parser.add_argument("--group", default=os.environ.get("EXPERIMENT_GROUP", "four-arm-full-mbpp"))
    parser.add_argument("--verify-one-update", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    run_dir = Path(args.run_root) / args.group / ("verification" if args.verify_one_update else "training") / args.arm
    with Experiment(config, args.arm, run_dir, args.group, args.verify_one_update) as experiment:
        if experiment.progress.get("complete"):
            code = 0
        else:
            code = experiment.run()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
