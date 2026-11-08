"""Ordinary generation only; held-out test scores never select checkpoints."""
from __future__ import annotations
import math
from pathlib import Path
import time
import json

from .checkpoint import atomic_json
from .config import config_digest, stable_seed
from .data import completion_text, prompt_for


class StopRequested(Exception):
    pass


def pass_at_k(n: int, c: int, k: int) -> float:
    if not (0 <= c <= n and 1 <= k <= n):
        raise ValueError("Invalid pass@k counts")
    return 1. if n - c < k else 1. - math.comb(n - c, k) / math.comb(n, k)


def evaluate(sampler, tokenizer, reward_runner, rows, split, label, config,
             run_dir, tracker, update, signals=None, checkpoint_id=None):
    """Persist each problem independently so validation/test can resume exactly."""
    cache = Path(run_dir) / "evaluation" / label / split
    cache.mkdir(parents=True, exist_ok=True)
    records = []
    started = time.monotonic()
    samples = config["evaluation_samples"]
    provenance = {"config_digest": config_digest(config),
                  "checkpoint_id": str(checkpoint_id if checkpoint_id is not None else label)}
    for row in rows:
        if signals is not None and signals.requested:
            raise StopRequested()
        path = cache / f"{row['task_id']}.json"
        if path.exists():
            record = json.loads(path.read_text())
            expected = {"task_id": row["task_id"], "split": split, "label": label,
                        "samples": samples, "provenance": provenance}
            if (any(record.get(key) != value for key, value in expected.items())
                    or len(record.get("rewards", [])) != samples
                    or len(record.get("trajectories", [])) != samples):
                raise ValueError(f"Evaluation cache provenance or sample-count mismatch: {path}")
        else:
            ids = tokenizer.encode(prompt_for(row, tokenizer), add_special_tokens=False)
            rewards, trajectories = [], []
            for sample in range(samples):
                seed = stable_seed(config["seed"], "evaluation", split, row["task_id"], sample)
                trajectory = sampler.sample(ids, seed, prompt_id=str(row["task_id"]))
                text = completion_text(trajectory.generated_ids, tokenizer)
                rewards.append(reward_runner.evaluate(text, row).to_dict())
                trajectories.append(trajectory.to_dict())
            record = {"task_id": row["task_id"], "split": split, "label": label,
                      "samples": samples, "rewards": rewards, "trajectories": trajectories,
                      "provenance": provenance}
            atomic_json(path, record)
            tracker.record("evaluation", record, update)
        records.append(record)
    all_rewards = [r for item in records for r in item["rewards"]]
    mean = lambda key: sum(float(r[key]) for r in all_rewards) / len(all_rewards)
    metrics = {"accuracy": mean("reward"), "partial": mean("partial"),
               "syntax_valid": mean("syntax_valid"), "timeouts": mean("timeout"),
               "execution_errors": sum(bool(r["error"]) for r in all_rewards) / len(all_rewards),
               "problems": len(records), "samples_per_problem": samples,
               "seconds": time.monotonic() - started}
    for k in (1, 2, 4, 8):
        if k <= samples:
            metrics[f"pass_at_{k}"] = sum(pass_at_k(samples,
                sum(int(r["reward"]) for r in item["rewards"]), k) for item in records) / len(records)
    atomic_json(cache / "summary.json", metrics)
    tracker.log({f"eval/{label}/{split}/{key}": value for key, value in metrics.items()}, update)
    tracker.artifact(f"{tracker.arm}-{label}-{split}", cache, type="evaluation")
    tracker.flush_records()
    return metrics
