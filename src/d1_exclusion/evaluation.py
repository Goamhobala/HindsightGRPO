"""Explicit, separately allocated ordinary-generation evaluation of d1 checkpoints."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import os
from pathlib import Path
import random
import shutil
import tempfile

from hindsight_grpo.checkpoint import SignalHandler, atomic_json
from hindsight_grpo.config import config_digest, stable_seed
from hindsight_grpo.evaluation import StopRequested, evaluate, pass_at_k


@contextmanager
def checkpoint_snapshot(path, temporary_dir=None):
    """Copy an atomically published checkpoint once, then identify exactly that copy.

    A training process may replace latest.pt while evaluation starts. An opened
    copy always refers to one saved inode; its hash and loaded weights agree.
    Only trusted local experiment checkpoints may be loaded (they contain pickle).
    """
    path = Path(path)
    with tempfile.TemporaryDirectory(prefix="d1-evaluation-", dir=temporary_dir) as work:
        snapshot = Path(work) / "checkpoint.pt"
        shutil.copyfile(path, snapshot)
        digest = hashlib.sha256()
        with snapshot.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        yield snapshot, digest.hexdigest()


@contextmanager
def evaluation_lock(run_dir, label, split):
    """Fail immediately if two allocations target the same evaluation cache."""
    directory = Path(run_dir) / "evaluation"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f".{label}-{split}.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This checkpoint/split evaluation is already running") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def evaluation_rows(split, seed, limit=None):
    """Deterministic optional task subset, independent of arm and past performance."""
    rows = list(split)
    if limit is None:
        return rows
    if not 1 <= limit <= len(rows):
        raise ValueError("Evaluation task limit must be within the selected split")
    indices = list(range(len(rows)))
    random.Random(stable_seed(seed, "d1-evaluation-subset")).shuffle(indices)
    return [rows[index] for index in indices[:limit]]


def _evaluate_loaded(args, config, model, tokenizer, checkpoint_id, progress):
    from hindsight_grpo.data import load_mbpp
    from hindsight_grpo.rewards import RewardRunner
    from hindsight_grpo.sampler import Sampler
    from hindsight_grpo.tracking import Tracker

    label = ("pretrained" if checkpoint_id == "pretrained" else f"sha256-{checkpoint_id}")
    if args.task_limit is not None:
        label += f"-n{args.task_limit}"
    run_dir = Path(args.run_dir)
    with evaluation_lock(run_dir, label, args.split):
        data = load_mbpp()
        rows = evaluation_rows(data[args.split], config["seed"], args.task_limit)
        output = run_dir / "evaluation" / label / args.split
        output.mkdir(parents=True, exist_ok=True)
        manifest = {
            "checkpoint_id": checkpoint_id, "checkpoint_source": args.checkpoint,
            "config_digest": config_digest(config), "split": args.split,
            "dataset_fingerprint": data[args.split]._fingerprint,
            "task_ids": [int(row["task_id"]) for row in rows],
            "samples_per_problem": config["evaluation_samples"],
            "training_progress": progress,
        }
        existing = output / "manifest.json"
        if existing.exists():
            import json
            cached = json.loads(existing.read_text())
            # A copied checkpoint or changed source alias has identical identity.
            for key in manifest.keys() - {"checkpoint_source"}:
                if cached.get(key) != manifest[key]:
                    raise ValueError("Evaluation manifest provenance mismatch")
        else:
            atomic_json(existing, manifest)
        reward_runner = RewardRunner(config["reward_timeout_seconds"], config["reward_memory_mb"],
                                     temporary_dir=Path("runtime/tmp"))
        reward_runner.check_available()
        sampler = Sampler(model, tokenizer, generation_length=config["generation_length"],
                          block_size=config["block_size"], temperature=config["temperature"],
                          steps=config["denoising_steps"])
        tracker = Tracker(output / "tracking", f"{config['arm']}-evaluation", config,
                          group=f"{run_dir.parent.name}-evaluation")
        exit_code = 0
        try:
            with SignalHandler() as signals:
                metrics = evaluate(sampler, tokenizer, reward_runner, rows, args.split, label,
                                   config, run_dir, tracker, progress.get("optimizer_steps", 0),
                                   signals=signals, checkpoint_id=checkpoint_id)
                tracker.summary({"evaluation": metrics, "checkpoint_id": checkpoint_id})
                return metrics
        except StopRequested:
            exit_code = 75
            raise
        except BaseException:
            exit_code = 1
            raise
        finally:
            tracker.finish(exit_code)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", required=True, help="Trusted d1 checkpoint path, or pretrained")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--task-limit", type=int, help="Explicit deterministic subset; omitted means full split")
    args = parser.parse_args(argv)
    # This command intentionally cannot evaluate on a login node or local GPU.
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Evaluation requires a separate SLURM compute allocation")
    import torch
    from .checkpoint import load_checkpoint
    from .config import load_config
    from hindsight_grpo.model import load_model

    if not torch.cuda.is_available():
        parser.error("A CUDA GPU allocation is required")
    config = load_config(args.config)
    if config["evaluation_samples"] != 8:
        parser.error("This protocol evaluates eight ordinary samples per problem")
    if args.checkpoint == "pretrained":
        bundle = load_model(config, device="cuda", adaptation=False)
        try:
            _evaluate_loaded(args, config, bundle.model, bundle.tokenizer, "pretrained", {})
        except StopRequested:
            return 75
        return 0
    with checkpoint_snapshot(args.checkpoint) as (snapshot, identity):
        bundle = load_model(config, device="cuda", adaptation=True)
        state = load_checkpoint(snapshot, bundle.model, restore_randomness=False)
        if config_digest(state["config"]) != config_digest(config):
            raise ValueError("Evaluation configuration differs from the saved checkpoint")
        try:
            _evaluate_loaded(args, config, bundle.model, bundle.tokenizer, identity, state["progress"])
        except StopRequested:
            return 75
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
