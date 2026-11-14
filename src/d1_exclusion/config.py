"""Validated, isolated protocol for the four d1 exclusion conditions."""
from __future__ import annotations

import json
from pathlib import Path

from hindsight_grpo.config import config_digest, stable_seed
from hindsight_grpo.model import MODEL_ID, MODEL_REVISION, UPSTREAM_COMMIT
from hindsight_grpo.data import DATASET_ID, DATASET_REVISION

ARMS = {"d1_root": None, "entropy_no_ban": 0.0,
        "entropy_soft_ban": 0.5, "entropy_hard_ban": 1.0}
FIXED = {
    "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
    "upstream_commit": UPSTREAM_COMMIT, "dataset_id": DATASET_ID,
    "dataset_revision": DATASET_REVISION, "generation_length": 256,
    "denoising_steps": 256, "block_size": 64, "temperature": 0.8,
    "cfg_scale": 0, "seed": 20261108, "training_tasks": 150,
    "root_trajectories": 8, "prompts_per_collection": 4,
    "inner_iterations": 12, "p_mask_prompt": 0.15, "clip_epsilon": 0.5,
    "ref_kl_beta": 0.04, "lora_rank": 32, "lora_alpha": 64,
    "lora_dropout": 0.0, "learning_rate": 3e-6,
    "adam_betas": [0.9, 0.99], "adam_epsilon": 1e-8,
    "weight_decay": 0.1, "gradient_clip": 0.2,
    "keep_continuations": 4, "proposal_continuations": 8,
    "cf_beta": 4.0, "cf_coefficient": 0.25,
    "scheduler": "constant_with_warmup", "warmup_ratio": 0.0001,
    "reference_selection": "outcome_stratified_v1",
}


def load_config(path: str | Path) -> dict:
    def read(current, ancestors):
        current = current.resolve()
        if current in ancestors:
            raise ValueError("Configuration inheritance cycle")
        result = json.loads(current.read_text())
        parent = result.pop("extends", None)
        if parent:
            result = {**read(current.parent / parent, ancestors | {current}), **result}
        return result
    config = read(Path(path), set())
    for key, value in FIXED.items():
        if config.get(key) != value:
            raise ValueError(f"d1 exclusion protocol requires {key}={value!r}")
    arm = config.get("arm")
    if arm not in ARMS or config.get("alpha") != ARMS[arm]:
        raise ValueError("Condition and exclusion strength do not match")
    if config.get("evaluation_during_training", False):
        raise ValueError("Training evaluations require a separately specified protocol")
    return config
