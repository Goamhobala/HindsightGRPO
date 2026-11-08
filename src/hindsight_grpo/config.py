"""One immutable scientific protocol, with configurable logging/evaluation metadata."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ARMS = ("root", "entropy_cf", "r_cf", "gradient_ce_cf")
FIXED = {
    "generation_length": 256, "denoising_steps": 256, "block_size": 64,
    "temperature": .8, "cfg_scale": 0, "lora_rank": 32,
    "learning_rate": 2e-5, "root_trajectories": 8, "prompts_per_update": 4,
    "keep_continuations": 4, "alternative_continuations": 8,
    "cf_beta": 4., "cf_coefficient": .25, "epochs": 3,
    "model_id": "dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1",
    "model_revision": "a284e895a6248256baf2f60502e54aba61b24c1a",
    "upstream_commit": "ca176752fbceec49c6b4777a2c18ae88e4eb10ed",
    "dataset_id": "google-research-datasets/mbpp",
    "dataset_revision": "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
    "lora_alpha": 64, "lora_dropout": 0.,
    "likelihood_convention": "upstream_full_completion_mask_seeded_prompt_mask_untempered_token_mean",
    "epoch_tail": "flush_two_prompts_no_padding",
}


def load_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text())
    for key, expected in FIXED.items():
        if config.get(key) != expected:
            raise ValueError(f"Scientific protocol requires {key}={expected!r}")
    if config["evaluation_samples"] < 1 or config["validation_every_updates"] < 1:
        raise ValueError("Evaluation interval and budget must be positive")
    return config


def stable_seed(*parts) -> int:
    """Independent reproducible streams, including common random numbers across arms."""
    data = json.dumps(parts, separators=(",", ":"), ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big") % (2**63 - 1)


def config_digest(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
