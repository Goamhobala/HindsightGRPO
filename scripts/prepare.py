"""Run `uv run python scripts/prepare.py` on the HPC to cache pinned public inputs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from huggingface_hub import snapshot_download

from hindsight_grpo.checkpoint import atomic_json
from hindsight_grpo.config import load_config
from hindsight_grpo.data import load_mbpp
from hindsight_grpo.model import MODEL_ID, MODEL_REVISION, UPSTREAM_COMMIT


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--runtime-dir", default="runtime")
    parser.add_argument("--metadata-only", action="store_true",
                        help="Inspect small pinned model metadata without downloading weights")
    args = parser.parse_args()
    config = load_config(args.config)
    if config["upstream_commit"] != UPSTREAM_COMMIT:
        raise RuntimeError("Upstream commit differs from the inspected implementation")
    runtime = Path(args.runtime_dir)
    metadata = runtime / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    hf_cache = Path(os.environ.get("HF_HUB_CACHE", str(runtime / "huggingface" / "hub")))
    model_path = Path(snapshot_download(
        repo_id=MODEL_ID, revision=MODEL_REVISION, cache_dir=str(hf_cache),
        allow_patterns=(["*.json", "*.py", "*.jinja", "merges.txt", "vocab.json"]
                        if args.metadata_only else None),
        ignore_patterns=["training_args.bin"],
    ))
    model_config = json.loads((model_path / "config.json").read_text())
    tokenizer_config = json.loads((model_path / "tokenizer_config.json").read_text())
    if model_config.get("model_type") != "a2d-qwen2":
        raise RuntimeError("Downloaded model is not the specified MDLM architecture")
    if tokenizer_config.get("mask_token") != "<|mask|>":
        raise RuntimeError("Downloaded tokenizer has the wrong mask token")
    files = {path.name: sha256(path) for path in model_path.iterdir()
             if path.is_file() and path.suffix in {".py", ".json", ".jinja"}}
    upstream_dir = runtime / "upstream" / "dllm"
    if not upstream_dir.exists():
        upstream_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", "https://github.com/ZHZisZZ/dllm.git",
                        str(upstream_dir)], check=True)
    subprocess.run(["git", "-C", str(upstream_dir), "checkout", "--detach", UPSTREAM_COMMIT], check=True)
    actual_commit = subprocess.check_output(["git", "-C", str(upstream_dir), "rev-parse", "HEAD"], text=True).strip()
    if actual_commit != UPSTREAM_COMMIT:
        raise RuntimeError("Upstream checkout pin mismatch")
    data = load_mbpp(
        cache_dir=Path(os.environ.get("HF_DATASETS_CACHE", str(runtime / "datasets"))),
        manifest_path=metadata / "dataset_manifest.json",
    )
    manifest = {
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "model_snapshot": str(model_path), "model_metadata_sha256": files,
        "metadata_only": args.metadata_only,
        "model_weights_present": (model_path / "model.safetensors").is_file(),
        "upstream_url": "https://github.com/ZHZisZZ/dllm",
        "upstream_commit": actual_commit, "upstream_checkout": str(upstream_dir),
        "dataset_id": config["dataset_id"], "dataset_revision": config["dataset_revision"],
        "dataset_manifest": str(metadata / "dataset_manifest.json"),
        "split_sizes": {split: len(rows) for split, rows in data.items()},
        "likelihood_convention": config["likelihood_convention"],
    }
    atomic_json(metadata / "inputs_manifest.json", manifest)
    print(json.dumps({"manifest": str(metadata / "inputs_manifest.json"),
                      "model_revision": MODEL_REVISION, "split_sizes": manifest["split_sizes"]}, indent=2))


if __name__ == "__main__":
    main()
