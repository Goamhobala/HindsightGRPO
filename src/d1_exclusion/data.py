"""Unscreened deterministic task selection; reuse official MBPP preprocessing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

from hindsight_grpo.checkpoint import atomic_json
from hindsight_grpo.data import DATASET_ID, DATASET_REVISION, EXPECTED_SPLIT_SIZES


def training_manifest(data, seed=20261108, count=150):
    sizes = {key: len(value) for key, value in data.items()}
    if sizes != EXPECTED_SPLIT_SIZES:
        raise ValueError(f"Official MBPP full split sizes differ: {sizes}")
    id_sets = {key: set(map(int, value["task_id"])) for key, value in data.items()}
    for key, ids in id_sets.items():
        if len(ids) != sizes[key] or any(ids & other for name, other in id_sets.items() if name != key):
            raise ValueError("Duplicate task IDs or leakage between MBPP splits")
    permutation = list(map(int, data["train"]["task_id"]))
    random.Random(seed).shuffle(permutation)
    if not 0 < count <= len(permutation):
        raise ValueError("Invalid number of distinct training tasks")
    return {"dataset_id": DATASET_ID, "dataset_revision": DATASET_REVISION,
            "configuration": "full", "seed": seed, "split_sizes": sizes,
            "fingerprints": {name: split._fingerprint for name, split in data.items()},
            "training_content_sha256": hashlib.sha256(json.dumps(
                list(data["train"]), sort_keys=True).encode()).hexdigest(),
            "permutation": permutation, "selected_task_ids": permutation[:count]}


def selected_training_rows(data, manifest_path, seed=20261108, count=150, reference_path=None):
    manifest = training_manifest(data, seed, count)
    if reference_path is not None:
        reference = json.loads(Path(reference_path).read_text())
        for key in ("dataset_id", "dataset_revision", "configuration", "seed", "split_sizes",
                    "training_content_sha256", "permutation", "selected_task_ids"):
            if manifest[key] != reference.get(key):
                raise ValueError(f"Dataset selection differs from the verified reference manifest: {key}")
    path = Path(manifest_path)
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Saved training manifest differs from the dataset or protocol")
    atomic_json(path, manifest)
    rows = {int(row["task_id"]): row for row in data["train"]}
    return [rows[task] for task in manifest["selected_task_ids"]]
