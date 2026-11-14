"""Manifest and held-out separation checks without network or reward execution."""
import copy
import json
import random

import pytest

from d1_exclusion.data import selected_training_rows, training_manifest
from hindsight_grpo.data import EXPECTED_SPLIT_SIZES


class Split(list):
    def __init__(self, rows, fingerprint):
        super().__init__(rows)
        self._fingerprint = fingerprint

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self]
        return super().__getitem__(key)


def official_shape_data():
    data = {}
    offset = 0
    for name, count in EXPECTED_SPLIT_SIZES.items():
        data[name] = Split([{"task_id": offset + index, "text": f"Task {offset+index}",
                             "test_list": ["assert f(1)==1"], "code": "reference implementation"}
                            for index in range(count)], f"fingerprint-{name}")
        offset += count
    return data


def test_manifest_exact_distinct150_unscreened_first_permutation_and_no_rng_side_effects():
    data = official_shape_data()
    original = copy.deepcopy(data)
    before = random.getstate()
    manifest = training_manifest(data)
    expected = data["train"]["task_id"]
    random.Random(20261108).shuffle(expected)
    assert manifest["permutation"] == expected
    assert manifest["selected_task_ids"] == expected[:150]
    assert len(manifest["selected_task_ids"]) == len(set(manifest["selected_task_ids"])) == 150
    assert len(manifest["permutation"]) == 374
    assert set(manifest["permutation"]) == set(data["train"]["task_id"])
    assert manifest["seed"] == 20261108 and manifest["configuration"] == "full"
    assert manifest["split_sizes"] == EXPECTED_SPLIT_SIZES
    assert manifest["fingerprints"] == {name: f"fingerprint-{name}" for name in data}
    assert len(manifest["training_content_sha256"]) == 64
    assert random.getstate() == before
    assert data == original
    assert training_manifest(data) == manifest
    assert training_manifest(data, seed=20261109)["permutation"] != expected
    for name in ("validation", "test", "prompt"):
        assert not set(manifest["selected_task_ids"]) & set(data[name]["task_id"])


def test_all_arm_manifests_and_rows_are_identical_and_reusable(tmp_path):
    data = official_shape_data()
    manifests = []
    selected_ids = []
    for arm in ("d1_root", "entropy_no_ban", "entropy_soft_ban", "entropy_hard_ban"):
        path = tmp_path / arm / "dataset_manifest.json"
        rows = selected_training_rows(data, path)
        selected_ids.append([row["task_id"] for row in rows])
        manifests.append(json.loads(path.read_text()))
        assert selected_training_rows(data, path) == rows
    assert all(manifest == manifests[0] for manifest in manifests)
    assert all(ids == selected_ids[0] for ids in selected_ids)
    assert selected_ids[0] == manifests[0]["selected_task_ids"]


def test_reference_manifest_checks_content_and_selection_without_requiring_same_cache_fingerprint(tmp_path):
    data = official_shape_data()
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps(training_manifest(data)))
    data["train"]._fingerprint = "different-platform-cache-fingerprint"
    path = tmp_path / "run/dataset_manifest.json"
    rows = selected_training_rows(data, path, reference_path=reference)
    assert len(rows) == 150
    data["train"][0]["text"] = "corrupted dataset contents"
    original = path.read_bytes()
    with pytest.raises(ValueError, match="reference manifest"):
        selected_training_rows(data, path, reference_path=reference)
    assert path.read_bytes() == original


@pytest.mark.parametrize("split", EXPECTED_SPLIT_SIZES)
def test_wrong_split_sizes_rejected(split):
    data = official_shape_data()
    data[split].pop()
    with pytest.raises(ValueError, match="split sizes"):
        training_manifest(data)


def test_unexpected_split_rejected():
    data = official_shape_data()
    data["extra"] = Split([], "extra")
    with pytest.raises(ValueError, match="split sizes"):
        training_manifest(data)


@pytest.mark.parametrize("kind", ("within_train", "within_validation", "across_train_test", "across_validation_prompt"))
def test_duplicate_ids_and_split_leakage_rejected(kind):
    data = official_shape_data()
    if kind == "within_train":
        data["train"][0]["task_id"] = data["train"][1]["task_id"]
    elif kind == "within_validation":
        data["validation"][0]["task_id"] = data["validation"][1]["task_id"]
    elif kind == "across_train_test":
        data["test"][0]["task_id"] = data["train"][0]["task_id"]
    else:
        data["prompt"][0]["task_id"] = data["validation"][0]["task_id"]
    with pytest.raises(ValueError, match="Duplicate|leakage"):
        training_manifest(data)


@pytest.mark.parametrize("split", EXPECTED_SPLIT_SIZES)
def test_fingerprint_drift_rejected_without_overwriting_manifest(tmp_path, split):
    data = official_shape_data()
    path = tmp_path / "dataset_manifest.json"
    selected_training_rows(data, path)
    original = path.read_bytes()
    data[split]._fingerprint = "changed"
    with pytest.raises(ValueError, match="manifest differs"):
        selected_training_rows(data, path)
    assert path.read_bytes() == original


@pytest.mark.parametrize("change", ("training_content", "seed", "count", "training_order"))
def test_content_or_selection_protocol_drift_rejected_without_overwrite(tmp_path, change):
    data = official_shape_data()
    path = tmp_path / "dataset_manifest.json"
    selected_training_rows(data, path)
    original = path.read_bytes()
    kwargs = {}
    if change == "training_content":
        # Content hash catches a corrupted cache even if its fingerprint lies.
        data["train"][1]["text"] = "Changed task"
    elif change == "seed":
        kwargs["seed"] = 7
    elif change == "count":
        kwargs["count"] = 149
    else:
        data["train"].reverse()
    with pytest.raises(ValueError, match="manifest differs"):
        selected_training_rows(data, path, **kwargs)
    assert path.read_bytes() == original


@pytest.mark.parametrize("count", (-1, 0, 375))
def test_invalid_training_task_counts_rejected(count):
    with pytest.raises(ValueError, match="number of distinct"):
        training_manifest(official_shape_data(), count=count)
