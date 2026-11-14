"""Configuration and operational checks without GPU work, SLURM calls or code execution."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

from d1_exclusion.config import ARMS, load_config
from d1_exclusion.evaluation import checkpoint_snapshot, evaluation_lock, evaluation_rows

REPOSITORY = Path(__file__).resolve().parents[2]


def test_four_conditions_share_exactly_one_scientific_configuration():
    names = ("root", "no_ban", "soft_ban", "hard_ban")
    configurations = [load_config(REPOSITORY / "configs/d1_exclusion" / f"{name}.json") for name in names]
    assert {item["arm"]: item["alpha"] for item in configurations} == ARMS
    shared = [{key: value for key, value in item.items() if key not in {"arm", "alpha"}}
              for item in configurations]
    assert all(item == shared[0] for item in shared)
    assert shared[0]["training_tasks"] == 150
    assert shared[0]["prompts_per_collection"] == 4
    assert shared[0]["inner_iterations"] == 12
    assert shared[0]["scheduler"] == "constant_with_warmup"
    assert shared[0]["warmup_ratio"] == 0.0001


def test_shell_syntax_and_submission_dry_run_do_not_need_cluster(tmp_path):
    scripts = sorted((REPOSITORY / "scripts/d1_exclusion").glob("*.sh"))
    scripts += sorted((REPOSITORY / "slurm/d1_exclusion").glob("*.sbatch"))
    for path in scripts:
        subprocess.run(["bash", "-n", str(path)], check=True, capture_output=True, text=True)
    script = REPOSITORY / "scripts/d1_exclusion/submit_all.sh"
    completed = subprocess.run(["bash", str(script), "--dry-run", "--group", "test-group"],
                               cwd=tmp_path, check=True, capture_output=True, text=True)
    assert len(completed.stdout.splitlines()) == 4
    assert all(arm in completed.stdout for arm in ARMS)
    assert list(tmp_path.iterdir()) == []


def test_submission_rejects_path_traversal_before_touching_environment(tmp_path):
    script = REPOSITORY / "scripts/d1_exclusion/submit_all.sh"
    completed = subprocess.run(["bash", str(script), "--dry-run", "--group", "../../old-results"],
                               cwd=tmp_path, capture_output=True, text=True)
    assert completed.returncode == 2
    assert list(tmp_path.iterdir()) == []


def test_batch_preserves_resource_requests_path_and_has_no_auto_successor():
    text = (REPOSITORY / "slurm/d1_exclusion/train.sbatch").read_text()
    assert "cd '../../scratch/yhxjin001/HindsightGRPO'" in text
    for setting in ("--partition=l40s", "--account=l40sfree", "--qos=l40sfree",
                    "--gres=gpu:1", "--time=24:00:00"):
        assert setting in text
    assert "sbatch " not in text and "srun " not in text and "salloc " not in text
    assert "trap forward_stop USR1 TERM INT" in text
    assert "--resume" in text


def test_checkpoint_snapshot_hash_identifies_loaded_copy_after_source_replacement(tmp_path):
    import hashlib
    source = tmp_path / "latest.pt"
    source.write_bytes(b"original frozen checkpoint")
    with checkpoint_snapshot(source, tmp_path) as (snapshot, identity):
        replacement = tmp_path / "new.pt"
        replacement.write_bytes(b"new current policy")
        replacement.replace(source)
        assert snapshot.read_bytes() == b"original frozen checkpoint"
        assert identity == hashlib.sha256(snapshot.read_bytes()).hexdigest()
    assert not snapshot.exists()


def test_evaluation_cache_excludes_concurrent_writer(tmp_path):
    with evaluation_lock(tmp_path, "sha256-checkpoint", "test"):
        with pytest.raises(RuntimeError, match="already running"):
            with evaluation_lock(tmp_path, "sha256-checkpoint", "test"):
                pass
        with evaluation_lock(tmp_path, "sha256-checkpoint", "validation"):
            pass
    with evaluation_lock(tmp_path, "sha256-checkpoint", "test"):
        pass


def test_evaluation_subset_is_distinct_reproducible_and_does_not_mutate_data():
    rows = [{"task_id": task} for task in range(500)]
    assert evaluation_rows(rows, 20261108) == rows
    selected = evaluation_rows(rows, 20261108, 100)
    assert selected == evaluation_rows(rows, 20261108, 100)
    assert len({row["task_id"] for row in selected}) == 100
    assert rows == [{"task_id": task} for task in range(500)]
    with pytest.raises(ValueError):
        evaluation_rows(rows, 20261108, 501)


def comparison_module():
    spec = importlib.util.spec_from_file_location("d1_compare", REPOSITORY / "scripts/d1_exclusion/compare.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_common_exposure_keeps_partial_collection_separate(tmp_path):
    cursors = [150, 148, 144, 148]
    for arm, cursor in zip(ARMS, cursors):
        directory = tmp_path / arm
        directory.mkdir()
        (directory / "progress.json").write_text(json.dumps({
            "cursor": cursor, "applied_prompts": min(150, cursor + 4),
            "collected_prompts": min(150, cursor + 4), "inner_iteration": 5,
            "optimizer_steps": cursor // 4 * 12 + 5,
        }))
    report = comparison_module().summarize_group(tmp_path)
    assert report["largest_common_completed_prompts"] == 144
    assert report["largest_common_applied_prompts"] == 148
    assert report["planned_optimizer_steps"] == 456


def test_missing_arm_means_no_common_exposure_and_output_never_overwrites(tmp_path):
    (tmp_path / "d1_root").mkdir()
    (tmp_path / "d1_root/progress.json").write_text('{"cursor":150}')
    module = comparison_module()
    assert module.summarize_group(tmp_path)["largest_common_completed_prompts"] == 0
    output = tmp_path / "comparison.json"
    output.write_text("previous result")
    with pytest.raises(FileExistsError):
        module.main(["--run-root", str(tmp_path), "--output", str(output)])
    assert output.read_text() == "previous result"


def test_conditional_comparison_distinguishes_missing_alternatives_and_no_selection():
    records = [
        {"cf": {"selected_index": None, "valid": False,
                "skip_reason": "no_eligible_implementation_commitment"}},
        {"cf": {"selected_index": 2, "n_alt": 0, "valid": False,
                "skip_reason": "no_alternative_observed", "q_keep": 0.5,
                "q_alt": None, "q_proposal": 0.5, "delta": None,
                "theoretical_change_probability": 0.1}},
        {"cf": {"selected_index": 2, "n_alt": 4, "valid": True,
                "q_keep": 0.5, "q_alt": 1., "q_proposal": 0.75, "delta": 0.5,
                "theoretical_change_probability": 0.5}},
    ]
    report = comparison_module().summarize_counterfactual(records)
    assert report["selected_states"] == 2
    assert report["states_with_alternative"] == report["valid_auxiliary_states"] == 1
    assert report["proposal_samples"] == 16
    assert report["changed_proposal_samples"] == 4
    assert report["repeated_proposal_samples"] == 12
    assert report["observed_change_fraction"] == 0.25
    assert report["mean_q_alt"] == 1.
    assert report["mean_delta"] == 0.5
    assert report["skip_reasons"]["no_alternative_observed"] == 1


def test_evaluation_orchestration_saves_and_checks_checkpoint_exposure(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import hindsight_grpo.data
    import hindsight_grpo.rewards
    import hindsight_grpo.sampler
    import hindsight_grpo.tracking
    from d1_exclusion import evaluation as module

    class Split(list):
        _fingerprint = "fixed-dataset-fingerprint"

    calls = []

    class Tracker:
        def __init__(self, directory, arm, config, **kwargs):
            calls.append((directory, arm))

        def summary(self, *args):
            pass

        def finish(self, code):
            assert code == 0

    monkeypatch.setattr(hindsight_grpo.data, "load_mbpp", lambda: {
        "test": Split([{"task_id": task} for task in range(3)])})
    monkeypatch.setattr(hindsight_grpo.rewards, "RewardRunner", lambda *args, **kwargs:
                        SimpleNamespace(check_available=lambda: None))
    monkeypatch.setattr(hindsight_grpo.sampler, "Sampler", lambda *args, **kwargs: object())
    monkeypatch.setattr(hindsight_grpo.tracking, "Tracker", Tracker)
    monkeypatch.setattr(module, "evaluate", lambda *args, **kwargs: {"problems": 3})
    config = {"arm": "d1_root", "seed": 20261108, "evaluation_samples": 8,
              "generation_length": 256, "block_size": 64, "temperature": .8,
              "denoising_steps": 256, "reward_timeout_seconds": 5, "reward_memory_mb": 512}
    args = SimpleNamespace(run_dir=str(tmp_path), checkpoint="final.pt", split="test", task_limit=None)
    progress = {"cursor": 148, "optimizer_steps": 444}
    module._evaluate_loaded(args, config, object(), object(), "identity", progress)
    manifest = json.loads((tmp_path / "evaluation/sha256-identity/test/manifest.json").read_text())
    assert manifest["checkpoint_id"] == "identity"
    assert manifest["training_progress"] == progress
    assert manifest["task_ids"] == [0, 1, 2]
    assert manifest["samples_per_problem"] == 8
    assert calls[0][0] == tmp_path / "evaluation/sha256-identity/test/tracking"
    args.checkpoint = "copied-final.pt"
    module._evaluate_loaded(args, config, object(), object(), "identity", progress)
    with pytest.raises(ValueError, match="provenance"):
        module._evaluate_loaded(args, config, object(), object(), "identity", {"cursor": 150})


def test_evaluation_cli_refuses_login_node_before_model_loading(monkeypatch, tmp_path):
    from d1_exclusion import evaluation as module
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(SystemExit) as error:
        module.main(["--config", "missing.json", "--run-dir", str(tmp_path),
                     "--checkpoint", "pretrained"])
    assert error.value.code == 2
    assert list(tmp_path.iterdir()) == []
