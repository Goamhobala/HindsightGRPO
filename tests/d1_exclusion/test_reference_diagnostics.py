"""Outcome summaries preserve conditional evidence and explicit missing metadata."""
import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest

from d1_exclusion.diagnostics import (OUTCOME_STRATA, outcome_strata,
                                      scalar_outcome_metrics, summarize_counterfactual)


def record(group, reward, delta=.5, *, skip=None):
    selected = skip != "no_eligible_implementation_commitment"
    changed = 0 if skip else 4
    valid = selected and changed > 0
    original, target = .6, .6 - .2 * delta
    state = {"root_group_type": group, "reference_reward": reward,
             "successful_roots": {"all_failed": 0, "mixed": 3, "all_successful": 8}[group],
             "reference_index": 2, "reference_selection_seed": 413,
             "mixed_reference_outcome_class": ("successful" if reward else "failed") if group == "mixed" else None,
             "selected_index": 3 if selected else None, "valid": valid, "skip_reason": skip,
             "n_alt": changed if selected else None, "n_keep": 8 - changed if selected else None,
             "old_keep_probability": original if selected else None,
             "target_probability": target if valid else None,
             "target_shift": target - original if valid else None,
             "q_keep": .5 if selected else None,
             "q_alt": .5 + delta if valid else None,
             "q_proposal": .5 if selected else None,
             "delta": delta if valid else None,
             "theoretical_change_probability": .7 if selected else None,
             "selection": {"selector_score": 2.0, "selected_progress": .25} if selected else {}}
    return {"task_id": 10, "advantages": [0.] * 8 if group != "mixed" else [-.375] * 5 + [.625] * 3,
            "cf": state}


def examples():
    return [record("all_failed", 0, .5), record("all_successful", 1, -.5),
            record("mixed", 1, .25), record("mixed", 0, -.25), record("mixed", 1, 0.),
            record("all_failed", 0, skip="no_alternative_observed"),
            record("all_successful", 1, skip="no_eligible_implementation_commitment")]


def test_five_overlapping_strata_and_homogeneous_zero_advantage_credit():
    records = examples()
    before = copy.deepcopy(records)
    report = summarize_counterfactual(records)
    assert report["prompts"] == 7
    assert report["valid_auxiliary_states"] == 5
    assert report["skipped_auxiliary_states"] == 2
    assert report["nonzero_contrasts"] == 4
    assert report["zero_root_advantage_nonzero_contrasts"] == 2
    assert report["outcome_strata"] == outcome_strata(records)
    strata = report["outcome_strata"]
    assert tuple(strata) == OUTCOME_STRATA
    assert strata["successful_references"]["prompts"] == 4
    assert strata["failed_references"]["prompts"] == 3
    assert strata["all_successful"]["prompts"] == 2
    assert strata["all_failed"]["prompts"] == 2
    assert strata["mixed"]["prompts"] == 3
    assert strata["all_successful"]["zero_root_advantage_nonzero_contrasts"] == 1
    assert strata["all_failed"]["zero_root_advantage_nonzero_contrasts"] == 1
    assert strata["mixed"]["zero_root_advantage_nonzero_contrasts"] == 0
    assert strata["mixed"]["mixed_successful_references"] == 2
    assert strata["mixed"]["mixed_failed_references"] == 1
    assert report["unclassified_reference_outcome_prompts"] == 0
    assert report["unclassified_root_group_prompts"] == 0
    assert records == before


def test_reference_outcome_never_changes_recorded_contrast_or_target_sign():
    successful = record("mixed", 1, .5)
    failed = record("mixed", 0, -.5)
    report = summarize_counterfactual([successful, failed])["outcome_strata"]
    assert report["successful_references"]["positive_contrasts"] == 1
    assert report["successful_references"]["mean_delta"] == .5
    assert report["successful_references"]["mean_target_shift"] < 0
    assert report["failed_references"]["negative_contrasts"] == 1
    assert report["failed_references"]["mean_delta"] == -.5
    assert report["failed_references"]["mean_target_shift"] > 0


def test_missing_alternative_excludes_unobserved_values_from_means_and_sign_counts():
    missing = record("all_failed", 0, skip="no_alternative_observed")
    report = summarize_counterfactual([missing])
    assert report["no_alternative_observed"] == 1
    assert report["states_with_alternative"] == 0
    assert report["mean_q_keep"] == .5
    assert report["mean_q_alt"] is None
    assert report["mean_delta"] is None
    assert report["mean_target_probability"] is None
    assert report["mean_target_shift"] is None
    assert report["nonzero_contrasts"] == report["zero_contrasts"] == 0
    assert report["repeated_proposal_samples"] == 8
    assert report["changed_proposal_samples"] == 0
    assert report["observed_change_fraction"] == 0


def test_original_probability_entropy_progress_and_targets_are_summarized():
    records = [record("mixed", 1, .5), record("mixed", 0, -.5)]
    report = summarize_counterfactual(records)
    assert report["mean_original_token_probability"] == .6
    assert report["mean_selected_entropy"] == 2.0
    assert report["mean_selected_progress"] == .25
    assert report["mean_target_probability"] == pytest.approx(.6)
    assert report["mean_target_shift"] == pytest.approx(0)
    assert report["mean_absolute_target_shift"] == pytest.approx(.1)
    assert report["mean_changed_proposals"] == report["mean_repeated_proposals"] == 4


def test_legacy_metadata_stays_unclassified_even_when_root_rewards_are_present():
    legacy = record("all_successful", 1, -.5)
    legacy["rewards"] = [{"reward": 1}] * 8
    for key in ("root_group_type", "reference_reward", "successful_roots", "mixed_reference_outcome_class"):
        legacy["cf"].pop(key)
    report = summarize_counterfactual([legacy])
    assert report["prompts"] == report["nonzero_contrasts"] == 1
    assert report["unclassified_reference_outcome_prompts"] == 1
    assert report["unclassified_root_group_prompts"] == 1
    assert report["zero_root_advantage_nonzero_contrasts"] == 0
    assert all(part["prompts"] == 0 for part in report["outcome_strata"].values())


def test_logged_logprob_target_fallback_does_not_require_materialized_probabilities():
    value = record("all_failed", 0, .5)
    value["cf"].pop("old_keep_probability")
    value["cf"].pop("target_probability")
    value["cf"].pop("target_shift")
    value["cf"].update(old_keep_logprob=math.log(.6), target_logprob=math.log(.5))
    report = summarize_counterfactual([value])
    assert report["mean_original_token_probability"] == pytest.approx(.6)
    assert report["mean_target_probability"] == pytest.approx(.5)
    assert report["mean_target_shift"] == pytest.approx(-.1)


def test_scalar_metrics_keep_counts_for_empty_strata_and_omit_unobserved_means():
    records = [record("all_failed", 0, skip="no_alternative_observed")]
    metrics = scalar_outcome_metrics(records)
    assert metrics["cf/all_failed/prompts"] == 1
    assert metrics["cf/failed_references/no_alternative_observed"] == 1
    assert metrics["cf/successful_references/prompts"] == 0
    assert metrics["cf/all_successful/nonzero_contrasts"] == 0
    assert "cf/all_failed/mean_q_alt" not in metrics
    assert "cf/all_failed/mean_delta" not in metrics
    assert all(isinstance(value, (int, float)) and math.isfinite(value) for value in metrics.values())
    assert scalar_outcome_metrics(records, prefix="collection/")["collection/all_failed/prompts"] == 1


def test_legacy_logprob_shift_preserves_changes_below_probability_rounding():
    value = record("all_successful", 1, .5)
    value["cf"].pop("old_keep_probability")
    value["cf"].pop("target_probability")
    value["cf"].pop("target_shift")
    value["cf"].update(old_keep_logprob=-1e-50, target_logprob=-2e-50)
    report = summarize_counterfactual([value])
    assert report["mean_original_token_probability"] == report["mean_target_probability"] == 1.
    assert report["mean_target_shift"] == -1e-50
    assert report["mean_absolute_target_shift"] == 1e-50


def test_root_only_records_and_empty_records_have_no_counterfactual_outcomes():
    for records in ([], [{"task_id": 3, "cf": None}]):
        report = summarize_counterfactual(records)
        assert report["prompts"] == 0
        assert report["mean_delta"] is None
        assert all(part["prompts"] == 0 for part in report["outcome_strata"].values())
        assert all(value == 0 for value in scalar_outcome_metrics(records).values())


def test_comparison_exposes_strata_for_saved_and_common_completed_prefix(tmp_path):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("reference_diagnostics_compare", root / "scripts/d1_exclusion/compare.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for arm in module.ARMS:
        directory = tmp_path / arm
        (directory / "collections").mkdir(parents=True)
        (directory / "progress.json").write_text(json.dumps({"cursor": 4, "applied_prompts": 8}))
        (directory / "collections/0000.json").write_text(json.dumps({"records": examples()[:4]}))
        (directory / "collections/0001.json").write_text(json.dumps({"records": examples()[4:]}))
    report = module.summarize_group(tmp_path)
    assert report["largest_common_completed_prompts"] == 4
    summary = report["arms"]["entropy_hard_ban"]
    saved = summary["counterfactual_saved_collections"]["outcome_strata"]
    common = summary["counterfactual_common_completed_prefix"]["outcome_strata"]
    assert saved["all_failed"]["no_alternative_observed"] == 1
    assert common["all_failed"]["no_alternative_observed"] == 0
    assert common["all_successful"]["nonzero_contrasts"] == 1
    assert common["mixed"]["successful_references"] == 1
