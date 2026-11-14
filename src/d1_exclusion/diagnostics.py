"""Outcome-stratified counterfactual summaries shared by training and reporting.

These are descriptive selected-state diagnostics. A reference outcome is never
used to change a contrast, target, loss weight, or conditional reward estimate.
Records without the amendment's outcome metadata remain explicitly unclassified.
"""
from __future__ import annotations

import math


ROOT_GROUP_TYPES = ("all_successful", "all_failed", "mixed")
OUTCOME_STRATA = ("successful_references", "failed_references", *ROOT_GROUP_TYPES)


def _mean(values):
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _probability(state, key, log_key):
    value = state.get(key)
    if value is not None:
        return float(value)
    log_value = state.get(log_key)
    return math.exp(float(log_value)) if log_value is not None else None


def _target_shift(state):
    if state.get("target_shift") is not None:
        return float(state["target_shift"])
    old_logp, target_logp = state.get("old_keep_logprob"), state.get("target_logprob")
    if old_logp is not None and target_logp is not None:
        return math.exp(float(old_logp)) * math.expm1(float(target_logp) - float(old_logp))
    target = _probability(state, "target_probability", "target_logprob")
    old = _probability(state, "old_keep_probability", "old_keep_logprob")
    return target - old if target is not None and old is not None else None


def _summary(states):
    selected = [state for state in states if state.get("selected_index") is not None]
    observed = [state for state in selected if (state.get("n_alt") or 0) > 0]
    contrasts = [state for state in observed if state.get("valid") and state.get("delta") is not None]
    # New records always preserve both counts and all rewards. Legacy records
    # may preserve just n_alt; the fixed eight-proposal allocation identifies
    # their repeat count, while absent proposal evidence is not invented.
    sampled = [state for state in selected if state.get("n_alt") is not None]
    changes = sum(int(state["n_alt"]) for state in sampled)
    repeats = sum(int(state["n_keep"]) if state.get("n_keep") is not None
                  else 8 - int(state["n_alt"]) for state in sampled)
    proposals = changes + repeats
    skips = {}
    for state in states:
        reason = state.get("skip_reason")
        if reason:
            skips[reason] = skips.get(reason, 0) + 1
    nonzero = [state for state in contrasts if float(state["delta"]) != 0.]
    shifts = [_target_shift(state) for state in contrasts]
    return {
        "prompts": len(states), "selected_states": len(selected),
        "valid_auxiliary_states": sum(bool(state.get("valid")) for state in states),
        "skipped_auxiliary_states": sum(not bool(state.get("valid")) for state in states),
        "states_with_alternative": len(observed), "skip_reasons": skips,
        "no_eligible_states": skips.get("no_eligible_implementation_commitment", 0),
        "no_alternative_observed": skips.get("no_alternative_observed", 0),
        "proposal_samples": proposals, "changed_proposal_samples": changes,
        "repeated_proposal_samples": repeats,
        "observed_change_fraction": changes / proposals if proposals else None,
        "mean_changed_proposals": _mean(state["n_alt"] for state in sampled),
        "mean_repeated_proposals": repeats / len(sampled) if sampled else None,
        "mean_theoretical_change_probability": _mean(state.get("theoretical_change_probability") for state in sampled),
        "mean_original_token_probability": _mean(_probability(state, "old_keep_probability", "old_keep_logprob") for state in selected),
        "mean_successful_roots": _mean(state.get("successful_roots") for state in states),
        "mean_selected_entropy": _mean(state.get("selection", {}).get("selector_score") for state in selected),
        "mean_selected_progress": _mean(state.get("selection", {}).get("selected_progress") for state in selected),
        "mean_q_keep": _mean(state.get("q_keep") for state in sampled),
        "mean_q_alt": _mean(state.get("q_alt") for state in observed),
        "mean_q_proposal": _mean(state.get("q_proposal") for state in sampled),
        "mean_delta": _mean(state["delta"] for state in contrasts),
        "positive_contrasts": sum(float(state["delta"]) > 0 for state in contrasts),
        "negative_contrasts": sum(float(state["delta"]) < 0 for state in contrasts),
        "zero_contrasts": sum(float(state["delta"]) == 0 for state in contrasts),
        "nonzero_contrasts": len(nonzero),
        # All eight rewards agree in these groups, hence the common d1 root
        # advantages are zero. Conditional resampling can still expose credit.
        "zero_root_advantage_nonzero_contrasts": sum(state.get("root_group_type") in
            {"all_successful", "all_failed"} for state in nonzero),
        "mean_target_probability": _mean(_probability(state, "target_probability", "target_logprob") for state in contrasts),
        "mean_target_shift": _mean(shifts),
        "mean_absolute_target_shift": _mean(abs(value) for value in shifts if value is not None),
        "successful_references": sum(state.get("reference_reward") == 1 for state in states),
        "failed_references": sum(state.get("reference_reward") == 0 for state in states),
        "mixed_successful_references": sum(state.get("root_group_type") == "mixed" and
            state.get("reference_reward") == 1 for state in states),
        "mixed_failed_references": sum(state.get("root_group_type") == "mixed" and
            state.get("reference_reward") == 0 for state in states),
    }


def outcome_strata(records):
    """Summarize five overlapping strata, without imputing missing outcomes.

    Reference-success and root-group strata overlap: each classified prompt
    belongs to one reference stratum and one group stratum, not five partitions.
    """
    states = [record["cf"] for record in records if record.get("cf") is not None]
    members = {
        "successful_references": [s for s in states if s.get("reference_reward") == 1],
        "failed_references": [s for s in states if s.get("reference_reward") == 0],
        **{kind: [s for s in states if s.get("root_group_type") == kind]
           for kind in ROOT_GROUP_TYPES},
    }
    return {name: _summary(members[name]) for name in OUTCOME_STRATA}


def summarize_counterfactual(records):
    """Describe conditional discovery overall and by recorded reference outcomes."""
    records = list(records)
    states = [record["cf"] for record in records if record.get("cf") is not None]
    return {
        **_summary(states),
        "outcome_strata": outcome_strata(records),
        "unclassified_reference_outcome_prompts": sum(s.get("reference_reward") not in (0, 1) for s in states),
        "unclassified_root_group_prompts": sum(s.get("root_group_type") not in ROOT_GROUP_TYPES for s in states),
        "outcome_strata_interpretation": (
            "Reference and root-group strata overlap. Missing legacy outcome metadata is not inferred. "
            "Nonzero conditional contrasts in homogeneous root groups have zero group-centred root "
            "advantages, but remain selected-state evidence rather than prompt-level policy gradients."
        ),
    }


def scalar_outcome_metrics(records, prefix="cf"):
    """Flatten numeric summaries for one frozen collection's training telemetry.

    Means with no observations are omitted; absence is never logged as a measured
    zero. Counts remain zero for empty strata. Per-step repetitions describe the
    same collection and must not be summed across its twelve inner iterations.
    """
    report = summarize_counterfactual(records)
    prefix = prefix.rstrip("/")
    result = {}
    for name, values in report["outcome_strata"].items():
        for key, value in values.items():
            if isinstance(value, (int, float)):
                result[f"{prefix}/{name}/{key}"] = value
    for key in ("unclassified_reference_outcome_prompts", "unclassified_root_group_prompts",
                "zero_root_advantage_nonzero_contrasts"):
        result[f"{prefix}/{key}"] = report[key]
    return result


__all__ = ["OUTCOME_STRATA", "outcome_strata", "summarize_counterfactual", "scalar_outcome_metrics"]
