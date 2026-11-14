"""Outcome-stratified references and one shared commitment-entropy selector."""
from __future__ import annotations

import math
import random
import time

from hindsight_grpo.selectors import Selection, eligible_candidates


REFERENCE_SELECTION_METHOD = "outcome_stratified_v1"


def select_reference(rewards, seed: int) -> dict:
    """Choose one reference, balancing outcome classes only in mixed groups.

    The reward is used solely to form the two classes. All-success and all-fail
    groups use their only available class; mixed groups first choose successful
    versus failed with equal probability, then choose uniformly within that
    class. The local generator and saved seed make the decision reproducible
    without changing global training randomness. No terminal reward is passed
    to the conditional reward estimator by this function.
    """
    try:
        outcomes = [float(reward) for reward in rewards]
    except (TypeError, ValueError) as error:
        raise ValueError("Reference selection requires eight binary root rewards") from error
    if len(outcomes) != 8 or any(reward not in (0.0, 1.0) for reward in outcomes):
        raise ValueError("Reference selection requires eight binary root rewards")
    successful = [index for index, reward in enumerate(outcomes) if reward == 1.0]
    failed = [index for index, reward in enumerate(outcomes) if reward == 0.0]
    generator = random.Random(int(seed))
    mixed_class = None
    if not successful:
        group_type, candidates = "all_failed", failed
    elif not failed:
        group_type, candidates = "all_successful", successful
    else:
        group_type = "mixed"
        mixed_class = "successful" if generator.randrange(2) else "failed"
        candidates = successful if mixed_class == "successful" else failed
    index = generator.choice(candidates)
    return {
        "reference_index": index,
        "reference_reward": int(outcomes[index]),
        "root_group_type": group_type,
        "successful_roots": len(successful),
        "mixed_reference_outcome_class": mixed_class,
        "reference_selection_seed": int(seed),
        "reference_selection_method": REFERENCE_SELECTION_METHOD,
    }


def reference_index(rewards, seed: int) -> int:
    """Index-only convenience for the same outcome-stratified selection rule."""
    return select_reference(rewards, seed)["reference_index"]


def select_entropy(trajectory, eligible) -> Selection:
    """Maximise recorded untempered commitment entropy at eligible progress < .5.

    The existing sampler records entropy at the actual committed position before
    inserting that token. Ties select the earlier historical commitment.
    """
    started = time.perf_counter()
    candidates, _ = eligible_candidates(trajectory, eligible)
    metrics = {"eligible_candidates": len(candidates), "states_scored": len(candidates),
               "selector": "commitment_time_untempered_entropy"}
    if not candidates:
        metrics.update(skip_reason="no_eligible_implementation_commitment",
                       scoring_seconds=time.perf_counter() - started)
        return Selection(None, None, metrics)
    if not all(math.isfinite(trajectory.commitments[index].entropy) for index, _ in candidates):
        raise FloatingPointError("Nonfinite recorded commitment-time entropy")
    index, progress = max(candidates, key=lambda item: trajectory.commitments[item[0]].entropy)
    commitment = trajectory.commitments[index]
    metrics.update(selected_index=index, selected_position=commitment.position,
                   selected_progress=progress, selector_score=commitment.entropy,
                   candidate_scores=[{"index": i, "progress": u,
                                      "position": trajectory.commitments[i].position,
                                      "score": trajectory.commitments[i].entropy}
                                     for i, u in candidates],
                   scoring_seconds=time.perf_counter() - started)
    return Selection(index, commitment.entropy, metrics)


__all__ = ["REFERENCE_SELECTION_METHOD", "select_reference", "reference_index", "select_entropy"]
