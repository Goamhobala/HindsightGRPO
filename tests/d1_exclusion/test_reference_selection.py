"""Deterministic checks for one outcome-stratified reference per prompt."""
from collections import Counter
import json
import random

import pytest
import torch

from d1_exclusion.selector import (REFERENCE_SELECTION_METHOD, reference_index,
                                   select_reference)


@pytest.mark.parametrize("reward,group_type", [(0, "all_failed"), (1, "all_successful")])
def test_homogeneous_groups_choose_only_available_outcome_uniformly(reward, group_type):
    counts = Counter()
    for seed in range(8000):
        result = select_reference([reward] * 8, seed)
        index = result["reference_index"]
        assert isinstance(index, int) and 0 <= index < 8
        assert result["reference_reward"] == reward
        assert result["successful_roots"] == 8 * reward
        assert result["root_group_type"] == group_type
        assert result["mixed_reference_outcome_class"] is None
        counts[index] += 1
    assert set(counts) == set(range(8))
    assert all(abs(count - 1000) < 120 for count in counts.values())


@pytest.mark.parametrize("successful_indices", [(0,), (0, 3), (0, 1, 2, 3, 4, 5, 6)])
def test_mixed_groups_balance_classes_then_sample_uniformly_within_class(successful_indices):
    rewards = [int(index in successful_indices) for index in range(8)]
    counts = Counter()
    samples = 20000
    for seed in range(samples):
        result = select_reference(rewards, seed)
        index = result["reference_index"]
        assert isinstance(index, int) and 0 <= index < 8
        assert result["reference_reward"] == rewards[index]
        assert result["successful_roots"] == len(successful_indices)
        assert result["root_group_type"] == "mixed"
        assert result["mixed_reference_outcome_class"] == ("successful" if rewards[index] else "failed")
        counts[index] += 1
    success_count = sum(counts[index] for index in successful_indices)
    assert abs(success_count / samples - .5) < .015
    # Condition on the chosen class, rather than treating all eight as uniform.
    for outcome, class_count in [(1, success_count), (0, samples - success_count)]:
        members = [index for index, reward in enumerate(rewards) if reward == outcome]
        expected_probability = 1 / len(members)
        for index in members:
            assert abs(counts[index] / class_count - expected_probability) < .025


def test_reference_choice_is_reproducible_json_friendly_and_does_not_mutate_inputs_or_rng():
    rewards = [0, 1, 0, 0, 0, 1, 0, 0]
    before = random.getstate()
    result = select_reference(rewards, 20261108)
    assert result == select_reference(rewards, result["reference_selection_seed"])
    assert reference_index(rewards, 20261108) == result["reference_index"]
    assert result["reference_selection_method"] == REFERENCE_SELECTION_METHOD == "outcome_stratified_v1"
    assert random.getstate() == before
    assert rewards == [0, 1, 0, 0, 0, 1, 0, 0]
    assert json.loads(json.dumps(result)) == result
    assert select_reference(torch.tensor(rewards), 20261108) == result


def test_mixed_draw_matches_explicit_class_then_member_algorithm():
    rewards = [0, 1, 0, 1, 0, 0, 0, 0]
    for seed in range(100):
        generator = random.Random(seed)
        selected_outcome = generator.randrange(2)
        candidates = [index for index, reward in enumerate(rewards) if reward == selected_outcome]
        expected = generator.choice(candidates)
        assert reference_index(rewards, seed) == expected


@pytest.mark.parametrize("rewards", [8, [], [0] * 7, [1] * 9,
                                      [0] * 7 + [.5], [0] * 7 + [2],
                                      [0] * 7 + [float("nan")],
                                      [0] * 7 + [float("inf")]])
def test_invalid_reward_groups_are_rejected(rewards):
    with pytest.raises(ValueError, match="eight binary"):
        select_reference(rewards, 7)
