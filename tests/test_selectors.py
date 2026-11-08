"""Run selector state, gradient and finite-difference checks with uv run pytest."""

import pytest
import torch
import torch.nn.functional as F

from hindsight_grpo.sampler import Commitment, Sampler, Trajectory
from hindsight_grpo.selectors import eligible_candidates, reconstruction_gradient, select_decision
from test_sampler import TinyMDLM, TOKENIZER


def test_candidate_eligibility_uses_commitment_order_not_textual_order():
    trajectory = Trajectory([1], [1, 2, 3, 4, 5], [
        Commitment(p, t, 0, float(i), -1., .3, 1)
        for i, (p, t) in enumerate([(4, 5), (3, 4), (2, 3), (1, 2)])
    ], 0, 4, .8, 1)
    candidates, _ = eligible_candidates(trajectory, [True] * 4)
    assert candidates == [(0, 0.), (1, .25)]
    model = TinyMDLM()
    selected = select_decision(model, trajectory, [True] * 4, 0, "entropy_cf")
    assert selected.index == 1
    assert model.seen_inputs == []  # Historical commitment entropies are reused.


def test_r_uses_completed_context_with_one_historical_token_remasked():
    model = TinyMDLM()
    trajectory = Sampler(model, TOKENIZER, 8, 4).sample([1, 2], 8)
    model.seen_inputs.clear()
    selected = select_decision(model, trajectory, [True] * 8, 0, "r_cf")
    assert selected.index is not None
    candidates, _ = eligible_candidates(trajectory, [True] * 8)
    for ids, (index, _) in zip(model.seen_inputs, candidates):
        expected = list(trajectory.tokens)
        expected[trajectory.commitments[index].position] = 0
        assert ids[0].tolist() == expected


def test_gradient_uses_actual_post_state_and_embedding_to_mask_direction():
    model = TinyMDLM().double()
    trajectory = Sampler(model, TOKENIZER, 8, 4).sample([1, 2], 9)
    eligible = [False, False] + [True] * 8
    index = 0
    result = reconstruction_gradient(model, trajectory, index, eligible, 0)
    assert result is not None
    assert result["historical_input"] == trajectory.state_after(index)
    # The three other current-block masks and all four future-block masks
    # enter the loss; future target tokens never enter its model input.
    assert result["targets"] == 7
    ids = torch.tensor([trajectory.state_after(index)])
    masked = ids[0] == 0
    target = torch.tensor(trajectory.tokens)[masked]
    position = trajectory.commitments[index].position
    with torch.no_grad():
        inputs = model.get_input_embeddings()(ids)
        direction = model.get_input_embeddings().weight[0] - inputs[0, position]
        def shifted_loss(amount):
            altered = inputs.clone()
            altered[0, position] += amount * direction
            logits = model(inputs_embeds=altered).logits[0, masked]
            return F.cross_entropy(logits, target).item()
        epsilon = 1e-3
        numerical = (shifted_loss(epsilon) - shifted_loss(-epsilon)) / (2 * epsilon)
    assert result["score"] == pytest.approx(numerical, abs=2e-5, rel=2e-3)
    assert all(p.grad is None for p in model.parameters())
    assert all(p.requires_grad for p in model.parameters())  # Flags restored.


def test_gradient_selector_evaluates_every_candidate_and_selects_absolute_score():
    model = TinyMDLM()
    trajectory = Sampler(model, TOKENIZER, 8, 4).sample([1, 2], 10)
    selected = select_decision(model, trajectory, [True] * 8, 0, "gradient_ce_cf")
    candidates, _ = eligible_candidates(trajectory, [True] * 8)
    assert selected.metrics["states_scored"] == len(candidates) == 4
    scores = selected.metrics["candidate_scores"]
    expected = max(scores, key=lambda r: abs(r["score"]))
    assert selected.index == expected["index"]
    assert selected.score == expected["score"]
    assert selected.metrics["gradient_forward_seconds"] >= 0
    assert selected.metrics["gradient_backward_seconds"] >= 0
    assert all(p.grad is None for p in model.parameters())


def test_empty_reconstruction_target_and_no_eligibility_are_logged():
    model = TinyMDLM()
    trajectory = Sampler(model, TOKENIZER, 4, 4).sample([1], 11)
    eligible = [False] * 4
    eligible[trajectory.commitments[0].position - 1] = True
    selected = select_decision(model, trajectory, eligible, 0, "gradient_ce_cf")
    assert selected.index is None
    assert selected.metrics["empty_reconstruction_targets"] == 1
    selected = select_decision(model, trajectory, [False] * 4, 0, "entropy_cf")
    assert selected.index is None
    assert selected.metrics["skip_reason"] == "no_eligible_implementation_commitment"
