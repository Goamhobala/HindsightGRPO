from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from d1_exclusion.counterfactual import (estimate_conditional_values,
    fixed_target_loss, cf_target, bernoulli_kl_from_logprobs)
from d1_exclusion.proposals import (ExclusionSampler, proposal_distribution,
    keep_logprob, proposal_value_decomposition, verify_branch_states)
from d1_exclusion.selector import reference_index, select_entropy
from d1_exclusion.trajectory import Commitment, Trajectory
from hindsight_grpo.sampler import Sampler


class ConstantMDLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor([-60., .7, -.3, .1, -.7, .3]))

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.bias[None, None].expand(*input_ids.shape, -1))


TOKENIZER = SimpleNamespace(mask_token_id=0)


@pytest.mark.parametrize("alpha", [0., .5, 1.])
def test_exact_mixture_and_common_changed_action_distribution(alpha):
    logits = torch.tensor([-100., -.4, 1.7, -.9, .5], dtype=torch.float64)
    p = torch.softmax(logits / .8, -1)
    a = 2
    q = proposal_distribution(logits, a, alpha)
    assert q.sum() == pytest.approx(1.)
    assert q[a] == pytest.approx(float((1 - alpha) * p[a]))
    assert 1 - q[a] == pytest.approx(float(1 - (1 - alpha) * p[a]))
    changed = torch.arange(len(logits)) != a
    q_exclude = p[changed] / p[changed].sum()
    assert torch.allclose(q[changed] / q[changed].sum(), q_exclude)
    if alpha == 0:
        assert torch.equal(q, p)
    if alpha == 1:
        assert q[a] == 0
    # Population reward identity uses q_exclude, regardless of intervention.
    rewards = torch.tensor([0., 1., 0., 1., 1.], dtype=torch.float64)
    q_keep = rewards[a]
    q_alt = (q_exclude * rewards[changed]).sum()
    q_b = (q * rewards).sum()
    expected = proposal_value_decomposition(float(q_keep), float(q_alt), float(p[a]), alpha)
    assert q_b == pytest.approx(expected)
    assert q_b - q_keep == pytest.approx(float((1 - (1 - alpha) * p[a]) * (q_alt - q_keep)))


def test_hard_exclusion_survives_saturated_keep_probability():
    logits = torch.tensor([50., 1., 2.], dtype=torch.float64)
    assert torch.softmax(logits / .8, -1)[0] == 1.
    q = proposal_distribution(logits, 0, 1.)
    assert q[0] == 0
    assert torch.isfinite(q).all()
    assert q.sum() == pytest.approx(1.)
    assert torch.allclose(q[1:], torch.softmax(logits[1:] / .8, -1))
    # Stable score retains a finite, nonzero change complement.
    logp = keep_logprob(logits, 0)
    assert logp < 0
    assert torch.isfinite(torch.log(-torch.expm1(logp)))


def test_partial_exclusion_is_probability_mixture_and_matches_empirical_draws():
    logits = torch.tensor([0., 1., 2.], dtype=torch.float64)
    p = torch.softmax(logits / .8, -1)
    q = proposal_distribution(logits, 2, .5)
    assert q[2] == pytest.approx(float(p[2] / 2))
    generator = torch.Generator().manual_seed(91)
    samples = torch.multinomial(q, 100_000, replacement=True, generator=generator)
    empirical = torch.bincount(samples, minlength=3).double() / samples.numel()
    assert torch.allclose(q, empirical, atol=.006)


@pytest.mark.parametrize("alpha", [0., .5, 1.])
def test_conditional_continuations_keep_position_and_restore_states(alpha):
    sampler = ExclusionSampler(ConstantMDLM(), TOKENIZER, 8, 4)
    reference = sampler.sample([1, 2], 7)
    index = 2
    keep, proposals = sampler.counterfactual_groups(reference, index, range(100, 104),
                                                   range(200, 208), alpha)
    verify_branch_states(reference, index, keep, proposals)
    assert len(keep) == 4 and len(proposals) == 8
    original = reference.commitments[index]
    for branch in proposals:
        selected = branch.commitments[index]
        assert selected.position == original.position
        assert branch.metadata["changed"] == (selected.token != original.token)
        assert branch.metadata["alpha"] == alpha
        assert branch.to_dict() == sampler.continue_proposal(reference, index, alpha, branch.seed).to_dict()
    if alpha == 1:
        assert all(branch.commitments[index].token != original.token for branch in proposals)
    else:
        # Other draws can genuinely repeat the original fixed-position action.
        all_proposals = [sampler.continue_proposal(reference, index, alpha, seed) for seed in range(30)]
        repeats = [branch for branch in all_proposals if not branch.metadata["changed"]]
        assert repeats
        assert all(branch.state_after(index) == reference.state_after(index) for branch in repeats)
    assert len({branch.seed for branch in keep + proposals}) == 12


def test_ordinary_sampler_is_reused_without_changes():
    model = ConstantMDLM()
    ordinary = Sampler(model, TOKENIZER, 8, 4).sample([1, 2], 321)
    exclusion = ExclusionSampler(model, TOKENIZER, 8, 4).sample([1, 2], 321)
    assert ordinary.to_dict() == exclusion.to_dict()


def test_hard_exclusion_does_not_ban_original_token_at_later_positions():
    sampler = ExclusionSampler(ConstantMDLM(), TOKENIZER, 8, 4)
    reference = sampler.sample([1, 2], 8)
    original = reference.commitments[0].token
    branches = [sampler.continue_proposal(reference, 0, 1., seed) for seed in range(10)]
    assert all(branch.commitments[0].token != original for branch in branches)
    assert any(original in [c.token for c in branch.commitments[1:]] for branch in branches)


def test_missing_alternative_rate_matches_repeat_probability_to_eighth_power():
    logits = torch.tensor([0., 2.], dtype=torch.float64)
    original = 1
    p = torch.softmax(logits / .8, dim=-1)[original]
    generator = torch.Generator().manual_seed(42)
    for alpha in (0., .5, 1.):
        q = proposal_distribution(logits, original, alpha)
        groups = torch.multinomial(q, 8 * 30_000, replacement=True, generator=generator).reshape(-1, 8)
        observed = (groups == original).all(dim=-1).double().mean()
        expected = ((1 - alpha) * p) ** 8
        assert observed == pytest.approx(float(expected), abs=.01)


def test_pooling_repeated_proposals_never_uses_proposal_mean_as_alternative():
    record = estimate_conditional_values([1, 0, 1, 0], [1, 1, 0, 0, 1, 0, 0, 0],
        [2, 2, 3, 4, 2, 3, 4, 3], 2, torch.log(torch.tensor(.7, dtype=torch.float64)))
    assert record["n_keep"] == 3 and record["n_alt"] == 5
    assert record["keep_denominator"] == 7
    assert record["q_keep"] == pytest.approx(5 / 7)
    assert record["q_alt"] == 0.
    assert record["q_proposal"] == pytest.approx(3 / 8)
    assert record["q_alt"] != record["q_proposal"]
    assert record["delta"] == pytest.approx(-5 / 7)
    assert record["target_logprob"] > record["old_keep_logprob"]


def test_no_alternative_preserves_observations_without_invented_target():
    keep, proposals = [1, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0, 0]
    record = estimate_conditional_values(keep, proposals, [2] * 8, 2, -.001)
    assert record["skip_reason"] == "no_alternative_observed"
    assert not record["valid"]
    assert record["q_alt"] is record["delta"] is record["target_logprob"] is None
    assert record["q_keep"] == pytest.approx(3 / 12)
    assert record["keep_rewards"] == keep and record["proposal_rewards"] == proposals
    with pytest.raises(ValueError, match="No auxiliary loss"):
        fixed_target_loss(F.logsigmoid(torch.tensor(1.)), record)


@pytest.mark.parametrize("log_odds", [-35., -1., 0., 2., 35.])
def test_auxiliary_log_odds_derivative_matches_analytic_and_finite_difference(log_odds):
    z = torch.tensor(log_odds, dtype=torch.float64, requires_grad=True)
    target = cf_target(F.logsigmoid(torch.tensor(.8, dtype=torch.float64)), .3)
    loss = bernoulli_kl_from_logprobs(target, F.logsigmoid(z))
    derivative, = torch.autograd.grad(loss, z)
    assert torch.isfinite(loss)
    expected = z.sigmoid() - target.exp()
    assert derivative == pytest.approx(float(expected.detach()), abs=1e-12)
    h = 1e-5
    plus = bernoulli_kl_from_logprobs(target, F.logsigmoid(z.detach() + h))
    minus = bernoulli_kl_from_logprobs(target, F.logsigmoid(z.detach() - h))
    assert derivative == pytest.approx(float((plus - minus) / (2 * h)), abs=5e-10)
    assert not target.requires_grad


def test_fixed_target_is_unchanged_across_trainable_scores():
    record = estimate_conditional_values([0, 0, 0, 0], [1] * 8, [3] * 8, 2, -.5)
    target = record["target_logprob"]
    losses = [fixed_target_loss(F.logsigmoid(torch.tensor(z, dtype=torch.float64)), record)
              for z in (-1., 0., 1.)]
    assert all(torch.isfinite(loss) for loss in losses)
    assert len({float(loss) for loss in losses}) == 3
    assert record["target_logprob"] == target


def test_estimator_preserves_tiny_float64_complement_of_keep_probability():
    old_logp = -1e-50
    record = estimate_conditional_values([0] * 4, [1] * 8, [3] * 8, 2, old_logp)
    assert record["old_keep_logprob"] == old_logp
    assert record["target_logprob"] < 0
    assert record["target_logprob"] == pytest.approx(-torch.exp(torch.tensor(4.)).item() * 1e-50,
                                                    rel=1e-7, abs=0)
    current = F.logsigmoid(torch.tensor(100., dtype=torch.float64, requires_grad=True))
    auxiliary = fixed_target_loss(current, record)
    assert torch.isfinite(auxiliary)
    derivative, = torch.autograd.grad(auxiliary, current)
    assert torch.isfinite(derivative)


def test_entropy_selector_uses_historical_progress_and_untempered_score():
    # Token positions deliberately disagree with historical commitment order.
    commitments = [Commitment(pos, pos, 0, entropy, -.5, .5, 1)
                   for pos, entropy in [(4, .2), (1, .9), (3, 20.), (2, 10.)]]
    trajectory = Trajectory([5], [5, 1, 2, 3, 4], commitments, 0, 4, .8, 1)
    trajectory.validate()
    selected = select_entropy(trajectory, [True] * 4)
    assert selected.index == 1
    assert selected.metrics["selected_progress"] == .25
    assert selected.metrics["selected_position"] == 1
    assert selected.score == .9
    assert select_entropy(trajectory, [False] * 4).metrics["skip_reason"] == "no_eligible_implementation_commitment"


def test_homogeneous_reference_selection_is_deterministic_and_covers_all_roots():
    selected = [reference_index([0] * 8, seed) for seed in range(3000)]
    assert set(selected) == set(range(8))
    counts = [selected.count(index) for index in range(8)]
    assert all(abs(count - 375) < 70 for count in counts)
    assert reference_index([0] * 8, 781) == reference_index([0] * 8, 781)


def test_continuation_seed_reuse_and_invalid_rewards_are_rejected():
    sampler = ExclusionSampler(ConstantMDLM(), TOKENIZER, 4, 4)
    trajectory = sampler.sample([1], 1)
    with pytest.raises(ValueError, match="independent seed"):
        sampler.counterfactual_groups(trajectory, 0, [1] * 4, range(8), 0.)
    with pytest.raises(ValueError, match="binary"):
        estimate_conditional_values([0, 0, 0, .5], [0] * 8, [2] * 8, 1, -.5)
