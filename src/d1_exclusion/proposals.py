"""Exact single-position mixtures followed by the unchanged upstream sampler."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from hindsight_grpo.model import model_device
from hindsight_grpo.sampler import Sampler, restricted_alternative
from .trajectory import Commitment, Trajectory


def _validate(logits, original_token, alpha, temperature):
    if logits.ndim != 1 or not 0 <= int(original_token) < logits.numel():
        raise ValueError("Expected one vocabulary vector and an in-range token")
    if not math.isfinite(float(alpha)) or not 0 <= alpha <= 1:
        raise ValueError("Exclusion strength alpha must lie in [0,1]")
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("Sampling temperature must be positive")
    if torch.isnan(logits).any() or torch.isposinf(logits).any():
        raise ValueError("Invalid proposal logits")


def proposal_distribution(logits: torch.Tensor, original_token: int, alpha: float,
                          temperature: float = .8) -> torch.Tensor:
    """Return (1-alpha)*p + alpha*q_exclude, normalising in probability space.

    Exclusion is computed from logits rather than dividing by 1-p_keep. This
    retains the alternative distribution even when floating point p_keep is 1.
    """
    _validate(logits, original_token, alpha, temperature)
    p = F.softmax(logits.double() / temperature, dim=-1)
    if not torch.isfinite(p).all():
        raise ValueError("No valid ordinary token distribution")
    if alpha == 0:
        return p
    alternative = restricted_alternative(logits, original_token, temperature)
    return (1 - alpha) * p + alpha * alternative


def keep_logprob(logits: torch.Tensor, original_token: int,
                 temperature: float = .8) -> torch.Tensor:
    """Stable log p_keep, preserving the complement of very confident tokens."""
    _validate(logits, original_token, 0., temperature)
    local = logits.double() / temperature
    others = local.clone()
    others[original_token] = -torch.inf
    return F.logsigmoid(local[original_token] - torch.logsumexp(others, dim=-1))


def proposal_value_decomposition(q_keep, q_alt, old_keep_probability, alpha):
    """Population Q_B and exploration multiplier, for diagnostics and tests.

    This identity concerns conditional expectations. Finite-sample estimates
    need not satisfy it exactly and Q_B must never replace Q_A in the loss.
    """
    if not 0 <= old_keep_probability <= 1 or not 0 <= alpha <= 1:
        raise ValueError("Probabilities and alpha must lie in [0,1]")
    repeat = (1 - alpha) * old_keep_probability
    return repeat * q_keep + (1 - repeat) * q_alt


class ExclusionSampler(Sampler):
    """Reuse ordinary sampling; intervene exactly once at a fixed position."""

    @torch.no_grad()
    def selected_keep_logprob(self, trajectory, index):
        c = trajectory.commitments[index]
        device = model_device(self.model)
        ids = torch.tensor(trajectory.state_before(index), device=device)[None]
        logits = self.model(ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0, c.position]
        return keep_logprob(logits, c.token, self.temperature)

    @torch.no_grad()
    def continue_proposal(self, trajectory, index: int, alpha: float, seed: int):
        if (trajectory.mask_id != self.mask_id or trajectory.block_size != self.block_size
                or trajectory.temperature != self.temperature
                or trajectory.generation_length != self.generation_length):
            raise ValueError("Trajectory configuration differs from continuation sampler")
        c = trajectory.commitments[index]
        pre_state = trajectory.state_before(index)
        device = model_device(self.model)
        ids = torch.tensor(pre_state, device=device, dtype=torch.long)[None]
        logits = self.model(ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0, c.position]
        q = proposal_distribution(logits, c.token, alpha, self.temperature)
        generator = torch.Generator(device=device).manual_seed(int(seed))
        token = int(torch.multinomial(q, 1, generator=generator))
        if token == self.mask_id:
            raise RuntimeError("Proposal sampled MASK, which cannot be committed")
        if alpha == 1 and token == c.token:
            raise AssertionError("Hard exclusion repeated the historical token")
        post_state = list(pre_state)
        post_state[c.position] = token
        if token == c.token and post_state != trajectory.state_after(index):
            raise AssertionError("Repeated proposal and keep post-states differ")
        local = F.log_softmax(logits.double(), dim=-1)
        nonzero = torch.isfinite(local)
        entropy = float(-(local[nonzero].exp() * local[nonzero]).sum())
        history = list(trajectory.commitments[:index]) + [Commitment(
            c.position, token, c.block, entropy, float(q[token].log()),
            float(F.softmax(logits, dim=-1)[token]), int(seed), "proposal_fixed_position")]
        old_logp = float(keep_logprob(logits, c.token, self.temperature))
        repeat_probability = (1 - alpha) * math.exp(old_logp)
        resumed = Trajectory(list(trajectory.prompt_ids), post_state, history,
            self.mask_id, self.block_size, self.temperature, int(seed), trajectory.prompt_id,
            {"reference_seed": trajectory.seed, "selected_index": index,
             "selected_position": c.position, "original_token": c.token,
             "proposal_token": token, "changed": token != c.token, "alpha": alpha,
             "old_keep_logprob": old_logp, "predicted_change_probability": 1 - repeat_probability,
             "predicted_no_alternative_probability": repeat_probability ** 8,
             "intervention": "proposal"})
        # After the single categorical draw, the ordinary any-order code uses
        # independent subsequent random draws. No vocabulary-wide future ban.
        return self._finish(resumed, int(seed), generator=generator)

    def counterfactual_groups(self, trajectory, index, keep_seeds, proposal_seeds,
                              alpha: float):
        keep_seeds, proposal_seeds = list(keep_seeds), list(proposal_seeds)
        if len(keep_seeds) != 4 or len(proposal_seeds) != 8:
            raise ValueError("Counterfactual groups require four keep and eight proposal samples")
        if len(set(map(int, keep_seeds + proposal_seeds))) != 12:
            raise ValueError("Each continuation needs an independent seed")
        keep = [self.continue_from(trajectory, index, True, seed) for seed in keep_seeds]
        proposals = [self.continue_proposal(trajectory, index, alpha, seed) for seed in proposal_seeds]
        verify_branch_states(trajectory, index, keep, proposals)
        return keep, proposals


def verify_branch_states(reference, index, keep, proposals):
    """Check that pooling repeated proposals compares identical post-states."""
    c = reference.commitments[index]
    pre = reference.state_before(index)
    post = reference.state_after(index)
    for branch in list(keep) + list(proposals):
        branch.validate()
        if branch.state_before(index) != pre or branch.commitments[index].position != c.position:
            raise AssertionError("Conditional branch differs before the targeted commitment")
    for branch in keep:
        if branch.state_after(index) != post:
            raise AssertionError("Keep branch did not restore the historical post-state")
    for branch in proposals:
        if branch.commitments[index].token == c.token and branch.state_after(index) != post:
            raise AssertionError("Repeated proposal cannot be pooled with keep continuations")


__all__ = ["ExclusionSampler", "proposal_distribution", "keep_logprob",
           "proposal_value_decomposition", "verify_branch_states"]
