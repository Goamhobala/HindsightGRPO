"""Conditional reward pooling and the fixed local KL policy-improvement target."""
from __future__ import annotations

import math

import torch

# The earlier implementation's Bernoulli mathematics is already stable and
# matches the new experiment. Its previous root objective is not imported.
from hindsight_grpo.objectives import cf_target, bernoulli_kl_from_logprobs


def estimate_conditional_values(keep_rewards, proposal_rewards, proposal_tokens,
                                original_token, old_keep_logprob, beta=4.) -> dict:
    """Pool repeats into Q_K; estimate Q_A only from genuinely changed tokens.

    Returned records retain every observed reward, including missing-alternative
    cases. Q_B is logged separately and never used as the alternative estimate.
    The scalar detached target can be persisted with a frozen collection.
    """
    keep_rewards = [float(v) for v in keep_rewards]
    proposal_rewards = [float(v) for v in proposal_rewards]
    proposal_tokens = [int(v) for v in proposal_tokens]
    if len(keep_rewards) != 4 or len(proposal_rewards) != 8 or len(proposal_tokens) != 8:
        raise ValueError("Expected four keep rewards and eight proposal rewards/tokens")
    if any(v not in (0., 1.) for v in keep_rewards + proposal_rewards):
        raise ValueError("The MBPP reward must be binary correctness")
    # Python floats from selected_token_logp are already float64. Converting
    # through the default float32 tensor can round a tiny negative log-p to
    # zero and incorrectly collapse a high-confidence target to probability 1.
    old_logp = float(torch.as_tensor(old_keep_logprob, dtype=torch.float64).detach())
    if math.isnan(old_logp) or old_logp > 0:
        raise ValueError("The old keep log probability must be nonpositive")
    repeated = [i for i, token in enumerate(proposal_tokens) if token == int(original_token)]
    changed = [i for i, token in enumerate(proposal_tokens) if token != int(original_token)]
    n_keep, n_alt = len(repeated), len(changed)
    if n_keep + n_alt != 8:
        raise AssertionError("Proposal partition does not contain all eight samples")
    q_keep = (sum(keep_rewards) + sum(proposal_rewards[i] for i in repeated)) / (4 + n_keep)
    record = {"keep_rewards": keep_rewards, "proposal_rewards": proposal_rewards,
              "proposal_tokens": proposal_tokens, "original_token": int(original_token),
              "repeated_indices": repeated, "changed_indices": changed,
              "n_keep": n_keep, "n_alt": n_alt, "keep_denominator": 4 + n_keep,
              "q_keep": q_keep, "q_proposal": sum(proposal_rewards) / 8,
              "old_keep_logprob": old_logp, "q_alt": None, "delta": None,
              "target_logprob": None, "valid": n_alt > 0, "skip_reason": None}
    if not n_alt:
        record["skip_reason"] = "no_alternative_observed"
        return record
    q_alt = sum(proposal_rewards[i] for i in changed) / n_alt
    delta = q_alt - q_keep
    target = cf_target(torch.tensor(old_logp, dtype=torch.float64), delta, beta)
    record.update(q_alt=q_alt, delta=delta, target_logprob=float(target))
    return record


def fixed_target_loss(current_keep_logprob: torch.Tensor, record: dict) -> torch.Tensor:
    """Evaluate one persisted target against the trainable selected-state score."""
    if not record.get("valid") or record.get("target_logprob") is None:
        raise ValueError("No auxiliary loss exists without an observed alternative")
    target = torch.as_tensor(record["target_logprob"], device=current_keep_logprob.device,
                             dtype=torch.float64).detach()
    return bernoulli_kl_from_logprobs(target, current_keep_logprob).mean()


__all__ = ["estimate_conditional_values", "fixed_target_loss", "cf_target",
           "bernoulli_kl_from_logprobs"]
