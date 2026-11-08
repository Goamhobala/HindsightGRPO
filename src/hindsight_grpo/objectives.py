"""Common objectives for every arm.

The root objective operates on a documented *token-probability surrogate* supplied
by the caller; it is not an exact confidence-selection trajectory likelihood.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def root_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Leave-one-out reward differences, without standard-deviation scaling."""
    rewards = torch.as_tensor(rewards).float()
    if rewards.ndim == 0 or rewards.shape[-1] < 2:
        raise ValueError("At least two root trajectories are required per prompt")
    return rewards - (rewards.sum(dim=-1, keepdim=True) - rewards) / (rewards.shape[-1] - 1)


def clipped_root_loss(
    current_logps: torch.Tensor,
    old_logps: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Tokenwise clipped PPO surrogate, then equal-weight trajectory mean.

    Inputs have shape [trajectories, tokens] (or [trajectories]); advantages have
    shape [trajectories]. Old scores and advantages are detached. Log ratios
    outside [-60,60] are capped solely to avoid overflow in float32; this cap
    must be monitored by the trainer and is outside normal operating ranges.
    """
    if current_logps.shape != old_logps.shape:
        raise ValueError("Current and rollout likelihood shapes differ")
    if current_logps.ndim not in (1, 2):
        raise ValueError("Expected [trajectories] or [trajectories,tokens]")
    if advantages.shape != current_logps.shape[:1]:
        raise ValueError("One advantage is required for every trajectory")
    if not 0 < clip_eps < 1:
        raise ValueError("clip_eps must be between zero and one")
    new = current_logps.float()
    old = old_logps.detach().to(device=new.device, dtype=torch.float32)
    ratio = torch.exp((new - old).clamp(-60.0, 60.0))
    adv = advantages.detach().to(device=new.device, dtype=torch.float32)
    if new.ndim == 2:
        adv = adv.unsqueeze(-1)
    objective = torch.minimum(ratio * adv, ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv)
    if mask is not None:
        if mask.shape != new.shape:
            raise ValueError("Likelihood mask shape differs")
        valid = mask.to(device=new.device, dtype=new.dtype)
        if new.ndim == 1:
            if valid.sum() == 0:
                return new.sum() * 0
            return -(objective * valid).sum() / valid.sum()
        counts = valid.sum(-1)
        selected = counts > 0
        if not bool(selected.any()):
            return new.sum() * 0
        return -((objective * valid).sum(-1)[selected] / counts[selected]).mean()
    return -objective.mean()


def log1mexp(log_p: torch.Tensor) -> torch.Tensor:
    """Stable log(1-exp(log_p)) for log_p <= 0, including p=0 and p=1."""
    if bool(torch.any(log_p > 0)):
        raise ValueError("Log probabilities must be nonpositive")
    cutoff = -math.log(2.0)
    # Avoid evaluating the singular unused branch (torch.where evaluates both).
    out = torch.empty_like(log_p)
    lower = log_p < cutoff
    out[lower] = torch.log1p(-torch.exp(log_p[lower]))
    out[~lower] = torch.log(-torch.expm1(log_p[~lower]))
    return out


def cf_target(old_keep_logprob: torch.Tensor, delta: torch.Tensor | float, beta: float = 4.0) -> torch.Tensor:
    """Return detached log p_target = log sigmoid(logit(p_old)-beta*delta)."""
    old = torch.as_tensor(old_keep_logprob).detach().double()
    contrast = torch.as_tensor(delta, device=old.device, dtype=old.dtype).detach()
    if not bool(torch.isfinite(contrast).all()):
        raise ValueError("Nonfinite counterfactual reward contrast")
    if beta <= 0:
        raise ValueError("KL strength must be positive")
    logit_old = old - log1mexp(old)
    return F.logsigmoid(logit_old - beta * contrast).detach()


def bernoulli_kl_from_logprobs(target_log_p: torch.Tensor, current_log_p: torch.Tensor) -> torch.Tensor:
    """Elementwise KL(Bern(target)||Bern(current)), stable near boundaries.

    Exact target probabilities zero/one are allowed. Current probabilities must
    lie strictly inside (0,1), as a finite softmax over a vocabulary does.
    """
    target = torch.as_tensor(target_log_p, device=current_log_p.device).detach().double()
    current = current_log_p.double()
    if bool(torch.any(current >= 0)) or bool(torch.any(~torch.isfinite(current))):
        raise ValueError("Current keep probability must lie strictly inside (0,1)")
    target_log_q = log1mexp(target)
    current_log_q = log1mexp(current)
    # xlogy handles 0 log 0 but not a 0*inf difference; evaluate only active terms.
    p, q = target.exp(), -torch.expm1(target)
    positive_p = p > 0
    positive_q = q > 0
    first = torch.zeros_like(current + target)
    second = torch.zeros_like(first)
    p, q, target, target_log_q, current, current_log_q = torch.broadcast_tensors(
        p, q, target, target_log_q, current, current_log_q
    )
    positive_p, positive_q = p > 0, q > 0
    first[positive_p] = p[positive_p] * (target[positive_p] - current[positive_p])
    second[positive_q] = q[positive_q] * (target_log_q[positive_q] - current_log_q[positive_q])
    return first + second


def cf_loss(current_log_p: torch.Tensor, old_log_p: torch.Tensor,
            delta: torch.Tensor | float, beta: float = 4.0,
            reduction: str = "mean") -> torch.Tensor:
    loss = bernoulli_kl_from_logprobs(cf_target(old_log_p, delta, beta), current_log_p)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"Unknown reduction: {reduction}")
