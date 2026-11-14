"""d1's one-step mean-field likelihood and diffu-GRPO loss.

The executable convention is pinned and audited in ``docs/d1_reference.md``.
In particular, d1 averages over *all valid response tokens in the batch*, not
over equally weighted response means. No historical response token is visible
to this likelihood estimator. Generation remains confidence-based any-order.
"""
from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn.functional as F

from hindsight_grpo.model import model_device

D1_REVISION = "faf85d6dcea40477c722c467023c73b08a46cb82"
D1_TRAINER_SHA256 = "b4c006d773af7f89cc7a6a333465ba2acd580d5767b9e56dd45d35eda3c534dc"


def root_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Reward minus the full group's mean; no LOO or std normalization."""
    rewards = torch.as_tensor(rewards).float()
    if rewards.ndim < 1 or rewards.shape[-1] < 2:
        raise ValueError("At least two responses are required in each reward group")
    if not bool(torch.isfinite(rewards).all()):
        raise ValueError("Rewards must be finite")
    return (rewards - rewards.mean(dim=-1, keepdim=True)).detach()


def _ids(value: int | Sequence[int] | None) -> list[int]:
    if value is None:
        return []
    return [int(value)] if isinstance(value, int) else [int(v) for v in value]


def completion_mask(response_ids: torch.Tensor, eos_token_id: int | Sequence[int] | None,
                    pad_token_id: int | None = None) -> torch.Tensor:
    """Keep through first EOS (inclusive); exclude distinct PAD and its suffix.

    Supports a single response or a batch. The PAD rule adapts d1's EOS-only
    scoring to this checkpoint's distinct padding ID and the shared MBPP decoder.
    No response padding is inserted by the training sampler.
    """
    ids = torch.as_tensor(response_ids)
    if ids.ndim not in (1, 2):
        raise ValueError("Expected [tokens] or [responses,tokens]")
    eos = _ids(eos_token_id)
    stops = torch.zeros_like(ids, dtype=torch.bool)
    for token in eos:
        stops |= ids == token
    # First EOS itself contributes to d1's objective.
    mask = (stops.cumsum(dim=-1) - stops.long()) == 0
    if pad_token_id is not None and int(pad_token_id) not in eos:
        mask &= (ids == int(pad_token_id)).cumsum(dim=-1) == 0
    return mask


def corrupt_prompt_response(prompt_ids, response_ids, mask_id: int, seed: int,
                            p_mask_prompt: float = .15, device=None,
                            prompt_attention_mask=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the corrupted model input and its bidirectional attention mask.

    Draw independent Bernoulli prompt masks using a local seeded generator.
    All response positions, including those after EOS, remain MASK: diffusion
    generation's fixed 256-position window is retained. Prompt padding can be
    supplied explicitly and receives attention zero, never a prompt corruption.
    The returned tensors always include a batch axis.
    """
    if not 0 <= p_mask_prompt <= 1:
        raise ValueError("Prompt mask probability must lie in [0,1]")
    prompt = torch.as_tensor(prompt_ids, dtype=torch.long, device=device)
    response = torch.as_tensor(response_ids, dtype=torch.long, device=device)
    if prompt.ndim == 1:
        prompt = prompt.unsqueeze(0)
    if response.ndim == 1:
        response = response.unsqueeze(0)
    if prompt.ndim != 2 or response.ndim != 2 or prompt.shape[0] != response.shape[0]:
        raise ValueError("Prompt and response batches must have equal row counts")
    attention_prompt = (torch.ones_like(prompt) if prompt_attention_mask is None else
                        torch.as_tensor(prompt_attention_mask, device=prompt.device, dtype=torch.long))
    if attention_prompt.ndim == 1:
        attention_prompt = attention_prompt.unsqueeze(0)
    if attention_prompt.shape != prompt.shape:
        raise ValueError("Prompt attention mask shape differs")
    # Draw the full shape, as d1.forward_process does, but do not mutate global RNG.
    generator = torch.Generator(device=prompt.device).manual_seed(int(seed))
    full_shape = (prompt.shape[0], prompt.shape[1] + response.shape[1])
    random_matrix = torch.rand(full_shape, device=prompt.device, generator=generator)
    hide_prompt = (random_matrix[:, :prompt.shape[1]] < p_mask_prompt) & attention_prompt.bool()
    corrupted = torch.cat([prompt.masked_fill(hide_prompt, int(mask_id)),
                           torch.full_like(response, int(mask_id))], dim=-1)
    attention = torch.cat([attention_prompt, torch.ones_like(response)], dim=-1)
    return corrupted, attention


def token_logps(model: Any, trajectory: Any, mask_id: int, seed: int,
                p_mask_prompt: float = .15, *, eos_token_id=None,
                pad_token_id=None) -> torch.Tensor:
    """One response's untempered per-token scores through first stop.

    Old, current and reference must be called with the same trajectory and seed.
    This function deliberately performs no adapter switch or ``no_grad``: those
    lifecycle choices belong to the trainer. The full response is masked before
    prediction, and there is no autoregressive causal shift.
    """
    device = model_device(model)
    response = torch.as_tensor(trajectory.generated_ids, device=device, dtype=torch.long)
    corrupted, attention = corrupt_prompt_response(
        trajectory.prompt_ids, response, mask_id, seed, p_mask_prompt, device=device)
    logits = model(corrupted, attention_mask=attention, use_cache=False).logits
    n_prompt = len(trajectory.prompt_ids)
    logps = -F.cross_entropy(logits[0, n_prompt:].float(), response, reduction="none")
    config = getattr(model, "config", None)
    if eos_token_id is None:
        eos_token_id = getattr(config, "eos_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(config, "pad_token_id", None)
    return logps[completion_mask(response, eos_token_id, pad_token_id)]


def per_token_terms(current_logps: torch.Tensor, old_logps: torch.Tensor,
                    reference_logps: torch.Tensor, advantages,
                    epsilon: float = .5, beta: float = .04) -> dict[str, torch.Tensor]:
    """Unreduced d1 terms for streaming accumulation with one global denominator.

    Accepts a token vector plus a scalar advantage, or [responses,tokens] plus
    [responses] advantages. Old scores, reference scores and advantages are
    detached. This follows official d1's unclamped likelihood-ratio algebra;
    nonfinite results are rejected, never silently saturated.
    """
    if not 0 < epsilon < 1 or beta < 0:
        raise ValueError("Require 0 < epsilon < 1 and beta >= 0")
    if current_logps.ndim not in (1, 2):
        raise ValueError("Expected a token vector or [responses,tokens]")
    current = current_logps.float()
    old = torch.as_tensor(old_logps, device=current.device).detach().float()
    reference = torch.as_tensor(reference_logps, device=current.device).detach().float()
    if current.shape != old.shape or current.shape != reference.shape:
        raise ValueError("Current, old and reference score shapes differ")
    advantage = torch.as_tensor(advantages, device=current.device).detach().float()
    if current.ndim == 2 and advantage.shape == current.shape[:1]:
        advantage = advantage.unsqueeze(-1)
    if advantage.ndim > current.ndim:
        raise ValueError("Invalid advantage shape")
    try:
        advantage = torch.broadcast_to(advantage, current.shape)
    except RuntimeError as exc:
        raise ValueError("Advantage shape differs from response scores") from exc
    ratio = torch.exp(current - old)
    unclipped = ratio * advantage
    clipped = ratio.clamp(1 - epsilon, 1 + epsilon) * advantage
    pg = -torch.minimum(unclipped, clipped)
    difference = reference - current
    # expm1(d)-d has the same mathematical definition as official exp(d)-d-1,
    # with less cancellation when current and reference almost agree.
    kl = torch.expm1(difference) - difference
    loss = pg + beta * kl
    if not bool(torch.isfinite(loss).all()) or not bool(torch.isfinite(ratio).all()):
        raise FloatingPointError("Nonfinite diffu-GRPO loss or ratio")
    return {"loss": loss, "pg": pg, "kl": kl, "ratio": ratio,
            "clipped": unclipped > clipped,
            "official_is_clipped": unclipped < clipped}


def diffu_grpo_loss(current_logps: torch.Tensor, old_logps: torch.Tensor,
                    reference_logps: torch.Tensor, advantages,
                    mask: torch.Tensor | None = None,
                    epsilon: float = .5, beta: float = .04) -> torch.Tensor:
    """Official d1 masked *global token mean* over the complete collection."""
    terms = per_token_terms(current_logps, old_logps, reference_logps,
                            advantages, epsilon, beta)
    if mask is None:
        if terms["loss"].numel() == 0:
            raise ValueError("No response tokens to optimize")
        return terms["loss"].mean()
    mask = torch.as_tensor(mask, device=current_logps.device, dtype=torch.bool)
    if mask.shape != current_logps.shape or not bool(mask.any()):
        raise ValueError("Invalid or empty completion mask")
    return terms["loss"][mask].mean()


d1_loss = diffu_grpo_loss
