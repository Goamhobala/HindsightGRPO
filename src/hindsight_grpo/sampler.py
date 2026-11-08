"""Recorded confidence-based MDLM sampler; run its invariants with pytest."""

from dataclasses import asdict, dataclass, field
import math
from typing import Any

import torch
import torch.nn.functional as F

from .model import model_device


@dataclass
class Commitment:
    position: int
    token: int
    block: int
    entropy: float
    old_logp: float
    confidence: float
    seed: int
    intervention: str = "ordinary"


@dataclass
class Trajectory:
    prompt_ids: list[int]
    tokens: list[int]
    commitments: list[Commitment]
    mask_id: int
    block_size: int
    temperature: float
    seed: int
    prompt_id: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def generated_ids(self):
        return self.tokens[len(self.prompt_ids):]

    @property
    def generation_length(self):
        return len(self.tokens) - len(self.prompt_ids)

    def state_before(self, index):
        if not 0 <= index <= len(self.commitments):
            raise IndexError(index)
        state = self.prompt_ids + [self.mask_id] * self.generation_length
        for commitment in self.commitments[:index]:
            state[commitment.position] = commitment.token
        return state

    def state_after(self, index):
        return self.state_before(index + 1)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["commitments"] = [Commitment(**c) for c in value["commitments"]]
        return cls(**value)

    def validate(self):
        n_prompt = len(self.prompt_ids)
        seen = set()
        last_block = -1
        for c in self.commitments:
            if c.position in seen or not n_prompt <= c.position < len(self.tokens):
                raise AssertionError("Invalid or repeated commitment position")
            if c.token == self.mask_id:
                raise AssertionError("A commitment must replace a mask")
            if c.block != (c.position - n_prompt) // self.block_size or c.block < last_block:
                raise AssertionError("Invalid block transition")
            seen.add(c.position)
            last_block = c.block
        if len(seen) != self.generation_length or self.state_before(len(self.commitments)) != self.tokens:
            raise AssertionError("History does not exactly reconstruct completed trajectory")


def provisional_tokens(logits, temperature, generator):
    """Exactly upstream's float64 Gumbel-Max algebra and RNG shape."""
    if temperature == 0:
        return logits.argmax(-1)
    logits64 = logits.to(torch.float64)
    noise = torch.rand(logits.shape, dtype=torch.float64, device=logits.device, generator=generator)
    noisy = logits64.exp() / (-noise.log()).pow(temperature)
    return noisy.argmax(-1)


def restricted_alternative(logits, original_token, temperature=0.8):
    """Normalised single-position intervention q_A; original token has zero mass."""
    if temperature <= 0:
        raise ValueError("Alternative interventions require positive temperature")
    restricted = logits.to(torch.float64) / temperature
    restricted = restricted.clone()
    restricted[original_token] = -torch.inf
    probabilities = F.softmax(restricted, -1)
    if not torch.isfinite(probabilities).all() or probabilities.sum() <= 0:
        raise RuntimeError("No valid alternative token distribution")
    return probabilities


class Sampler:
    def __init__(self, model, tokenizer, generation_length=256, block_size=64,
                 temperature=0.8, steps=None):
        if generation_length % block_size or (steps is not None and steps != generation_length):
            raise ValueError("Require one commitment per step and complete blocks")
        self.model = model
        self.tokenizer = tokenizer
        self.mask_id = int(tokenizer.mask_token_id)
        self.generation_length = generation_length
        self.block_size = block_size
        self.temperature = temperature

    @torch.no_grad()
    def sample(self, prompt_ids, seed, prompt_id=""):
        prompt_ids = [int(v) for v in prompt_ids]
        trajectory = Trajectory(
            prompt_ids, prompt_ids + [self.mask_id] * self.generation_length,
            [], self.mask_id, self.block_size, self.temperature, int(seed), str(prompt_id),
        )
        return self._finish(trajectory, int(seed))

    @torch.no_grad()
    def _finish(self, trajectory, seed, generator=None):
        device = model_device(self.model)
        if generator is None:
            generator = torch.Generator(device=device).manual_seed(seed)
        ids = torch.tensor(trajectory.tokens, device=device, dtype=torch.long)[None]
        n_prompt = len(trajectory.prompt_ids)
        for block in range(self.generation_length // self.block_size):
            start = n_prompt + block * self.block_size
            end = start + self.block_size
            while (ids[0, start:end] == self.mask_id).any():
                logits = self.model(ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
                predictions = provisional_tokens(logits, self.temperature, generator)
                logp = F.log_softmax(logits.float(), -1)
                # Upstream confidence softmax retains the model-logit dtype.
                # Promoting BF16 before ranking can alter ties and commitments.
                confidence = F.softmax(logits, -1).gather(-1, predictions[..., None]).squeeze(-1)
                eligible = torch.zeros_like(ids, dtype=torch.bool)
                eligible[:, start:end] = ids[:, start:end] == self.mask_id
                scores = confidence.masked_fill(~eligible, -torch.inf)
                # Match upstream torch.topk rather than an independently chosen tie rule.
                position = int(torch.topk(scores[0], k=1).indices[0])
                token = int(predictions[0, position])
                if token == self.mask_id:
                    raise RuntimeError("Upstream sampler selected MASK; cannot record a token commitment")
                local = logp[0, position]
                entropy = float(-(local.exp() * local).sum())
                tempered = F.log_softmax(logits[0, position].float() / self.temperature, -1)
                trajectory.commitments.append(Commitment(
                    position, token, block, entropy, float(tempered[token]),
                    float(confidence[0, position]), seed,
                ))
                ids[0, position] = token
        trajectory.tokens = ids[0].tolist()
        trajectory.validate()
        return trajectory

    @torch.no_grad()
    def continue_from(self, trajectory, index, keep, seed):
        """Resume exact historical post-state or perform fixed-position exclusion."""
        if trajectory.mask_id != self.mask_id or trajectory.block_size != self.block_size:
            raise ValueError("Trajectory configuration differs from continuation sampler")
        c = trajectory.commitments[index]
        device = model_device(self.model)
        generator = torch.Generator(device=device).manual_seed(int(seed))
        history = list(trajectory.commitments[:index + 1])
        state = trajectory.state_after(index)
        if not keep:
            pre = torch.tensor(trajectory.state_before(index), device=device)[None]
            logits = self.model(pre, attention_mask=torch.ones_like(pre), use_cache=False).logits[0, c.position]
            q = restricted_alternative(logits, c.token, self.temperature)
            alternative = int(torch.multinomial(q, 1, generator=generator))
            if alternative == c.token:
                raise AssertionError("Alternative failed to change the selected token")
            if alternative == self.mask_id:
                raise RuntimeError("Alternative intervention sampled MASK, which cannot be committed")
            state[c.position] = alternative
            local = F.log_softmax(logits.float(), -1)
            history[-1] = Commitment(c.position, alternative, c.block,
                float(-(local.exp() * local).sum()), float(q[alternative].log()),
                float(local[alternative].exp()), int(seed), "alternative_fixed_position")
        resumed = Trajectory(
            list(trajectory.prompt_ids), state, history, self.mask_id,
            self.block_size, self.temperature, int(seed), trajectory.prompt_id,
            {"reference_seed": trajectory.seed, "selected_index": index,
             "selected_position": c.position, "original_token": c.token,
             "intervention": "keep" if keep else "alternative"},
        )
        # One independent RNG stream per continuation. Alternatives consume the
        # intervention draw first, then continue in that same stream.
        return self._finish(resumed, int(seed), generator=generator)

    def counterfactual_groups(self, trajectory, index, keep_seeds, alternative_seeds):
        if len(keep_seeds) != 4 or len(alternative_seeds) != 8:
            raise ValueError("Counterfactual groups must contain four keep and eight alternatives")
        seeds = [int(v) for v in list(keep_seeds) + list(alternative_seeds)]
        if len(set(seeds)) != 12:
            raise ValueError("Each continuation requires an independent sampling seed")
        return ([self.continue_from(trajectory, index, True, s) for s in keep_seeds],
                [self.continue_from(trajectory, index, False, s) for s in alternative_seeds])
