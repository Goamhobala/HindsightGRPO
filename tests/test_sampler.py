"""Run CPU trajectory and upstream-reference tests with uv run pytest tests/test_sampler.py."""

import itertools
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from hindsight_grpo.model import frozen_snapshot, root_logps
from hindsight_grpo.sampler import Sampler, Trajectory, restricted_alternative


class TinyMDLM(nn.Module):
    """Small genuinely bidirectional model, with a negligible MASK output logit."""
    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(123)
        self.embedding = nn.Embedding(8, 5)
        self.output = nn.Linear(5, 8)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.randn(8, 5, generator=generator) * .3)
            self.output.weight.copy_(torch.randn(8, 5, generator=generator) * .4)
            self.output.bias.zero_()
            self.output.bias[0] = -100
        self.config = SimpleNamespace(eos_token_id=7)
        self.seen_inputs = []

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        if input_ids is not None:
            self.seen_inputs.append(input_ids.detach().clone())
        embeds = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        states = embeds + .7 * embeds.mean(dim=1, keepdim=True)
        return SimpleNamespace(logits=self.output(states))


TOKENIZER = SimpleNamespace(mask_token_id=0)


def upstream_reference(model, prompt, seed, length=8, block_size=4, temperature=.8):
    """Independent literal upstream loop for the fixed one-transfer schedule."""
    ids = torch.tensor([prompt + [0] * length])
    generator = torch.Generator().manual_seed(seed)
    history = []
    with torch.no_grad():
        for block in range(length // block_size):
            for _ in range(block_size):
                logits = model(ids, attention_mask=torch.ones_like(ids)).logits
                noise = torch.rand(logits.shape, dtype=torch.float64, generator=generator)
                predictions = (logits.double().exp() / (-noise.log()).pow(temperature)).argmax(-1)
                confidence = F.softmax(logits, -1).gather(-1, predictions[..., None]).squeeze(-1)
                valid = torch.zeros_like(ids, dtype=torch.bool)
                start = len(prompt) + block * block_size
                valid[:, start:start + block_size] = ids[:, start:start + block_size] == 0
                position = int(confidence.masked_fill(~valid, -torch.inf)[0].topk(1).indices[0])
                token = int(predictions[0, position])
                ids[0, position] = token
                history.append((position, token))
    return ids[0].tolist(), history


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_matches_upstream_confidence_commitment_and_full_canvas(dtype):
    model = TinyMDLM().to(dtype=dtype)
    sampler = Sampler(model, TOKENIZER, generation_length=8, block_size=4)
    actual = sampler.sample([1, 2], 91)
    tokens, history = upstream_reference(model, [1, 2], 91)
    assert actual.tokens == tokens
    assert [(c.position, c.token) for c in actual.commitments] == history
    assert len(actual.commitments) == 8
    assert [c.block for c in actual.commitments] == [0] * 4 + [1] * 4
    assert all(len(ids[0]) == 10 for ids in model.seen_inputs)


def test_exact_state_reconstruction_and_serialisation():
    trajectory = Sampler(TinyMDLM(), TOKENIZER, 8, 4).sample([1, 2], 42)
    trajectory.validate()
    assert Trajectory.from_dict(trajectory.to_dict()) == trajectory
    for index, c in enumerate(trajectory.commitments):
        before, after = trajectory.state_before(index), trajectory.state_after(index)
        assert before[c.position] == 0
        assert after[c.position] == c.token
        assert sum(a != b for a, b in zip(before, after)) == 1
        assert before[:2] == after[:2] == [1, 2]


def test_restricted_distribution_and_twelve_independent_seeded_interventions():
    model = TinyMDLM()
    sampler = Sampler(model, TOKENIZER, 8, 4)
    reference = sampler.sample([1, 2], 3)
    selected = 2
    keep, alternatives = sampler.counterfactual_groups(reference, selected, range(100, 104), range(200, 208))
    original = reference.commitments[selected]
    assert len(keep) == 4 and len(alternatives) == 8
    assert len({t.seed for t in keep + alternatives}) == 12
    for trajectory in keep + alternatives:
        trajectory.validate()
        assert trajectory.state_before(selected) == reference.state_before(selected)
        assert trajectory.commitments[selected].position == original.position
    assert all(t.tokens[original.position] == original.token for t in keep)
    assert all(t.tokens[original.position] != original.token for t in alternatives)
    assert alternatives[0].to_dict() == sampler.continue_from(reference, selected, False, 200).to_dict()
    q = restricted_alternative(torch.tensor([1., 2., 3.]), 1, .8)
    assert q[1] == 0
    assert torch.allclose(q.sum(), torch.tensor(1., dtype=q.dtype))
    expected = torch.softmax(torch.tensor([1., 3.], dtype=torch.float64) / .8, -1)
    assert torch.allclose(q[[0, 2]], expected)


def test_root_surrogate_old_current_agree_and_frozen_snapshot_unchanged():
    current = TinyMDLM()
    trajectory = Sampler(current, TOKENIZER, 8, 4).sample([1, 2], 4)
    old = frozen_snapshot(current)
    original_old = {k: v.clone() for k, v in old.state_dict().items()}
    old_logps = root_logps(old, [trajectory], 0, 77)
    current_logps = root_logps(current, [trajectory], 0, 77)
    assert torch.equal(old_logps[0], current_logps[0])
    (-current_logps[0].mean()).backward()
    with torch.no_grad():
        for p in current.parameters():
            if p.grad is not None:
                p.add_(p.grad, alpha=-.01)
    assert all(torch.equal(v, old.state_dict()[k]) for k, v in original_old.items())
    assert all(p.grad is None and not p.requires_grad for p in old.parameters())


def test_enumerable_sampler_token_probability_is_not_transition_likelihood():
    # Provisional candidates at A and B have categorical probabilities (.9,.1)
    # and (.6,.4). Confidence selection means B0 is committed only with A1.
    probabilities = [[.9, .1], [.6, .4]]
    kernel = {}
    for a, b in itertools.product(range(2), repeat=2):
        selected_position = 0 if probabilities[0][a] > probabilities[1][b] else 1
        selected_token = [a, b][selected_position]
        key = (selected_position, selected_token)
        kernel[key] = kernel.get(key, 0.) + probabilities[0][a] * probabilities[1][b]
    assert sum(kernel.values()) == pytest.approx(1.)
    assert kernel[(1, 0)] == pytest.approx(.06)
    assert probabilities[1][0] == .6
    assert kernel[(1, 0)] != probabilities[1][0]
    assert (0, 1) not in kernel
