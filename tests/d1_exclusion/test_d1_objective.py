"""Numerical parity with the pinned executable d1 conventions, using CPU toys."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from d1_exclusion.d1_objective import (
    completion_mask, corrupt_prompt_response, diffu_grpo_loss, per_token_terms,
    root_advantages, token_logps,
)


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.arange(12).float() / 5)
        self.config = SimpleNamespace(eos_token_id=9, pad_token_id=10)
        self.calls = []

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        self.calls.append((input_ids.detach().clone(), attention_mask.detach().clone()))
        # A bidirectional context dependency makes response leakage observable.
        context = input_ids.float().sum(-1, keepdim=True).unsqueeze(-1)
        logits = self.logits[None, None, :].expand(*input_ids.shape, -1)
        logits = logits + context * torch.arange(12, device=input_ids.device)[None, None, :] / 100
        return SimpleNamespace(logits=logits)


def test_group_centering_is_neither_loo_nor_std_normalized():
    rewards = torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0.], [1.] * 8], requires_grad=True)
    advantages = root_advantages(rewards)
    torch.testing.assert_close(advantages[0], torch.tensor([.875] + [-.125] * 7))
    assert not advantages.requires_grad
    assert torch.equal(advantages[1], torch.zeros(8))


def test_prompt_corruption_matches_official_independent_bernoulli_algebra():
    prompt = torch.arange(100).reshape(2, 50)
    response = torch.tensor([[2, 3, 4], [5, 6, 7]])
    before = torch.get_rng_state().clone()
    actual, attention = corrupt_prompt_response(prompt, response, 11, 47)
    generator = torch.Generator().manual_seed(47)
    random_matrix = torch.rand((2, 53), generator=generator)
    expected = torch.cat([prompt.masked_fill(random_matrix[:, :50] < .15, 11),
                          torch.full_like(response, 11)], dim=-1)
    assert torch.equal(actual, expected)
    assert torch.equal(before, torch.get_rng_state())
    assert attention.eq(1).all()
    different, _ = corrupt_prompt_response(prompt, response, 11, 48)
    assert not torch.equal(actual[:, :50], different[:, :50])


def test_prompt_padding_is_not_corrupted_and_receives_attention_zero():
    actual, attention = corrupt_prompt_response([10, 10, 2, 3], [4, 5], 11, 5,
                                                p_mask_prompt=1., prompt_attention_mask=[0, 0, 1, 1])
    assert actual.tolist() == [[10, 10, 11, 11, 11, 11]]
    assert attention.tolist() == [[0, 0, 1, 1, 1, 1]]


def test_eos_inclusive_distinct_pad_exclusive_and_empty_shapes():
    ids = torch.tensor([[1, 9, 2, 3], [1, 10, 9, 2], [1, 2, 3, 4]])
    assert completion_mask(ids, 9, 10).tolist() == [
        [True, True, False, False], [True, False, False, False], [True] * 4]
    assert completion_mask(torch.tensor([9, 9]), 9, 9).tolist() == [True, False]
    assert completion_mask(torch.tensor([2, 8, 9]), [8, 9]).tolist() == [True, True, False]
    assert completion_mask(torch.empty(0, dtype=torch.long), 9).numel() == 0


def test_scores_have_no_response_context_leakage_no_causal_shift_and_keep_eos():
    model = RecordingModel()
    trajectory = SimpleNamespace(prompt_ids=[1, 2], generated_ids=[3, 9, 4, 5])
    actual = token_logps(model, trajectory, 11, seed=12, p_mask_prompt=0.)
    corrupted, attention = model.calls[-1]
    assert corrupted.tolist() == [[1, 2, 11, 11, 11, 11]]
    assert attention.eq(1).all()
    logits = model(corrupted, attention_mask=attention).logits[0, 2:]
    expected = -F.cross_entropy(logits, torch.tensor([3, 9, 4, 5]), reduction="none")
    torch.testing.assert_close(actual, expected[:2])
    changed = SimpleNamespace(prompt_ids=[1, 2], generated_ids=[8, 9, 7, 6])
    token_logps(model, changed, 11, seed=12, p_mask_prompt=0.)
    assert torch.equal(model.calls[-1][0], corrupted)
    actual.sum().backward()
    assert model.logits.grad.abs().sum() > 0


def test_identical_policy_and_corruption_scores_are_equal():
    import copy
    current = RecordingModel()
    old = copy.deepcopy(current).requires_grad_(False)
    trajectory = SimpleNamespace(prompt_ids=list(range(8)), generated_ids=[1, 2, 3])
    new = token_logps(current, trajectory, 11, seed=11)
    frozen = token_logps(old, trajectory, 11, seed=11)
    torch.testing.assert_close(new, frozen, atol=0, rtol=0)
    assert torch.equal(current.calls[-1][0], old.calls[-1][0])


def test_loss_matches_literal_official_formula_including_global_token_mean():
    old = torch.tensor([[-2., -1., -3.], [-1., -2., -1.]], requires_grad=True)
    current = torch.tensor([[-1., -2., -2.], [-2., -3., -1.5]], requires_grad=True)
    reference = torch.tensor([[-3., -2., -1.], [-2., -1., -3.]], requires_grad=True)
    adv = torch.tensor([.7, -.3], requires_grad=True)
    mask = torch.tensor([[1, 0, 0], [1, 1, 1]], dtype=torch.bool)
    ratio = (current - old.detach()).exp()
    pg = -torch.minimum(ratio * adv.detach()[:, None],
                        ratio.clamp(.5, 1.5) * adv.detach()[:, None])
    difference = reference.detach() - current
    kl = difference.exp() - difference - 1
    expected = ((pg + .04 * kl) * mask).sum() / mask.sum()
    actual = diffu_grpo_loss(current, old, reference, adv, mask)
    torch.testing.assert_close(actual, expected)
    response_mean = (((pg + .04 * kl) * mask).sum(-1) / mask.sum(-1)).mean()
    assert not torch.isclose(actual, response_mean)
    actual.backward()
    assert current.grad.abs().sum() > 0
    assert old.grad is None and reference.grad is None and adv.grad is None


def test_streaming_loss_and_gradient_equal_whole_collection_including_two_prompt_tail():
    # Different response lengths make an incorrect average-of-means detectable.
    for prompts in (4, 2):
        generator = torch.Generator().manual_seed(102 + prompts)
        current = torch.randn(prompts * 8, 5, generator=generator, requires_grad=True)
        old = current.detach() - .12
        reference = current.detach() + .04
        advantages = root_advantages(torch.randint(0, 2, (prompts, 8), generator=generator)).flatten()
        mask = torch.arange(5)[None, :] <= torch.arange(prompts * 8)[:, None] % 5
        full = diffu_grpo_loss(current, old, reference, advantages, mask)
        full_grad, = torch.autograd.grad(full, current, retain_graph=True)
        streamed = current.sum() * 0
        for row in range(len(current)):
            terms = per_token_terms(current[row, mask[row]], old[row, mask[row]],
                                    reference[row, mask[row]], advantages[row])
            streamed = streamed + terms["loss"].sum() / mask.sum()
        stream_grad, = torch.autograd.grad(streamed, current)
        torch.testing.assert_close(streamed, full)
        torch.testing.assert_close(stream_grad, full_grad)


def test_clipping_and_reference_kl_have_correct_directions():
    current = torch.tensor([2., -2., 0.], requires_grad=True)
    terms = per_token_terms(current, torch.zeros(3), torch.zeros(3), torch.tensor(1.))
    torch.testing.assert_close(terms["pg"], -torch.tensor([1.5, torch.exp(torch.tensor(-2.)), 1.]))
    assert terms["clipped"].tolist() == [True, False, False]
    assert terms["official_is_clipped"].tolist() == [False, True, False]
    assert (terms["kl"] >= 0).all() and terms["kl"][2] == 0
    terms["kl"].sum().backward()
    assert current.grad[0] > 0 and current.grad[1] < 0 and current.grad[2] == 0


def test_invalid_inputs_fail_before_optimization():
    with pytest.raises(ValueError):
        root_advantages(torch.tensor([float("nan"), 1.]))
    with pytest.raises(FloatingPointError):
        per_token_terms(torch.tensor([1000.]), torch.zeros(1), torch.zeros(1), 1.)
    with pytest.raises(ValueError):
        diffu_grpo_loss(torch.zeros(1), torch.zeros(1), torch.zeros(1), 1., mask=torch.tensor([False]))


def test_actual_mdlm_peft_reference_is_fixed_while_current_adapter_changes():
    """Use the exact audited architecture, tiny dimensions and random CPU weights."""
    from pathlib import Path
    from unittest.mock import patch
    from peft import LoraConfig, get_peft_model
    from hindsight_grpo.model import frozen_snapshot, load_pinned_architecture

    source = Path(__file__).resolve().parents[1] / "fixtures/hf_mdlm_modeling_qwen2.py"
    with patch("hindsight_grpo.model.hf_hub_download", return_value=str(source)):
        architecture = load_pinned_architecture()
    config = architecture.config_class(vocab_size=16, hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, max_position_embeddings=32,
        bos_token_id=1, eos_token_id=9, pad_token_id=10)
    config._attn_implementation = "eager"
    with torch.random.fork_rng():
        torch.manual_seed(41)
        model = get_peft_model(architecture.model_class(config), LoraConfig(
            r=32, lora_alpha=64, lora_dropout=0., bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
            task_type="CAUSAL_LM")).eval()
    old = frozen_snapshot(model)
    trajectory = SimpleNamespace(prompt_ids=[1, 2, 3], generated_ids=[4, 5, 9, 6])
    with torch.no_grad(), model.disable_adapter():
        reference_before = token_logps(model, trajectory, 11, 44)
    frozen_before = token_logps(old, trajectory, 11, 44)
    trainable_before = {name for name, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    for _ in range(3):
        optimizer.zero_grad()
        current = token_logps(model, trajectory, 11, 44)
        diffu_grpo_loss(current, frozen_before, reference_before, 1.).backward()
        optimizer.step()
    with torch.no_grad(), model.disable_adapter():
        reference_after = token_logps(model, trajectory, 11, 44)
    torch.testing.assert_close(reference_before, reference_after, atol=0, rtol=0)
    torch.testing.assert_close(frozen_before, token_logps(old, trajectory, 11, 44), atol=0, rtol=0)
    assert not torch.allclose(frozen_before, token_logps(model, trajectory, 11, 44))
    assert {name for name, p in model.named_parameters() if p.requires_grad} == trainable_before
    assert all(not p.requires_grad for p in old.parameters())
