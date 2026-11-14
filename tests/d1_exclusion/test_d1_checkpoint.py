"""Restore actual MDLM LoRA, fixed old weights, RNG and scheduler together."""
import copy
from pathlib import Path
import random
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from d1_exclusion.checkpoint import load_checkpoint, restore_policy, save_checkpoint
from d1_exclusion.d1_objective import token_logps
from hindsight_grpo.model import frozen_snapshot, load_pinned_architecture


def test_actual_peft_old_policy_optimizer_scheduler_and_rng_round_trip(tmp_path):
    from peft import LoraConfig, get_peft_model
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        source = Path(__file__).parents[1] / "fixtures/hf_mdlm_modeling_qwen2.py"
        with patch("hindsight_grpo.model.hf_hub_download", return_value=str(source)):
            architecture = load_pinned_architecture()
        cfg = architecture.config_class(vocab_size=16, hidden_size=16,
            intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=1, max_position_embeddings=32,
            bos_token_id=1, eos_token_id=9, pad_token_id=10)
        cfg._attn_implementation = "eager"
        torch.manual_seed(15)
        model = get_peft_model(architecture.model_class(cfg), LoraConfig(
            r=32, lora_alpha=64, lora_dropout=0., bias="none",
            target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")).eval()
        original = copy.deepcopy(model)
        old = frozen_snapshot(model)
        trajectory = SimpleNamespace(prompt_ids=[1, 2, 3], generated_ids=[4, 5, 9, 6])
        old_score = token_logps(old, trajectory, 11, 57)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.005)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1., step))
        def step(current, opt, sched):
            opt.zero_grad(set_to_none=True)
            (-token_logps(current, trajectory, 11, 57).mean()).backward()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
        step(model, optimizer, scheduler)
        step(model, optimizer, scheduler)
        collection = [{"task_id": 5, "target_logprob": -.3, "rewards": [0., 1.]}]
        path = tmp_path / "latest.pt"
        save_checkpoint(path, model, optimizer, {"inner_iteration": 2, "cursor": 0},
                        {"arm": "entropy_soft_ban"}, scheduler=scheduler,
                        old_model=old, collection=collection)
        expected_rng = (random.random(), np.random.rand(), torch.rand(3))
        step(model, optimizer, scheduler)
        restored = copy.deepcopy(original)
        opt = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad], lr=.005)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: min(1., step))
        state = load_checkpoint(path, restored, opt, sched)
        assert state["collection"] == collection
        assert state["progress"] == {"inner_iteration": 2, "cursor": 0}
        actual_rng = (random.random(), np.random.rand(), torch.rand(3))
        assert actual_rng[:2] == expected_rng[:2]
        torch.testing.assert_close(actual_rng[2], expected_rng[2], rtol=0, atol=0)
        resumed_old = frozen_snapshot(restored)
        restore_policy(resumed_old, state["old_weights"])
        torch.testing.assert_close(token_logps(resumed_old, trajectory, 11, 57), old_score, rtol=0, atol=0)
        step(restored, opt, sched)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
        assert sched.state_dict() == scheduler.state_dict()
        with restored.disable_adapter(), torch.no_grad():
            ref = token_logps(restored, trajectory, 11, 57)
        with original.disable_adapter(), torch.no_grad():
            expected_ref = token_logps(original, trajectory, 11, 57)
        torch.testing.assert_close(ref, expected_ref, rtol=0, atol=0)
    finally:
        torch.set_num_threads(threads)
