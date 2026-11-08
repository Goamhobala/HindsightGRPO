"""Load the pinned MDLM, not the autoregressive base; used by train/preflight."""

from dataclasses import dataclass
import copy
from typing import Any

import torch
import torch.nn.functional as F


MODEL_ID = "dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1"
MODEL_REVISION = "a284e895a6248256baf2f60502e54aba61b24c1a"
UPSTREAM_COMMIT = "ca176752fbceec49c6b4777a2c18ae88e4eb10ed"


@dataclass
class ModelBundle:
    model: Any
    tokenizer: Any


def _get(config, key, default):
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def model_device(model):
    return next(model.parameters()).device


def load_model(config=None, device="cuda", adaptation=True):
    """Load pinned, bidirectional custom architecture and optional rank-32 LoRA."""
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    config = config or {}
    model_id = _get(config, "model_id", MODEL_ID)
    revision = _get(config, "model_revision", MODEL_REVISION)
    if model_id != MODEL_ID:
        raise ValueError("This experiment requires the specified diffusion MDLM checkpoint")
    if revision != MODEL_REVISION:
        raise ValueError("Model revision differs from the inspected immutable revision")
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    model = AutoModelForMaskedLM.from_pretrained(
        model_id, revision=revision, code_revision=revision,
        trust_remote_code=True, dtype=dtype, attn_implementation="eager",
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if model.config.model_type != "a2d-qwen2":
        raise RuntimeError("Expected the bidirectional A2D Qwen2 architecture")
    if tokenizer.mask_token_id != 151665 or tokenizer.mask_token != "<|mask|>":
        raise RuntimeError("Unexpected MDLM mask token")
    model.config.use_cache = False
    if adaptation:
        from peft import LoraConfig, get_peft_model
        rank = int(_get(config, "lora_rank", 32))
        if rank != 32:
            raise ValueError("The full experiment fixes LoRA rank at 32")
        model = get_peft_model(model, LoraConfig(
            r=rank, lora_alpha=64, lora_dropout=0.0, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "up_proj", "down_proj", "gate_proj"],
            task_type="CAUSAL_LM",
        ))
    # Evaluation mode disables dropout while preserving gradients for LoRA updates.
    model.eval()
    return ModelBundle(model, tokenizer)


def frozen_snapshot(model):
    """Make the rollout checkpoint independent of current LoRA parameter updates."""
    frozen = copy.deepcopy(model).eval()
    frozen.requires_grad_(False)
    return frozen


def root_logps(model, trajectories, mask_id, seed, p_mask_prompt=0.3):
    """Upstream denoising surrogate: all response tokens masked; seeded prompt masks.

    Untempered token log probabilities, no causal shift. This is neither the
    exact any-order trajectory probability nor its committed-token product.
    Call with the same seed for old/current policy for identical corrupted inputs.
    Each trajectory uses seed+its batch index, independent of global RNG state.
    """
    device = model_device(model)
    output = []
    for index, trajectory in enumerate(trajectories):
        ids = torch.tensor(trajectory.tokens, device=device, dtype=torch.long)[None]
        n_prompt = len(trajectory.prompt_ids)
        generator = torch.Generator(device=device).manual_seed(int(seed) + index)
        prompt_mask = torch.rand(ids.shape, device=device, generator=generator) < p_mask_prompt
        prompt_mask[:, n_prompt:] = True
        corrupted = ids.masked_fill(prompt_mask, mask_id)
        logits = model(corrupted, attention_mask=torch.ones_like(ids), use_cache=False).logits
        logits = logits[:, n_prompt:].float()
        token_logp = -F.cross_entropy(
            logits.flatten(0, 1), ids[:, n_prompt:].flatten(), reduction="none",
        )
        # Include EOS itself, then exclude padding and tokens after first stop.
        eos = getattr(getattr(model, "config", None), "eos_token_id", None)
        if eos is not None:
            eos_ids = torch.as_tensor(eos, device=device).reshape(-1)
            stop = torch.isin(ids[0, n_prompt:], eos_ids).nonzero()
            if len(stop):
                token_logp = token_logp[:int(stop[0]) + 1]
        output.append(token_logp)
    return output


def selected_token_logp(model, trajectory, index, temperature=0.8):
    """Current temperature-scaled log probability at the historical fixed state."""
    device = model_device(model)
    ids = torch.tensor(trajectory.state_before(index), device=device)[None]
    commitment = trajectory.commitments[index]
    logits = model(ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
    local = logits[0, commitment.position].double() / temperature
    # Direct float32 log_softmax can round log p_keep to zero for confident
    # tokens, losing a nonzero change probability. Compute binary keep/change
    # log-odds first and use stable log-sigmoid, preserving that complement.
    others = local.clone()
    others[commitment.token] = -torch.inf
    keep_log_odds = local[commitment.token] - torch.logsumexp(others, dim=-1)
    return F.logsigmoid(keep_log_odds)
