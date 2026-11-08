"""Run inside an L40S SLURM allocation before any four-arm verification update."""

import argparse
import hashlib
import inspect
import os
from pathlib import Path

import torch

from hindsight_grpo.checkpoint import atomic_json
from hindsight_grpo.config import config_digest, load_config
from hindsight_grpo.model import frozen_snapshot, load_model, root_logps
from hindsight_grpo.objectives import clipped_root_loss
from hindsight_grpo.sampler import Commitment, Trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--output", default="runtime/verification/gpu_preflight.json")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("GPU preflight requires a SLURM compute allocation")
    gpu_name = torch.cuda.get_device_name()
    if "L40S" not in gpu_name:
        raise RuntimeError(f"This experiment requires L40S resources, got {gpu_name}")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The assigned GPU does not support the requested BF16 precision")
    torch.manual_seed(1931)
    config = load_config(args.config)
    bundle = load_model(config)
    model, tokenizer = bundle.model, bundle.tokenizer
    old = frozen_snapshot(model)
    device = next(model.parameters()).device
    ids = tokenizer.encode("def identity(x):\n    return x\n", add_special_tokens=False)
    if len(ids) < 5:
        raise RuntimeError("Unexpected tokenizer output in model verification")
    prompt, targets = ids[:3], ids[3:7]
    trajectory = Trajectory(prompt, prompt + targets, [
        Commitment(len(prompt) + i, token, 0, 0., -1., .5, 1931)
        for i, token in enumerate(targets)
    ], tokenizer.mask_token_id, len(targets), .8, 1931)
    # Quantitative causal-mask check: changing a later real input must affect an
    # earlier masked position. A causal AR model cannot pass this check.
    first = torch.tensor([prompt + [tokenizer.mask_token_id] + targets], device=device)
    second = first.clone()
    second[0, -1] = tokenizer.encode("banana", add_special_tokens=False)[0]
    if second[0, -1] == first[0, -1]:
        second[0, -1] = tokenizer.encode("orange", add_special_tokens=False)[0]
    with torch.no_grad():
        before = old(first, attention_mask=torch.ones_like(first), use_cache=False).logits[0, len(prompt)].float()
        after = old(second, attention_mask=torch.ones_like(second), use_cache=False).logits[0, len(prompt)].float()
    future_effect = float((before - after).abs().max())
    if not future_effect > 1e-4:
        raise RuntimeError("Later input does not affect earlier logits: bidirectional attention check failed")
    versions = [p._version for p in old.parameters()]
    old_logps = root_logps(old, [trajectory], tokenizer.mask_token_id, 81)[0]
    current_logps = root_logps(model, [trajectory], tokenizer.mask_token_id, 81)[0]
    max_logp_difference = float((old_logps - current_logps.detach()).abs().max())
    if not torch.equal(old_logps, current_logps.detach()):
        raise RuntimeError("Frozen/current denoising surrogate differs at identical parameters")
    loss = clipped_root_loss(current_logps[None], old_logps[None], torch.ones(1, device=device))
    loss.backward()
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise RuntimeError("Unexpected trainable non-LoRA parameters")
    gradients = [p.grad for _, p in trainable if p.grad is not None]
    if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
        raise RuntimeError("LoRA gradients are missing or nonfinite")
    gradient_norm = sum(float(g.float().square().sum()) for g in gradients) ** .5
    if gradient_norm <= 0:
        raise RuntimeError("LoRA gradients are zero")
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=config["learning_rate"])
    optimizer.step()
    if [p._version for p in old.parameters()] != versions:
        raise RuntimeError("Frozen rollout parameters were modified")
    if any(p.grad is not None or p.requires_grad for p in old.parameters()):
        raise RuntimeError("Gradient accumulation leaked into the frozen rollout policy")
    changed = any(not torch.equal(p, dict(old.named_parameters())[name]) for name, p in trainable)
    if not changed:
        raise RuntimeError("Finite LoRA gradients did not change adapter weights")
    base = model.get_base_model()
    custom_source = Path(inspect.getfile(type(base.model)))
    output = {
        "status": "passed", "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "gpu_name": gpu_name, "precision": "bfloat16",
        "model_id": config["model_id"], "model_revision": config["model_revision"],
        "config_digest": config_digest(config), "mask_token_id": tokenizer.mask_token_id,
        "model_type": base.config.model_type,
        "custom_model_source_sha256": hashlib.sha256(custom_source.read_bytes()).hexdigest(),
        "future_token_effect_max_abs_logit": future_effect,
        "old_current_surrogate_max_difference": max_logp_difference,
        "lora_gradient_norm": gradient_norm, "lora_parameters_changed": changed,
        "frozen_parameters_unchanged": True,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
    }
    atomic_json(args.output, output)
    print(f"GPU preflight passed; evidence: {args.output}")


if __name__ == "__main__":
    main()
