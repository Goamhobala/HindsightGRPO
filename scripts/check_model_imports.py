"""Check the pinned model source on CPU without downloading pretrained weights.

This is a login-node import/architecture check. Real model weights, BF16 and
CUDA remain the responsibility of the separate GPU preflight.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import tempfile

from huggingface_hub import hf_hub_download
import torch
import torch.nn.functional as F

from hindsight_grpo.checkpoint import atomic_json
from hindsight_grpo.config import load_config
from hindsight_grpo.model import (
    MODEL_ID, MODEL_REVISION, frozen_snapshot, load_pinned_architecture,
)


def verify_architecture(architecture, original_config: dict) -> dict:
    """Exercise the actual custom model class with small, random CPU weights."""
    from peft import LoraConfig, get_peft_model

    values = dict(original_config)
    values.update(vocab_size=64, hidden_size=32, intermediate_size=64,
                  num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                  max_position_embeddings=64, max_window_layers=2,
                  layer_types=["full_attention", "full_attention"],
                  bos_token_id=1, eos_token_id=2, pad_token_id=0,
                  dtype="float32", use_cache=False, attention_dropout=0.0)
    config = architecture.config_class.from_dict(values)
    config._attn_implementation = "eager"
    torch.manual_seed(1931)
    model = architecture.model_class(config).to(device="cpu", dtype=torch.float32).eval()
    ids = torch.tensor([[4, 3, 6, 7, 8]], dtype=torch.long)
    attention = torch.ones_like(ids)
    with torch.no_grad():
        output = model(ids, attention_mask=attention, use_cache=False).logits
        changed_ids = ids.clone()
        changed_ids[0, -1] = 9
        changed = model(changed_ids, attention_mask=attention, use_cache=False).logits
    if output.shape != (1, 5, 64) or not bool(torch.isfinite(output).all()):
        raise RuntimeError("Pinned custom architecture produced invalid CPU logits")
    future_effect = float((output[0, 1] - changed[0, 1]).abs().max())
    if future_effect <= 1e-6:
        raise RuntimeError("Pinned custom architecture failed bidirectional attention check")

    # Exercise the standard HF save/load path too. The explicit classes avoid
    # the Auto import scanner's guarded-demo false dependency on dllm.
    with tempfile.TemporaryDirectory(prefix="hindsight-model-import-") as directory:
        model.save_pretrained(directory)
        restored_config = architecture.config_class.from_pretrained(directory)
        restored = architecture.model_class.from_pretrained(
            directory, config=restored_config, dtype=torch.float32,
            attn_implementation="eager",
        ).eval()
        with torch.no_grad():
            restored_output = restored(ids, attention_mask=attention, use_cache=False).logits
        if not torch.equal(output, restored_output):
            raise RuntimeError("Tiny checkpoint changed during standard HF save/load")
        # Release any safetensors mappings before deleting files on NFS scratch.
        del restored, restored_output
        gc.collect()

    model = get_peft_model(model, LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "up_proj", "down_proj", "gate_proj"],
    )).eval()
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise RuntimeError("Pinned custom architecture has unexpected trainable parameters")
    logits = model(ids, attention_mask=attention, use_cache=False).logits
    loss = F.cross_entropy(logits[0].float(), torch.tensor([4, 5, 6, 7, 8]))
    loss.backward()
    gradients = [p.grad for _, p in trainable if p.grad is not None]
    if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
        raise RuntimeError("Pinned custom architecture produced invalid LoRA gradients")
    gradient_norm = sum(float(g.float().square().sum()) for g in gradients) ** .5
    if gradient_norm <= 0:
        raise RuntimeError("Pinned custom architecture produced zero LoRA gradients")

    # Input-gradient scoring must also work through the same frozen architecture.
    frozen = frozen_snapshot(model)
    embeddings = frozen.get_input_embeddings()(ids).requires_grad_(True)
    logits = frozen(inputs_embeds=embeddings, attention_mask=attention, use_cache=False).logits
    reconstruction = F.cross_entropy(logits[0, 1:4].float(), torch.tensor([5, 6, 7]))
    input_gradient, = torch.autograd.grad(reconstruction, embeddings)
    if not bool(torch.isfinite(input_gradient).all()) or not bool(input_gradient.abs().sum() > 0):
        raise RuntimeError("Pinned custom architecture produced invalid input gradients")
    if any(p.grad is not None or p.requires_grad for p in frozen.parameters()):
        raise RuntimeError("Frozen architecture accumulated parameter gradients")
    return {
        "status": "passed", "device": "cpu", "precision": "float32",
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "model_type": config.model_type,
        "source_path": str(architecture.source_path),
        "source_sha256": architecture.source_sha256,
        "tiny_logits_shape": list(output.shape), "tiny_parameter_count": sum(p.numel() for p in model.parameters()),
        "future_token_effect_max_abs_logit": future_effect,
        "lora_gradient_norm": gradient_norm,
        "input_gradient_norm": float(input_gradient.norm()),
        "frozen_parameters_have_no_gradients": True,
        "tiny_checkpoint_roundtrip": True,
        "pretrained_weights_loaded": False,
    }


def verify_model_imports(config: dict, offline: bool = False) -> dict:
    architecture = load_pinned_architecture(config, local_files_only=offline)
    path = Path(hf_hub_download(MODEL_ID, filename="config.json", revision=MODEL_REVISION,
                               local_files_only=offline))
    original = json.loads(path.read_text())
    if original.get("model_type") != "a2d-qwen2":
        raise RuntimeError("Pinned checkpoint metadata has an unexpected architecture")
    report = verify_architecture(architecture, original)
    report["config_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--output", default="runtime/metadata/model_imports.json")
    parser.add_argument("--offline", action="store_true", help="Require cached model source/config; never contact the Hub")
    args = parser.parse_args()
    report = verify_model_imports(load_config(args.config), offline=args.offline)
    atomic_json(args.output, report)
    print(f"Pinned model import/CPU architecture checks passed; evidence: {args.output}")


if __name__ == "__main__":
    main()
