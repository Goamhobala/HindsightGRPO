"""Reuse atomic saves, adding frozen collections, old adapters and scheduler state."""
from __future__ import annotations

import copy

from hindsight_grpo.checkpoint import (SignalHandler, atomic_json, capture_rng,
                                       restore_rng, save_checkpoint as _save,
                                       load_checkpoint as _load)


def policy_weights(model):
    if hasattr(model, "peft_config"):
        from peft import get_peft_model_state_dict
        weights = get_peft_model_state_dict(model)
    else:
        weights = model.state_dict()
    return {key: value.detach().cpu().clone() for key, value in weights.items()}


def restore_policy(model, weights):
    if hasattr(model, "peft_config"):
        from peft import set_peft_model_state_dict
        result = set_peft_model_state_dict(model, weights)
        if result.unexpected_keys:
            raise ValueError(f"Unexpected saved adapter keys: {result.unexpected_keys}")
    else:
        model.load_state_dict(weights, strict=True)


def save_checkpoint(path, model, optimizer, progress, config, *, scheduler=None,
                    old_model=None, collection=None):
    saved = copy.deepcopy(progress)
    saved["d1_exclusion"] = {"schema": 1, "scheduler": scheduler.state_dict() if scheduler else None,
                             "old_weights": policy_weights(old_model) if old_model is not None else None,
                             "collection": collection}
    return _save(path, model, optimizer, saved, config)


def load_checkpoint(path, model, optimizer=None, scheduler=None, restore_randomness=True,
                    map_location="cpu"):
    state = _load(path, model, optimizer, map_location=map_location,
                  restore_randomness=restore_randomness)
    extra = state["progress"].pop("d1_exclusion", None)
    if extra is None or extra.get("schema") != 1:
        raise ValueError("Not a d1 exclusion checkpoint; earlier experiments cannot be resumed here")
    if scheduler is not None:
        if extra["scheduler"] is None:
            raise ValueError("Checkpoint is missing its scheduler state")
        scheduler.load_state_dict(extra["scheduler"])
    return {**state, **extra}
