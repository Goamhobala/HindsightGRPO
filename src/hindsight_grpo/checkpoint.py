"""Atomic, trusted local training checkpoints with exact RNG restoration."""
from __future__ import annotations

import json
import os
from pathlib import Path
import random
import signal
import tempfile
from typing import Any

import numpy as np
import torch


SCHEMA_VERSION = 1


def capture_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("Cannot restore CUDA checkpoint RNG without a CUDA device")
        if len(state["cuda"]) != torch.cuda.device_count():
            raise RuntimeError("CUDA device count differs from the checkpoint")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def atomic_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_checkpoint(path: str | Path, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer, progress: dict[str, Any],
                    config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Atomically persist adapters (or full model), optimizer, cursor and all RNGs.

    Progress must describe the *next* unconsumed prompt, include the epoch,
    shuffled-order/cursor, accumulation_count and stable wandb_run_id. Save after
    each completed prompt; accumulated parameter gradients are restored too.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    peft = hasattr(model, "peft_config")
    if peft:
        from peft import get_peft_model_state_dict
        weights = get_peft_model_state_dict(model)
    else:
        weights = model.state_dict()
    weights = {key: value.detach().cpu().clone() for key, value in weights.items()}
    state = {
        "schema_version": SCHEMA_VERSION,
        "model_kind": "peft" if peft else "full",
        "model": weights,
        "optimizer": optimizer.state_dict(),
        "gradients": {name: parameter.grad.detach().cpu().clone()
                      for name, parameter in model.named_parameters()
                      if parameter.requires_grad and parameter.grad is not None},
        "progress": dict(progress),
        "config": dict(config or {}),
        "rng": capture_rng(),
    }
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return dict(progress)


def load_checkpoint(path: str | Path, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer | None = None,
                    map_location: Any = "cpu", restore_randomness: bool = True) -> dict[str, Any]:
    # Only open checkpoints produced by this experiment: optimizer/RNG require
    # trusted pickle state, rather than the tensor-only weights_only format.
    state = torch.load(path, map_location=map_location, weights_only=False)
    if state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported training checkpoint schema")
    if state["model_kind"] == "peft":
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(model, state["model"])
    else:
        model.load_state_dict(state["model"], strict=True)
    for name, parameter in model.named_parameters():
        parameter.grad = None
        if name in state.get("gradients", {}):
            parameter.grad = state["gradients"][name].to(device=parameter.device, dtype=parameter.dtype)
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if restore_randomness:
        restore_rng(state["rng"])
    return {"progress": state["progress"], "config": state["config"]}


class SignalHandler:
    """Flag termination for checkpointing at an optimizer transaction boundary."""
    def __init__(self) -> None:
        self.requested = False
        self.signal_number: int | None = None
        self._previous: dict[int, Any] = {}

    def _handle(self, number: int, frame: Any) -> None:
        self.requested = True
        self.signal_number = number

    def install(self) -> "SignalHandler":
        for number in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
            self._previous[number] = signal.signal(number, self._handle)
        return self

    def close(self) -> None:
        for number, previous in self._previous.items():
            signal.signal(number, previous)
        self._previous.clear()

    def __enter__(self) -> "SignalHandler":
        return self.install()

    def __exit__(self, *args: Any) -> None:
        self.close()
