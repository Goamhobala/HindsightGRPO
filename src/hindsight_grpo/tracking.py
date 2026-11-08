"""Required online W&B tracking with durable, secret-free structured records."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Iterable
import uuid

from .checkpoint import atomic_json


_CREDENTIAL_FIELDS = {"wandb_api_key", "api_key", "hf_token", "access_token", "password", "secret"}


def _safe(value: Any) -> Any:
    """Convert metrics/records without capturing environment or credentials."""
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        for key in value:
            if str(key).lower() in _CREDENTIAL_FIELDS:
                raise ValueError(f"Credential field cannot be logged: {key}")
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        return _safe(value.item() if value.numel() == 1 else value.tolist())
    if hasattr(value, "item"):
        return _safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite metrics cannot be logged")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported structured-log type: {type(value).__name__}")


class Tracker:
    """One resumable online run per arm, plus durable JSONL artifact records.

    `log` sends all update/evaluation metrics immediately. `record` saves detailed
    trajectories, selector decisions and rewards; `flush_records` uploads those
    structured files as versioned artifacts. Call it at every checkpoint/validation
    and finish so detailed records reach the same W&B run throughout training.
    """
    def __init__(self, run_dir: str | Path, arm: str, config: dict[str, Any],
                 run_id: str | None = None, group: str | None = None,
                 project: str | None = None, entity: str | None = None) -> None:
        if not os.environ.get("WANDB_API_KEY", "").strip():
            raise RuntimeError("WANDB_API_KEY is required; refusing to train without online W&B logging")
        if os.environ.get("WANDB_MODE", "online").lower() not in ("online", "run"):
            raise RuntimeError("WANDB_MODE must be online; offline/disabled logging is not permitted")
        config = _safe(config)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.arm = arm
        id_path = self.run_dir / "wandb_run_id"
        existing = id_path.read_text().strip() if id_path.exists() else None
        if existing and run_id and existing != run_id:
            raise RuntimeError("Checkpoint W&B run ID differs from the run directory")
        self.run_id = run_id or existing or uuid.uuid4().hex[:12]
        if not existing:
            # Atomic creation protects concurrent resumptions from creating two runs.
            with id_path.open("x") as stream:
                stream.write(self.run_id + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        import wandb
        self._wandb = wandb
        self._lock = threading.Lock()
        self._records_dirty = False
        self._files: dict[str, Any] = {}
        self.run = wandb.init(
            project=os.environ.get("WANDB_PROJECT") or project or config.get("wandb_project") or "HindsightGRPO",
            entity=os.environ.get("WANDB_ENTITY") or entity or config.get("wandb_entity") or None,
            id=self.run_id,
            resume="allow",
            name=arm,
            group=group or "four-arm-full-mbpp",
            job_type="training",
            config=config,
            dir=str(self.run_dir),
            mode="online",
            settings=wandb.Settings(init_timeout=120),
        )
        if (self.run is None or getattr(self.run, "offline", False)
                or getattr(self.run, "disabled", False)):
            raise RuntimeError("W&B did not establish an online run")
        self.run.define_metric("update")
        self.run.define_metric("*", step_metric="update")
        atomic_json(self.run_dir / "wandb_run.json", {
            "run_id": self.run_id, "url": self.run.url,
            "project": self.run.project, "entity": self.run.entity,
            "arm": arm, "group": group or "four-arm-full-mbpp",
        })
        atomic_json(self.run_dir / "config.json", config)
        self.artifact(f"{arm}-configuration", [self.run_dir / "config.json"], type="configuration")

    @property
    def url(self) -> str:
        return self.run.url

    def _append(self, kind: str, payload: Any, step: int | None) -> None:
        if not kind or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in kind):
            raise ValueError("Record kind must be a simple filename component")
        record = {"time_unix": time.time(), "arm": self.arm,
                  "run_id": self.run_id, "step": step, "payload": _safe(payload)}
        with self._lock:
            stream = self._files.get(kind)
            if stream is None:
                stream = (self.run_dir / f"{kind}.jsonl").open("a", buffering=1)
                self._files[kind] = stream
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            self._records_dirty = True

    def log(self, metrics: dict[str, Any], step: int) -> None:
        payload = _safe(metrics)
        payload["update"] = int(step)
        self._append("metrics", payload, step)
        # W&B manages its internal event step; explicit update is the plotted axis.
        # This allows validation, resumed records and training to share update ids.
        self.run.log(payload)

    def record(self, kind: str, payload: Any, step: int | None = None) -> None:
        self._append(kind, payload, step)

    def artifact(self, name: str, paths: Iterable[str | Path] | str | Path,
                 type: str = "metadata", aliases: list[str] | None = None) -> Any:
        if isinstance(paths, (str, Path)):
            paths = [paths]
        artifact = self._wandb.Artifact(name=name, type=type)
        for item in paths:
            path = Path(item)
            if path.is_dir():
                artifact.add_dir(str(path), name=path.name)
            elif path.is_file():
                artifact.add_file(str(path), name=path.name)
            else:
                raise FileNotFoundError(path)
        return self.run.log_artifact(artifact, aliases=aliases or ["latest"])

    def flush_records(self) -> None:
        with self._lock:
            if not self._records_dirty:
                return
            for stream in self._files.values():
                stream.flush()
                os.fsync(stream.fileno())
            paths = [self.run_dir / f"{kind}.jsonl" for kind in self._files]
            self.artifact(f"{self.arm}-structured-records", paths, type="experiment-records")
            self._records_dirty = False

    def summary(self, values: dict[str, Any]) -> None:
        self.run.summary.update(_safe(values))

    def finish(self, exit_code: int = 0) -> None:
        self.flush_records()
        for stream in self._files.values():
            stream.close()
        self._files.clear()
        self.run.finish(exit_code=exit_code)
