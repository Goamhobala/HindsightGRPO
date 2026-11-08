"""Full-candidate entropy, reconstruction-doubt, and embedding-to-mask selectors."""

from dataclasses import dataclass, field
import time

import torch
import torch.nn.functional as F

from .model import model_device


@dataclass
class Selection:
    index: int | None
    score: float | None
    metrics: dict = field(default_factory=dict)


def eligible_candidates(trajectory, eligible):
    """Use commitment progress, not textual token position, for the first half."""
    eligible = [bool(v) for v in eligible]
    n_prompt = len(trajectory.prompt_ids)
    if len(eligible) == trajectory.generation_length:
        eligible = [False] * n_prompt + eligible
    if len(eligible) != len(trajectory.tokens):
        raise ValueError("Eligibility must align with full or generated-only token positions")
    total = sum(eligible[n_prompt:])
    before = 0
    candidates = []
    if total:
        for index, commitment in enumerate(trajectory.commitments):
            if eligible[commitment.position]:
                progress = before / total
                if progress < 0.5:
                    candidates.append((index, progress))
                before += 1
    return candidates, eligible


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reconstruction_gradient(model, trajectory, index, eligible, mask_id):
    """Signed directional derivative at the actual historical post-state.

    Future blocks remain masks in inputs. The full bidirectional canvas is the
    model prediction context, so all eligible remaining masks are targets,
    including masked future blocks. Reference tokens occur only in CE labels.
    """
    device = model_device(model)
    ids = torch.tensor(trajectory.state_after(index), device=device)[None]
    eligible_tensor = torch.tensor(eligible, device=device, dtype=torch.bool)
    if len(eligible_tensor) == trajectory.generation_length:
        eligible_tensor = torch.cat([
            torch.zeros(len(trajectory.prompt_ids), device=device, dtype=torch.bool),
            eligible_tensor,
        ])
    targets_mask = (ids[0] == mask_id) & eligible_tensor
    target_positions = targets_mask.nonzero(as_tuple=True)[0]
    if not len(target_positions):
        return None
    targets = torch.tensor(trajectory.tokens, device=device)[target_positions]
    committed_position = trajectory.commitments[index].position
    embedding_layer = model.get_input_embeddings()
    # Parameters of the old policy stay frozen. Enable gradients only on the
    # actual embedding tensor produced by the historical token inputs.
    parameter_flags = [(p, p.requires_grad) for p in model.parameters()]
    try:
        for p, _ in parameter_flags:
            p.requires_grad_(False)
        with torch.enable_grad():
            embeddings = embedding_layer(ids)
            embeddings.requires_grad_(True)
            _sync(device)
            forward_start = time.perf_counter()
            logits = model(inputs_embeds=embeddings,
                           attention_mask=torch.ones_like(ids), use_cache=False).logits
            loss = F.cross_entropy(logits[0, target_positions].float(), targets, reduction="mean")
            _sync(device)
            forward_seconds = time.perf_counter() - forward_start
            backward_start = time.perf_counter()
            input_gradient, = torch.autograd.grad(loss, embeddings, retain_graph=False, create_graph=False)
            _sync(device)
            backward_seconds = time.perf_counter() - backward_start
            # This is the prescribed embedding-to-MASK direction, not |gradient|.
            direction = embedding_layer.weight[mask_id] - embeddings[0, committed_position]
            directional = (direction.float() * input_gradient[0, committed_position].float()).sum()
            if not torch.isfinite(directional) or not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite reconstruction gradient")
            return {
                "score": float(directional.detach()), "ce": float(loss.detach()),
                "targets": len(target_positions), "forward_seconds": forward_seconds,
                "backward_seconds": backward_seconds,
                "historical_input": ids[0].tolist(),
            }
    finally:
        for p, flag in parameter_flags:
            p.requires_grad_(flag)


def select_decision(model, trajectory, eligible, mask_id, arm, temperature=0.8):
    candidates, full_eligible = eligible_candidates(trajectory, eligible)
    metrics = {"eligible_candidates": len(candidates), "states_scored": 0,
               "empty_reconstruction_targets": 0, "gradient_failures": 0}
    if not candidates:
        metrics["skip_reason"] = "no_eligible_implementation_commitment"
        return Selection(None, None, metrics)
    started = time.perf_counter()
    device = model_device(model)
    baseline_memory = 0
    if device.type == "cuda":
        _sync(device)
        baseline_memory = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    scored = []
    candidate_records = []
    for index, progress in candidates:
        c = trajectory.commitments[index]
        extra = {}
        if arm in {"entropy_cf", "entropy"}:
            score = c.entropy
        elif arm in {"r_cf", "r"}:
            ids = torch.tensor(trajectory.tokens, device=device)[None]
            ids[0, c.position] = mask_id
            with torch.no_grad():
                logits = model(ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
                score = float(1 - F.softmax(logits[0, c.position].float(), -1)[c.token])
        elif arm in {"gradient_ce_cf", "gradient"}:
            try:
                result = reconstruction_gradient(model, trajectory, index, full_eligible, mask_id)
            except Exception:
                metrics["gradient_failures"] += 1
                # Failing exact scoring is an experiment failure; never silently
                # drop difficult candidates and choose from a reduced candidate set.
                raise
            if result is None:
                metrics["empty_reconstruction_targets"] += 1
                continue
            score = result["score"]
            extra = {k: v for k, v in result.items() if k not in {"score", "historical_input"}}
            for key in ("forward_seconds", "backward_seconds"):
                metrics["gradient_" + key] = metrics.get("gradient_" + key, 0.0) + result[key]
        else:
            raise ValueError(f"Unknown selector {arm}")
        metrics["states_scored"] += 1
        scored.append((abs(score) if arm in {"gradient_ce_cf", "gradient"} else score,
                       index, score, progress, extra))
        candidate_records.append({"index": index, "position": c.position, "progress": progress,
                                  "score": score, **extra})
    _sync(device)
    metrics["scoring_seconds"] = time.perf_counter() - started
    metrics["candidate_scores"] = candidate_records
    if device.type == "cuda":
        metrics["peak_memory_bytes"] = torch.cuda.max_memory_allocated(device)
        metrics["additional_peak_memory_bytes"] = max(0, metrics["peak_memory_bytes"] - baseline_memory)
    else:
        metrics["peak_memory_bytes"] = metrics["additional_peak_memory_bytes"] = 0
    if not scored:
        metrics["skip_reason"] = "all_candidates_have_empty_reconstruction_targets"
        return Selection(None, None, metrics)
    # Stable ties use the earlier historical commitment.
    _, index, score, progress, extra = max(scored, key=lambda item: item[0])
    metrics.update({"selected_index": index, "selected_position": trajectory.commitments[index].position,
                    "selected_progress": progress, "selector_score": score,
                    "selector_abs_score": abs(score), **extra})
    return Selection(index, score, metrics)
