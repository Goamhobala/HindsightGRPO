#!/usr/bin/env python3
"""Summarize saved exposure, conditional-sampling diagnostics and ordinary evaluations.

Read-only: this command never loads a model, runs generated code or submits jobs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from d1_exclusion.diagnostics import summarize_counterfactual

ARMS = ("d1_root", "entropy_no_ban", "entropy_soft_ban", "entropy_hard_ban")


def _json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def summarize_group(run_root):
    root = Path(run_root)
    arms = {}
    for arm in ARMS:
        directory = root / arm
        progress = _json(directory / "progress.json", {})
        evaluations = []
        for path in sorted((directory / "evaluation").glob("*/*/summary.json")):
            manifest = _json(path.parent / "manifest.json", {})
            evaluations.append({"label": path.parent.parent.name, "split": path.parent.name,
                                "metrics": _json(path), "manifest": manifest})
        arms[arm] = {
            "present": directory.is_dir(),
            "completed_prompts": int(progress.get("cursor", 0)),
            "applied_prompts": int(progress.get("applied_prompts", 0)),
            "collected_prompts": int(progress.get("collected_prompts", 0)),
            "optimizer_steps": int(progress.get("optimizer_steps", 0)),
            "completed_collections": int(progress.get("outer_collection", 0)),
            "inner_iteration": int(progress.get("inner_iteration", 0)),
            "complete": bool(progress.get("complete", False)),
            "wall_seconds": float(progress.get("wall_seconds", 0)),
            "applied_prompts_per_hour": (
                3600 * int(progress.get("applied_prompts", 0)) / float(progress["wall_seconds"])
                if float(progress.get("wall_seconds", 0)) > 0 else None),
            "costs": progress.get("costs", {}),
            "evaluations": evaluations,
        }
    common = min(arm["completed_prompts"] for arm in arms.values())
    common_applied = min(arm["applied_prompts"] for arm in arms.values())
    for arm, result in arms.items():
        records = []
        common_records = []
        completed_prefix = 0
        for path in sorted((root / arm / "collections").glob("[0-9]*.json")):
            collection = _json(path)
            batch = collection.get("records", [])
            records.extend(batch)
            # Only complete collections inside the common 12-iteration prefix.
            completed_prefix += len(batch)
            if completed_prefix <= common:
                common_records.extend(batch)
        result["counterfactual_saved_collections"] = summarize_counterfactual(records)
        result["counterfactual_common_completed_prefix"] = summarize_counterfactual(common_records)
    return {
        "run_root": str(root), "arms": arms,
        "largest_common_completed_prompts": common,
        "largest_common_applied_prompts": common_applied,
        "planned_prompts": 150, "planned_outer_collections": 38,
        "planned_optimizer_steps": 456,
        "interpretation": (
            "Completed exposure counts only prompts whose collection finished all 12 inner iterations. "
            "Applied exposure may include a partially optimized collection. Evaluation manifests identify "
            "the actual checkpoint exposure; differing exposures are not matched training comparisons. "
            "Elapsed process time is not measured GPU utilization or scheduler-billed GPU-hours."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, help="runtime/runs/d1-exclusion/GROUP")
    parser.add_argument("--output", help="Optional new JSON output file; refuses to overwrite")
    args = parser.parse_args(argv)
    report = summarize_group(args.run_root)
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        with Path(args.output).open("x") as stream:
            stream.write(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
