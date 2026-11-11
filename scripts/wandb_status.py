"""Export small, read-only W&B summaries without cluster/VPN access."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def snapshot(api, entity, project, group):
    rows = []
    for run in api.runs(f"{entity}/{project}", filters={"group": group}):
        summary = dict(run.summary)
        selected = {key: value for key, value in summary.items()
                    if key.startswith(("progress/", "checkpoint/", "budget/", "protocol/", "scheduler/"))
                    or key in ("latest_update", "train/prompts_processed", "complete", "budget_exhausted",
                               "_timestamp", "compute/gpu_hours")}
        rows.append({"name": run.name, "id": run.id, "state": run.state,
                     "url": run.url, "summary": selected})
    return sorted(rows, key=lambda row: row["name"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", default="HindsightGRPO")
    parser.add_argument("--group", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    import wandb
    try:
        rows = snapshot(wandb.Api(timeout=30), args.entity, args.project, args.group)
    except Exception as error:
        # Do not echo exception text, request headers, or credentials.
        raise SystemExit(f"W&B query failed ({type(error).__name__}). "
                         "Run .venv/bin/wandb login locally, then retry. Do not paste API keys into chat.") from None
    result = json.dumps(rows, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result)
        print(f"Saved {len(rows)} run summaries to {args.output}")
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
