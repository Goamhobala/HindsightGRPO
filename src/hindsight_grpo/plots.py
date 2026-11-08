"""Publishable comparison plots from durable, paired per-arm JSONL records."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import ARMS


def read_metrics(path):
    rows = []
    if not path.exists():
        return rows
    with path.open() as stream:
        for line in stream:
            try:
                rows.append(json.loads(line)["payload"])
            except json.JSONDecodeError:
                # A concurrently running arm may be in the middle of one write.
                continue
    return rows


def make_plots(run_root, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    figures = {}
    for name in ("validation_by_update", "validation_by_gpu_hours", "root_reward", "counterfactual_delta"):
        fig, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        figures[name] = (fig, axis)
    comparison = {}
    for arm in ARMS:
        metrics = read_metrics(Path(run_root) / arm / "metrics.jsonl")
        validation, rewards, deltas = {}, {}, {}
        gpu_hours = 0.
        for row in metrics:
            gpu_hours = row.get("compute/gpu_hours", gpu_hours)
            update = row["update"]
            if "validation/accuracy" in row:
                validation[update] = (gpu_hours, row["validation/accuracy"])
            if "root/reward_mean" in row:
                rewards[update] = row["root/reward_mean"]
            if "cf/delta" in row:
                deltas[update] = row["cf/delta"]
        if validation:
            updates = sorted(validation)
            figures["validation_by_update"][1].plot(updates, [validation[u][1] for u in updates], label=arm)
            figures["validation_by_gpu_hours"][1].plot([validation[u][0] for u in updates],
                                                      [validation[u][1] for u in updates], label=arm)
        for name, values in (("root_reward", rewards), ("counterfactual_delta", deltas)):
            if values:
                updates = sorted(values)
                figures[name][1].plot(updates, [values[u] for u in updates], label=arm, alpha=.85)
        comparison[arm] = {"validation": validation, "root_reward": rewards, "counterfactual_delta": deltas}
    labels = {"validation_by_update": ("Optimizer updates", "Validation binary accuracy"),
              "validation_by_gpu_hours": ("Cumulative allocated GPU hours", "Validation binary accuracy"),
              "root_reward": ("Optimizer updates", "Mean ordinary root reward"),
              "counterfactual_delta": ("Optimizer updates", "Mean alternative − keep reward")}
    paths = []
    for name, (fig, axis) in figures.items():
        axis.set(xlabel=labels[name][0], ylabel=labels[name][1])
        axis.grid(alpha=.2)
        if axis.lines:
            axis.legend(frameon=False)
        for suffix in ("png", "pdf"):
            path = output / f"{name}.{suffix}"
            fig.savefig(path, dpi=180)
            paths.append(path)
        plt.close(fig)
    data_path = output / "comparison.json"
    data_path.write_text(json.dumps(comparison, indent=2) + "\n")
    return paths + [data_path]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, help=".../<group>/training directory")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    for path in make_plots(args.run_root, args.output):
        print(path)


if __name__ == "__main__":
    main()
