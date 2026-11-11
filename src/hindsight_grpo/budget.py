"""A persisted runtime budget, separate from immutable scientific configuration."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time

from .config import stable_seed


@dataclass(frozen=True)
class RunBudget:
    training_prompts: int
    test_problems: int
    activated_unix: float
    training_deadline_unix: float
    deadline_unix: float
    version: int = 1

    def __post_init__(self):
        if self.version != 1:
            raise ValueError("Unsupported run budget version")
        if not 0 < self.training_prompts <= 374 or self.training_prompts % 4:
            raise ValueError("Training cutoff must be a multiple of four, at most 374")
        if not 0 < self.test_problems <= 500:
            raise ValueError("Test subset must contain 1–500 problems")
        if not self.activated_unix < self.training_deadline_unix < self.deadline_unix:
            raise ValueError("Budget deadlines must follow activation and reserve evaluation time")

    @property
    def label(self):
        return f"pilot-{self.training_prompts}"

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def test_rows(self, rows, seed):
        if len(rows) < self.test_problems:
            raise ValueError("Test split is smaller than the budgeted subset")
        # Selection uses task IDs and an isolated RNG, never model outputs/scores.
        ids = sorted(row["task_id"] for row in rows)
        chosen = set(random.Random(stable_seed(seed, "pilot-test-subset", self.version))
                     .sample(ids, self.test_problems))
        return [row for row in rows if row["task_id"] in chosen]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", required=True)
    parser.add_argument("--run-root", default="runtime/runs")
    parser.add_argument("--training-prompts", type=int, default=200)
    parser.add_argument("--test-problems", type=int, default=100)
    parser.add_argument("--training-hours", type=float, default=36)
    parser.add_argument("--total-hours", type=float, default=48)
    args = parser.parse_args()
    if Path(args.group).name != args.group or args.group in (".", ".."):
        parser.error("Group must be a directory name")
    path = Path(args.run_root) / args.group / "run_budget.json"
    if path.exists():
        budget = RunBudget.load(path)
        requested = (args.training_prompts, args.test_problems,
                     args.training_hours, args.total_hours)
        existing = (budget.training_prompts, budget.test_problems,
                    (budget.training_deadline_unix - budget.activated_unix) / 3600,
                    (budget.deadline_unix - budget.activated_unix) / 3600)
        if requested != existing:
            parser.error("A different budget already exists; deadlines cannot be reset on resume")
        print(f"Keeping existing budget and deadlines: {path}")
        return
    activated = time.time()
    budget = RunBudget(args.training_prompts, args.test_problems, activated,
                       activated + args.training_hours * 3600,
                       activated + args.total_hours * 3600)
    # Activate once, before submitting resumptions. Scientific config stays intact.
    from .checkpoint import atomic_json
    atomic_json(path, asdict(budget))
    print(f"Activated {budget.label}: {path}")
    print("Resume the same group. The budget and deadlines apply to every arm and successor.")


if __name__ == "__main__":
    main()
