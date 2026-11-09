"""Probe reward isolation in a compute allocation before loading model weights."""
import argparse
import json
import os
from pathlib import Path

from hindsight_grpo.checkpoint import atomic_json
from hindsight_grpo.rewards import RewardRunner, SandboxUnavailable, sandbox_diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runtime/preflight/sandbox.json")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Run this probe inside a SLURM compute allocation.")
    report = {"slurm_job_id": os.environ["SLURM_JOB_ID"], "resources": sandbox_diagnostics()}
    runner = RewardRunner(temporary_dir=Path("runtime/tmp"))
    try:
        report["sandbox"] = runner.check_available()
        report["status"] = "passed"
    except SandboxUnavailable as error:
        report.update(status="failed", error=str(error))
        atomic_json(args.output, report)
        raise SystemExit(f"Sandbox probe failed; diagnostics: {args.output}\n{error}") from error
    atomic_json(args.output, report)
    print(f"Sandbox execution probe passed; evidence: {args.output}")


if __name__ == "__main__":
    main()
