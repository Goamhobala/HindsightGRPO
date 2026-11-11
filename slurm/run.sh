#!/usr/bin/env bash
#SBATCH --partition=l40s
#SBATCH --account=l40sfree
#SBATCH --qos=l40sfree
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=yhxjin001@myuct.ac.za
#SBATCH --mail-type=ALL
#SBATCH --signal=B:USR1@1200
#SBATCH --requeue
#SBATCH --output=runtime/logs/%x-%j.out
#SBATCH --error=runtime/logs/%x-%j.err
set -euo pipefail
# SLURM executes a spooled copy of this script, so $0 is not the repo path.
cd "$HOME"
cd '../../scratch/yhxjin001/HindsightGRPO'
source scripts/env.sh
arm="${1:?arm required}"
group="${2:?experiment group required}"
case "$arm" in root|entropy_cf|r_cf|gradient_ce_cf) ;; *) exit 2 ;; esac
: "${WANDB_API_KEY:?WANDB_API_KEY must be inherited from the submission shell}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
run_child=0
successor_scheduled=0
stop_requested=0
budget_file="runtime/runs/$group/run_budget.json"
if [[ -f "$budget_file" ]]; then
  budget_expired="$(.venv/bin/python -c 'import json,sys,time; print(int(time.time() >= json.load(open(sys.argv[1]))["deadline_unix"]))' "$budget_file")"
  if [[ "$budget_expired" == 1 ]]; then
    echo "Run budget has expired for $group; no GPU work will start."
    exit 0
  fi
fi
schedule_successor() {
  if [[ "$stop_requested" == 1 ]]; then return; fi
  if [[ "$successor_scheduled" == 0 ]]; then
    successor_scheduled=1
    next_id="$(sbatch --parsable --dependency="afterany:$SLURM_JOB_ID" \
      --job-name="$arm" --time="$HINDSIGHT_TIME_LIMIT" --export=ALL slurm/run.sh "$arm" "$group")"
    printf '%s\t%s\t%s\tpredecessor=%s\n' "$arm" "$next_id" "$group" "$SLURM_JOB_ID" >> runtime/logs/submissions.tsv
  fi
  if [[ "$run_child" != 0 ]]; then kill -USR1 "$run_child" 2>/dev/null || true; fi
}
forward_stop() {
  stop_requested=1
  if [[ "$run_child" != 0 ]]; then kill -TERM "$run_child" 2>/dev/null || true; fi
}
trap schedule_successor USR1
trap forward_stop TERM INT
run_python() {
  if [[ "$successor_scheduled" == 1 || "$stop_requested" == 1 ]]; then return 75; fi
  .venv/bin/python -u "$@" &
  run_child=$!
  set +e
  wait "$run_child"
  result=$?
  while kill -0 "$run_child" 2>/dev/null; do wait "$run_child"; result=$?; done
  set -e
  run_child=0
  if [[ ( "$successor_scheduled" == 1 || "$stop_requested" == 1 ) && "$result" == 0 ]]; then return 75; fi
  return "$result"
}
report="runtime/preflight/$group/$arm.json"
mkdir -p "$(dirname "$report")"
run_python scripts/install_bwrap.py --check
run_python scripts/check_sandbox.py --output "runtime/preflight/$group/$arm-sandbox.json"
run_python scripts/gpu_preflight.py --output "$report"
# One real, full-size optimizer update per arm is a correctness gate, saved in
# its own W&B group/checkpoint directory. Full training starts from pretrained.
marker="runtime/runs/$group/verification/$arm/verification_passed.json"
if ! .venv/bin/python -c 'import json,sys; from pathlib import Path; from hindsight_grpo.config import config_digest,load_config; p=Path(sys.argv[1]); sys.exit(0 if p.exists() and json.loads(p.read_text()).get("config_digest")==config_digest(load_config("configs/experiment.json")) else 1)' "$marker"; then
  run_python -m hindsight_grpo.train --arm "$arm" --group "$group" --verify-one-update
fi
run_python -m hindsight_grpo.train --arm "$arm" --group "$group"
