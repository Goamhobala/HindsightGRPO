#!/usr/bin/env bash
#SBATCH --partition=l40s
#SBATCH --account=l40sfree
#SBATCH --qos=l40sfree
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@900
set -euo pipefail
# SLURM runs a spooled copy; preserve the intentional HOME-relative HPC path.
cd "$HOME"
cd '../../scratch/yhxjin001/HindsightGRPO'
source scripts/env.sh
config="${1:?configuration required}"; run_dir="${2:?run directory required}"; mode="${3:-train}"
case "$run_dir" in runtime/runs/d1-exclusion/*) ;; *) echo 'Refusing a run directory outside the isolated d1 experiment.' >&2; exit 2 ;; esac
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
child=0
stop_requested=0
forward_stop() {
  stop_requested=1
  if [[ "$child" != 0 ]]; then kill -USR1 "$child" 2>/dev/null || true; fi
}
trap forward_stop USR1 TERM INT
run_python() {
  if [[ "$stop_requested" == 1 ]]; then return 75; fi
  .venv/bin/python -u "$@" &
  child=$!
  set +e
  wait "$child"
  result=$?
  while kill -0 "$child" 2>/dev/null; do wait "$child"; result=$?; done
  set -e
  child=0
  if [[ "$stop_requested" == 1 && "$result" == 0 ]]; then return 75; fi
  return "$result"
}
run_python scripts/install_bwrap.py --check
run_python scripts/check_sandbox.py --output "$run_dir/preflight/sandbox-${SLURM_JOB_ID}.json"
case "$mode" in
  train) [[ $# == 3 ]] || exit 2; run_python -m d1_exclusion.trainer --config "$config" --run-dir "$run_dir" ;;
  resume) [[ $# == 3 ]] || exit 2; run_python -m d1_exclusion.trainer --config "$config" --run-dir "$run_dir" --resume ;;
  evaluate) [[ $# == 5 ]] || exit 2; run_python -m d1_exclusion.evaluation --config "$config" --run-dir "$run_dir" --checkpoint "$4" --split "$5" ;;
  *) echo 'Expected train, resume or evaluate mode.' >&2; exit 2 ;;
esac
