#!/usr/bin/env bash
# Shared operational checks. Sourcing this file never submits a job.
d1_config_for_arm() {
  case "$1" in
    d1_root) printf '%s\n' configs/d1_exclusion/root.json ;;
    entropy_no_ban) printf '%s\n' configs/d1_exclusion/no_ban.json ;;
    entropy_soft_ban) printf '%s\n' configs/d1_exclusion/soft_ban.json ;;
    entropy_hard_ban) printf '%s\n' configs/d1_exclusion/hard_ban.json ;;
    *) echo "Unknown d1 exclusion arm: $1" >&2; return 2 ;;
  esac
}
d1_validate_group() {
  [[ "$1" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || {
    echo 'Group must contain only letters, digits, underscores and hyphens.' >&2; return 2;
  }
}
d1_require_environment() {
  [[ -x .venv/bin/python ]] || { echo 'The existing .venv is missing; use the repository setup first.' >&2; return 1; }
  [[ -f runtime/metadata/inputs_manifest.json ]] || { echo 'Pinned model/data cache is unprepared; use the existing repository setup first.' >&2; return 1; }
  : "${WANDB_API_KEY:?Export WANDB_API_KEY before submitting.}"
  command -v sbatch >/dev/null || { echo 'SLURM sbatch is unavailable.' >&2; return 1; }
  .venv/bin/python scripts/install_bwrap.py --check
  .venv/bin/python -c 'from d1_exclusion.config import load_config; import sys; [load_config(p) for p in sys.argv[1:]]' configs/d1_exclusion/{root,no_ban,soft_ban,hard_ban}.json
}
d1_submit() {
  local config="$1" run_dir="$2" mode="$3" log_dir="$4"
  shift 4
  mkdir -p "$log_dir"
  local job_id
  job_id="$(sbatch --parsable --job-name="d1-$(basename "$run_dir")" \
    --chdir="$PWD" --output="$PWD/$log_dir/%j.out" --error="$PWD/$log_dir/%j.err" \
    --export=ALL slurm/d1_exclusion/train.sbatch "$config" "$run_dir" "$mode" "$@")"
  printf '%s\t%s\t%s\t%s\n' "$mode" "$run_dir" "$job_id" "$config" | tee -a "$(dirname "$log_dir")/submissions.tsv"
}
