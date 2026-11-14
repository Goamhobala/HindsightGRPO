#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help || $# -lt 1 || $# -gt 2 ]]; then
  echo 'Usage: bash scripts/d1_exclusion/resume.sh GROUP [ARM|all]'; exit 0
fi
group="$1"; selected_arm="${2:-all}"
source "$(dirname "$0")/common.sh"
d1_validate_group "$group"
if [[ "$selected_arm" == all ]]; then
  arms=(d1_root entropy_no_ban entropy_soft_ban entropy_hard_ban)
else
  d1_config_for_arm "$selected_arm" >/dev/null
  arms=("$selected_arm")
fi
source "$(dirname "$0")/../env.sh"
d1_require_environment
for arm in "${arms[@]}"; do
  run_dir="runtime/runs/d1-exclusion/$group/$arm"
  [[ -f "$run_dir/checkpoints/latest.pt" ]] || { echo "Missing checkpoint: $run_dir/checkpoints/latest.pt" >&2; exit 1; }
done
for arm in "${arms[@]}"; do
  run_dir="runtime/runs/d1-exclusion/$group/$arm"
  if [[ -f "$run_dir/completed.json" ]]; then echo "Already complete: $arm"; continue; fi
  d1_submit "$(d1_config_for_arm "$arm")" "$run_dir" resume "runtime/logs/d1-exclusion/$group/$arm"
done
