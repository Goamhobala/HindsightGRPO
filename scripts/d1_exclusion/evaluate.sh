#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help || $# -lt 2 || $# -gt 4 ]]; then
  echo 'Usage: bash scripts/d1_exclusion/evaluate.sh GROUP ARM [validation|test] [final|latest|pretrained|PATH]'; exit 0
fi
group="$1"; arm="$2"; split="${3:-test}"; checkpoint="${4:-final}"
source "$(dirname "$0")/common.sh"
d1_validate_group "$group"
config="$(d1_config_for_arm "$arm")"
case "$split" in validation|test) ;; *) echo 'Expected validation or test split.' >&2; exit 2 ;; esac
source "$(dirname "$0")/../env.sh"
d1_require_environment
run_dir="runtime/runs/d1-exclusion/$group/$arm"
[[ -d "$run_dir" ]] || { echo "Run directory does not exist: $run_dir" >&2; exit 1; }
case "$checkpoint" in
  final|latest) checkpoint="$run_dir/checkpoints/$checkpoint.pt" ;;
  pretrained) ;;
  *) [[ "$checkpoint" != -* ]] || { echo 'Invalid checkpoint path.' >&2; exit 2; } ;;
esac
[[ "$checkpoint" == pretrained || -f "$checkpoint" ]] || { echo "Checkpoint does not exist: $checkpoint" >&2; exit 1; }
d1_submit "$config" "$run_dir" evaluate "runtime/logs/d1-exclusion/$group/$arm-evaluation" "$checkpoint" "$split"
