#!/usr/bin/env bash
set -euo pipefail
group="d1-exclusion-$(date -u +%Y%m%dT%H%M%SZ)-$$"
dry_run=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --group) group="${2:?--group needs a value}"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    --help) echo 'Usage: bash scripts/d1_exclusion/submit_all.sh [--group NAME] [--dry-run]'; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
source "$(dirname "$0")/common.sh"
d1_validate_group "$group"
arms=(d1_root entropy_no_ban entropy_soft_ban entropy_hard_ban)
if [[ "$dry_run" == 1 ]]; then
  for arm in "${arms[@]}"; do
    printf 'prepare %s: %s -> runtime/runs/d1-exclusion/%s/%s (24h, l40s/l40sfree, one GPU)\n' "$arm" "$(d1_config_for_arm "$arm")" "$group" "$arm"
  done
  exit 0
fi
source "$(dirname "$0")/../env.sh"
d1_require_environment
run_root="runtime/runs/d1-exclusion/$group"
# Atomic reservation makes repeated submission fail before queuing any jobs.
mkdir -p runtime/runs/d1-exclusion
mkdir "$run_root" || { echo 'Group already exists; select a new group or use resume.sh.' >&2; exit 1; }
printf '%s\n' "$group" > "$run_root/group.txt"
for arm in "${arms[@]}"; do
  d1_submit "$(d1_config_for_arm "$arm")" "$run_root/$arm" train "runtime/logs/d1-exclusion/$group/$arm"
done
echo "Submitted four arms in $group. Resume incomplete arms manually with resume.sh."
