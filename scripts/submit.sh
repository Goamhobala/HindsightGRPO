#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
: "${WANDB_API_KEY:?Export WANDB_API_KEY before submission; its value is never written to the repository.}"
for command in sbatch sinfo scontrol sacctmgr; do
  command -v "$command" >/dev/null || { echo "Missing cluster requirement: $command" >&2; exit 1; }
done
[[ -f runtime/metadata/inputs_manifest.json ]] || { echo 'Run bash scripts/setup.sh first.' >&2; exit 1; }
[[ -f .venv/bin/python ]] || { echo 'Python environment missing.' >&2; exit 1; }
.venv/bin/python scripts/install_bwrap.py --check
.venv/bin/python scripts/check_model_imports.py --offline
# Preserve actual resource evidence in scratch. Do not modify the experiment if
# the expected allocation is unavailable: fail on configuration, queue on load.
sinfo -p l40s -o '%P %a %l %D %G' > runtime/preflight/partition.txt
scontrol show partition l40s -o > runtime/preflight/partition-detail.txt
sacctmgr -nP show association user="$USER" account=l40sfree format=Account,QOS,Partition > runtime/preflight/association.txt
sacctmgr -nP show qos l40sfree format=Name,MaxWall > runtime/preflight/qos.txt
.venv/bin/python scripts/cluster_preflight.py
export HINDSIGHT_TIME_LIMIT
HINDSIGHT_TIME_LIMIT="$(.venv/bin/python scripts/time_limit.py)"
export EXPERIMENT_GROUP="${EXPERIMENT_GROUP:-hindsight-$(date -u +%Y%m%dT%H%M%SZ)}"
.venv/bin/python -m pytest
# Authentication/network check is online; no model work is queued if it fails.
.venv/bin/python scripts/check_wandb.py
for arm in root entropy_cf r_cf gradient_ce_cf; do
  job_id="$(sbatch --parsable --job-name="$arm" --time="$HINDSIGHT_TIME_LIMIT" \
      --export=ALL slurm/run.sh "$arm" "$EXPERIMENT_GROUP")"
  printf '%s\t%s\t%s\n' "$arm" "$job_id" "$EXPERIMENT_GROUP" | tee -a runtime/logs/submissions.tsv
done
echo "Queued four arms in group $EXPERIMENT_GROUP. SLURM and W&B progress: runtime/logs and runtime/runs."
