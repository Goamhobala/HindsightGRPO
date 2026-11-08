#!/usr/bin/env bash
# Source from the repository; always anchor the requested relative path at HPC home.
export PROJECT_REL='../../scratch/yhxjin001/HindsightGRPO'
if [[ "${HINDSIGHT_LOCAL:-0}" != 1 ]]; then
  cd "$HOME"
  cd "$PROJECT_REL"
else
  cd "$(dirname "${BASH_SOURCE[0]}")/.."
fi
export HF_HOME="$PWD/runtime/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_MODULES_CACHE="$HF_HOME/modules"
export UV_CACHE_DIR="$PWD/runtime/uv-cache"
export UV_PYTHON_INSTALL_DIR="$PWD/runtime/uv-python"
export PIP_CACHE_DIR="$PWD/runtime/pip-cache"
export TRITON_CACHE_DIR="$PWD/runtime/triton"
export MPLCONFIGDIR="$PWD/runtime/matplotlib"
export WANDB_DIR="$PWD/runtime/wandb"
export WANDB_DATA_DIR="$PWD/runtime/wandb-data"
export WANDB_ARTIFACT_DIR="$PWD/runtime/wandb-artifacts"
export WANDB_CONFIG_DIR="$PWD/runtime/wandb-config"
export WANDB_CACHE_DIR="$PWD/runtime/wandb-cache"
export XDG_CACHE_HOME="$PWD/runtime/xdg-cache"
export TORCH_HOME="$PWD/runtime/torch"
export TMPDIR="$PWD/runtime/tmp"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$UV_CACHE_DIR" "$TMPDIR" \
  "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_ARTIFACT_DIR" "$WANDB_CONFIG_DIR" \
  "$WANDB_CACHE_DIR" "$HF_MODULES_CACHE" "$XDG_CACHE_HOME" "$TORCH_HOME" \
  "$MPLCONFIGDIR" runtime/logs runtime/runs runtime/preflight
if [[ "${WANDB_MODE:-online}" != online && "${WANDB_MODE:-online}" != run ]]; then
  echo 'ERROR: WANDB_MODE must be online for this experiment.' >&2
  return 1
fi
export WANDB_MODE=online
export WANDB_PROJECT="${WANDB_PROJECT:-HindsightGRPO}"
