#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
command -v uv >/dev/null || { echo 'Install uv, then rerun scripts/setup.sh.' >&2; exit 1; }
uv sync --frozen --python 3.11
# Pinned HF remote code provides the architecture. The upstream source is kept
# for the reproducibility audit, rather than installing its optional DeepSpeed stack.
.venv/bin/python scripts/prepare.py
uv pip freeze --python .venv/bin/python > runtime/installed-requirements.txt
.venv/bin/python -m pytest
echo 'Setup and CPU checks passed. Next: export WANDB_API_KEY, then bash scripts/submit.sh.'
