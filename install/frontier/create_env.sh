#!/usr/bin/env bash
set -euo pipefail

# Default location can be overridden by passing a path argument
CONDA_ENV_DIR="${1:-${PWD}/.venv-frontier}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Frontier typically uses modules; load conda if available
if command -v module >/dev/null 2>&1; then
  module use /soft/modulefiles || true
  module load conda || true
fi

# Prefer an existing conda base if the module succeeds; otherwise rely on system python
if command -v conda >/dev/null 2>&1; then
  conda activate base
fi

echo "[frontier] Creating virtual environment at: ${CONDA_ENV_DIR}"
python -m venv "${CONDA_ENV_DIR}" --upgrade-deps
source "${CONDA_ENV_DIR}/bin/activate"

# Use uv for fast, reproducible installs
python -m pip install --upgrade uv
export UV_LINK_MODE=copy

uv pip install -r "${REPO_ROOT}/install/frontier/requirements.in" \
  -c "${REPO_ROOT}/install/frontier/constraints.txt"

# Install lumina-core in editable mode without re-pulling deps
uv pip install -e "${REPO_ROOT}" --no-deps

echo "[frontier] Environment setup complete. Activate with:"
echo "source ${CONDA_ENV_DIR}/bin/activate"
