#!/usr/bin/bash
set -euo pipefail

CONDA_ENV_DIR="${1:-${PWD}/.venv-polaris}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

module use /soft/modulefiles
module load conda
conda activate base

echo "[polaris] Creating conda env at: ${CONDA_ENV_DIR}"

python -m venv ${CONDA_ENV_DIR} --system-site-packages
source ${CONDA_ENV_DIR}/bin/activate

export UV_LINK_MODE=copy
uv pip install -r "${REPO_ROOT}/install/polaris/requirements.in" \
  -c "${REPO_ROOT}/install/polaris/constraints.txt"

uv pip install -e "${REPO_ROOT}" --no-deps

echo "[polaris] Conda env setup complete."