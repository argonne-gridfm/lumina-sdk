#!/usr/bin/bash
set -euo pipefail

# $CFS/amsc004/conda_envs/lumina
CONDA_ENV_DIR="${1:-${PWD}/.venv-perlmutter}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

module load conda

echo "[perlmutter] Creating conda env at: ${CONDA_ENV_DIR}"

conda create --prefix ${CONDA_ENV_DIR} python=3.12
conda activate ${CONDA_ENV_DIR}

python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade uv

export UV_LINK_MODE=copy
uv pip install -r "${REPO_ROOT}/install/perlmutter/requirements.in" \
  -c "${REPO_ROOT}/install/perlmutter/constraints.txt"

uv pip install -e "${REPO_ROOT}" --no-deps

echo "[perlmutter] Conda env setup complete."
