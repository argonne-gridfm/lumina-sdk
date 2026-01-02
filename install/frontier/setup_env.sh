#!/usr/bin/env bash
# setup_env.sh - Frontier environment setup for lumina-core using modules + miniforge
set -euo pipefail

hr() { printf '%*s\n' "${COLUMNS:-80}" '' | tr ' ' '='; }
banner() { hr; echo ">>> $1"; hr; }

banner "Starting lumina-core environment setup ($(date))"

# Module init (Frontier uses Lmod; keep graceful fallbacks)
if ! command -v module >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then
    source /etc/profile.d/modules.sh
  elif [[ -f /usr/share/lmod/lmod/init/bash ]]; then
    source /usr/share/lmod/lmod/init/bash
  elif [[ -f /usr/share/Modules/init/bash ]]; then
    source /usr/share/Modules/init/bash
  fi
fi

if command -v module >/dev/null 2>&1; then
  module reset
  ml cpe/24.07
  ml cce/18.0.0
  ml rocm/6.4.0
  ml amd-mixed/6.4.0
  ml craype-accel-amd-gfx90a
  ml PrgEnv-gnu
  ml miniforge3/23.11.0-0
  ml git-lfs
  module unload darshan-runtime || true
else
  echo "[warn] module command not found; proceeding without Frontier stack"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INSTALL_ROOT="${INSTALL_ROOT:-${PWD}/lumina-frontier-install}"
VENV_PATH="${VENV_PATH:-${INSTALL_ROOT}/.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
RECREATE_ENV="${RECREATE_ENV:-0}"

banner "Installation directories"
echo "Install root: ${INSTALL_ROOT}"
echo "Virtual env : ${VENV_PATH}"
mkdir -p "${INSTALL_ROOT}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[error] conda not found (load miniforge3 module)." >&2
  exit 1
fi

if [[ -d "${VENV_PATH}" && "${RECREATE_ENV}" -eq 1 ]]; then
  echo "Removing existing env at ${VENV_PATH}"
  conda env remove -p "${VENV_PATH}" -y || rm -rf "${VENV_PATH}"
fi

if [[ ! -d "${VENV_PATH}" ]]; then
  echo "Creating conda env at ${VENV_PATH} (Python ${PYTHON_VERSION})"
  conda create -y -p "${VENV_PATH}" python="${PYTHON_VERSION}"
fi

# shellcheck disable=SC1091
source activate "${VENV_PATH}"
python -m pip install --upgrade pip uv
export UV_LINK_MODE=copy

banner "Install lumina-core dependencies"
uv pip install -r "${REPO_ROOT}/install/frontier/requirements.in" \
  -c "${REPO_ROOT}/install/frontier/constraints.txt"

banner "Install lumina-core (editable, no extra deps)"
uv pip install -e "${REPO_ROOT}" --no-deps

banner "Done"
echo "Activate environment with: source activate ${VENV_PATH}"
