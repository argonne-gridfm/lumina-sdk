#!/usr/bin/env bash
# setup_env.sh - Minimal Frontier environment setup for lumina-core
set -euo pipefail

hr() { printf '%*s\n' "${COLUMNS:-80}" '' | tr ' ' '='; }
banner() { hr; echo ">>> $1"; hr; }

banner "Starting lumina-core environment setup ($(date))"

# Optional module init (Frontier uses Lmod, but script should degrade gracefully)
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
  module use /soft/modulefiles || true
  module load conda || true
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
  echo "[warn] conda not found; using system python for venv"
fi

if [[ -d "${VENV_PATH}" && "${RECREATE_ENV}" -eq 1 ]]; then
  echo "Removing existing env at ${VENV_PATH}"
  rm -rf "${VENV_PATH}"
fi

if [[ ! -d "${VENV_PATH}" ]]; then
  python -m venv "${VENV_PATH}" --upgrade-deps --prompt lumina
fi

source "${VENV_PATH}/bin/activate"
python -m pip install --upgrade pip uv
export UV_LINK_MODE=copy

banner "Install lumina-core dependencies"
uv pip install -r "${REPO_ROOT}/install/frontier/requirements.in" \
  -c "${REPO_ROOT}/install/frontier/constraints.txt"

banner "Install lumina-core (editable, no extra deps)"
uv pip install -e "${REPO_ROOT}" --no-deps

banner "Done"
echo "Activate environment with: source ${VENV_PATH}/bin/activate"
