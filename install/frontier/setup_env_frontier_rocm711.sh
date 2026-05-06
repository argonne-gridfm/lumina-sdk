#!/usr/bin/env bash
# setup_env_frontier_rocm711.sh - Frontier environment setup for lumina-core with ROCm 7.1.1
set -euo pipefail

hr() { printf '%*s\n' "${COLUMNS:-80}" '' | tr ' ' '='; }
banner() { hr; echo ">>> $1"; hr; }
subbanner() { echo "-- $1"; }

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
  ml rocm/7.1.1
  ml amd-mixed/7.1.1
  ml craype-accel-amd-gfx90a
  ml PrgEnv-gnu
  ml miniforge3/23.11.0-0
  ml git-lfs
  module unload darshan-runtime || true
else
  echo "[warn] module command not found; proceeding without Frontier stack"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INSTALL_ROOT="${INSTALL_ROOT:-${PWD}/lumina-frontier-rocm711-install}"
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
python -m pip install --upgrade pip setuptools wheel

# pip with retries
PIP_FLAGS=(--upgrade-strategy only-if-needed)
pip_retry() {
  local tries=3 delay=3
  for ((i=1; i<=tries; i++)); do
    if pip install "${PIP_FLAGS[@]}" "$@"; then
      return 0
    fi
    echo "pip install failed (attempt $i/$tries). Retrying in ${delay}s..."
    sleep "$delay"; delay=$((delay*2))
  done
  return 1
}

# ROCm detection and torch install
detect_rocm_mm() {
  local v=""
  if command -v module >/dev/null 2>&1; then
    local mlist
    mlist="$(module -t list 2>&1 || true)"
    v="$(grep -Eo 'rocm/[0-9]+\.[0-9]+' <<<"$mlist" | head -n1 | sed 's#rocm/##')"
  fi
  if [[ -z "$v" ]] && command -v hipcc >/dev/null 2>&1; then
    v="$(hipcc --version 2>&1 | grep -Eo 'HIP version:\s*[0-9]+\.[0-9]+' | grep -Eo '[0-9]+\.[0-9]+' | head -n1 || true)"
  fi
  echo "$v"
}

EXPECTED_ROCM_MM="7.1"
ROCM_MM="${ROCM_MM:-$(detect_rocm_mm)}"
if [[ -z "$ROCM_MM" ]]; then
  echo "❌ Could not detect ROCm version. Ensure the rocm module is loaded."
  exit 1
fi
echo "Detected ROCm: $ROCM_MM"
if [[ "$ROCM_MM" != "$EXPECTED_ROCM_MM" ]]; then
  echo "❌ ROCm version mismatch. Detected $ROCM_MM but expecting rocm${EXPECTED_ROCM_MM}."
  exit 1
fi

PYTORCH_ROCM_INDEX_URL="https://download.pytorch.org/whl/rocm${EXPECTED_ROCM_MM}"
subbanner "Install ROCm torch from ${PYTORCH_ROCM_INDEX_URL}"
pip_retry --index-url "${PYTORCH_ROCM_INDEX_URL}" "torch" "torchvision"

python - <<PY
import torch
print("torch.__version__ =", torch.__version__)
print("torch.version.hip =", torch.version.hip)
PY

# PyTorch Geometric: ROCm wheels matching installed torch
TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
PYG_ROCM_URL="https://data.pyg.org/whl/torch-${TORCH_VERSION}+rocm${EXPECTED_ROCM_MM}.html"
subbanner "Install torch-geometric from ${PYG_ROCM_URL}"
pip_retry torch-geometric -f "${PYG_ROCM_URL}"

banner "Install core lumina-core Python packages"
pip_retry numpy pandas scipy networkx joblib pyyaml
pip_retry pandapower wandb optuna lightning

banner "Install lumina-core (editable, no extra deps)"
pip_retry -e "${REPO_ROOT}" --no-deps

banner "Done"
echo "Activate environment with: source activate ${VENV_PATH}"
