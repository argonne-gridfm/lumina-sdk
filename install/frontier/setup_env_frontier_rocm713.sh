#!/usr/bin/env bash
# setup_env_frontier_rocm713.sh - Frontier environment setup for lumina-sdk with ROCm 7.13.0
#
# Companion to setup_env_frontier_rocm711.sh. This variant targets the ROCm/7.13.0
# module (AMD's "TheRock" pre-release, added to Frontier on 2026-07-01) and installs
# the AMD-published gfx90a PyTorch wheels, per AMD collaborator guidance to move off
# ROCm 7.1.1 (whose bundled RCCL 2.27.7 crashes MI250X collectives with
# HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION).
set -euo pipefail

hr() { printf '%*s\n' "${COLUMNS:-80}" '' | tr ' ' '='; }
banner() { hr; echo ">>> $1"; hr; }
subbanner() { echo "-- $1"; }

banner "Starting lumina-sdk environment setup - ROCm 7.13.0 ($(date))"

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
  # Clear any conda environment inherited from the launching shell. Otherwise the
  # miniforge module's deactivate hook runs `conda deactivate` in this
  # non-interactive shell and fails ("Run 'conda init' before 'conda deactivate'"),
  # which aborts the script under `set -e`.
  unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_EXE CONDA_PYTHON_EXE \
        CONDA_PROMPT_MODIFIER 2>/dev/null || true
  module reset
  ml PrgEnv-gnu/8.7.0
  ml cpe/26.03
  ml miniforge3/23.11.0-0
  ml rocm/7.13.0
  ml rccl-net-plugin
  ml craype-accel-amd-gfx90a
  ml git-lfs
  module unload darshan-runtime || true
else
  echo "[warn] module command not found; proceeding without Frontier stack"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Default to a sibling of the repo (matches the rocm711 convention), so the venv
# path is stable regardless of the current working directory.
INSTALL_ROOT="${INSTALL_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)/lumina-frontier-rocm713-install}"
VENV_PATH="${VENV_PATH:-${INSTALL_ROOT}/.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
RECREATE_ENV="${RECREATE_ENV:-0}"
# Set BUILD_PYG_FROM_SOURCE=1 to rebuild the PyG C++/HIP extensions from the ROCm
# forks against the installed torch 2.11+rocm7.13 (fixes the prebuilt-wheel ABI
# mismatch: "libpyg.so: undefined symbol: _ZNK5torch8autograd4Node4nameEv").
BUILD_PYG_FROM_SOURCE="${BUILD_PYG_FROM_SOURCE:-0}"

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
  conda env remove -p "${VENV_PATH}" -y
fi

if [[ ! -d "${VENV_PATH}" ]]; then
  echo "Creating conda env at ${VENV_PATH} (Python ${PYTHON_VERSION})"
  conda create -y -p "${VENV_PATH}" python="${PYTHON_VERSION}" -c conda-forge
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

EXPECTED_ROCM_MM="7.13"
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

# AMD-published gfx90a wheels for ROCm 7.13.0 (per AMD collaborator guidance).
# These bundle an RCCL build matched to ROCm 7.13, replacing the crashing
# RCCL 2.27.7 shipped with the ROCm 7.1 torch wheel.
AMD_ROCM_INDEX_URL="https://repo.amd.com/rocm/whl/gfx90a/"
subbanner "Install AMD gfx90a PyTorch 2.11 wheels for ROCm 7.13.0"
pip_retry --index-url "${AMD_ROCM_INDEX_URL}" \
  "torch==2.11.0+rocm7.13.0" \
  "torchvision==0.26.0+rocm7.13.0" \
  "torchaudio==2.11.0+rocm7.13.0"

python - <<PY
import torch
print("torch.__version__ =", torch.__version__)
print("torch.version.hip =", torch.version.hip)
print("arch list =", torch.cuda.get_arch_list())
PY

# OLCF provides ROCm-specific PyG extension packages for Frontier.
# NOTE: the *-rocm PyG wheels may be built against a specific torch/ROCm combo;
# if these fail to resolve for torch 2.11/rocm7.13, they may need to be built
# from source or pulled from an OLCF-provided index matching this ROCm version.
subbanner "Install PyTorch Geometric stack"
pip_retry ninja packaging scipy
if [[ "${BUILD_PYG_FROM_SOURCE}" -eq 1 ]]; then
  subbanner "Removing prebuilt PyG wheels (ABI-incompatible with torch 2.11+rocm7.13)"
  # The prebuilt gfx90a wheels — especially pyg_lib — are linked against a
  # different libtorch ABI and fail to load with:
  #   libpyg.so: undefined symbol: _ZNK5torch8autograd4Node4nameEv
  # torch_sparse.typing does `import pyg_lib` and only catches ImportError, so a
  # leftover broken pyg_lib becomes a hard OSError. Uninstall the whole prebuilt
  # set (plain and -rocm names) before rebuilding from source. pyg_lib is
  # intentionally NOT rebuilt: torch_geometric and the source-built torch_sparse
  # work without it (accelerated sampling ops only).
  python -m pip uninstall -y \
    pyg-lib pyg-lib-rocm \
    torch-scatter torch-scatter-rocm \
    torch-sparse torch-sparse-rocm \
    torch-cluster torch-cluster-rocm \
    torch-spline-conv torch-spline-conv-rocm 2>/dev/null || true

  subbanner "Building torch-scatter/sparse/cluster/spline-conv from ROCm forks"
  PYG_FRONTIER="${INSTALL_ROOT}/PyTorch-Geometric-${ROCM_MM}"
  mkdir -p "${PYG_FRONTIER}"
  pushd "${PYG_FRONTIER}" >/dev/null
  build_pyg() {  # name repo_url ref
    local name="$1" url="$2" ref="$3"
    if [[ ! -d "${name}/.git" ]]; then git clone --recursive "${url}" "${name}"; fi
    pushd "${name}" >/dev/null
    git fetch --all; git checkout "${ref}"; git submodule update --init --recursive
    rm -rf build
    CC=gcc CXX=g++ PYTORCH_ROCM_ARCH=gfx90a FORCE_CUDA=1 python setup.py build
    CC=gcc CXX=g++ PYTORCH_ROCM_ARCH=gfx90a FORCE_CUDA=1 python setup.py install
    popd >/dev/null
  }
  build_pyg pytorch_scatter     https://github.com/Looong01/pytorch_scatter-rocm.git 9799c51
  build_pyg pytorch_sparse      https://github.com/Looong01/pytorch_sparse-rocm.git  2340737
  build_pyg pytorch_cluster     https://github.com/rusty1s/pytorch_cluster.git        1.6.3-11-g4126a52
  build_pyg pytorch_spline_conv https://github.com/rusty1s/pytorch_spline_conv.git    1.2.2-9-ga6d1020
  popd >/dev/null
  pip_retry torch-geometric
else
  subbanner "Install ROCm PyTorch Geometric wheels (gfx90a)"
  pip_retry torch-geometric torch-sparse-rocm torch-spline-conv-rocm \
    torch-scatter-rocm torch-cluster-rocm pyg-lib-rocm
fi

banner "Install core lumina-sdk Python packages"
pip_retry numpy pandas scipy networkx joblib pyyaml h5py pydantic tqdm
pip_retry pandapower pypower matpowercaseframes wandb optuna lightning

banner "Install lumina-sdk (editable, no extra deps)"
pip_retry -e "${REPO_ROOT}" --no-deps

banner "Validate installation"
python - <<'PY'
import torch
import torch_geometric
import h5py
import pydantic
import lumina

print("torch:", torch.__version__)
print("HIP:", torch.version.hip)
print("torch-geometric:", torch_geometric.__version__)
print("GPU visible:", torch.cuda.is_available())
print("lumina import: OK")
PY

banner "Done"
echo "Activate environment with: source activate ${VENV_PATH}"
