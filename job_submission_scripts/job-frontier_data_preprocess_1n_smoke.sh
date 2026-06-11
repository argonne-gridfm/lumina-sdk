#!/bin/bash
# Frontier data preprocessing smoke test (1 node, 1 task)
#
# Submit with:
#   sbatch job_submission_scripts/job-frontier_data_preprocess_1n_smoke.sh
#
#SBATCH -A eng164
#SBATCH -J lumina_dataproc_smoke
#SBATCH -o dataproc-smoke-%j.log
#SBATCH -e dataproc-smoke-%j.log
#SBATCH -t 00:20:00
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
FRONTIER_VENV_BIN=${FRONTIER_VENV_BIN:-${REPO_ROOT}/lumina-frontier-rocm711-install/.venv/bin}

# Optional overrides for a different smoke subset
SMOKE_CASES=${SMOKE_CASES:-pglib_opf_case500_goc}
SMOKE_GROUP_IDS=${SMOKE_GROUP_IDS:-0}

module reset
ml cpe/24.07
ml cce/18.0.0
ml rocm/7.1.1
ml amd-mixed/7.1.1
ml craype-accel-amd-gfx90a
ml PrgEnv-gnu
ml miniforge3/23.11.0-0
module unload darshan-runtime || true
export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}

export PATH="${FRONTIER_VENV_BIN}:${PATH}"
cd "${REPO_ROOT}"

echo "Python: $(which python3)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"

echo "Smoke cases: ${SMOKE_CASES}"
echo "Smoke groups: ${SMOKE_GROUP_IDS}"

export LUMINA_PREPROCESS_ROOT="${REPO_ROOT}"
export LUMINA_PREPROCESS_CASES="${SMOKE_CASES}"
export LUMINA_PREPROCESS_GROUP_IDS="${SMOKE_GROUP_IDS}"

srun --ntasks=1 \
     --ntasks-per-node=1 \
     --cpus-per-task=${SLURM_CPUS_PER_TASK:-7} \
     python3 scripts/data_process_frontier.py

ROOT="${REPO_ROOT}/"
TMP_DIR="${ROOT}OPFData/raw/dataset_release_1/gridopt-dataset-tmp"
if [[ -d "${TMP_DIR}" ]]; then
    echo "Removing temp dir: ${TMP_DIR}"
    rm -rf "${TMP_DIR}"
fi

for CASE in ${SMOKE_CASES}; do
    echo "Building smoke shards for ${CASE} ..."
    python3 scripts/opf_build_shards.py \
        --root "${ROOT}" \
        --case-name "${CASE}" \
        --group-ids ${SMOKE_GROUP_IDS} \
        --train-split 0.8 \
        --link-mode hardlink \
        --overwrite
    echo "Shards done for ${CASE}"
done

echo "Smoke preprocessing done: $(date '+%Y-%m-%d %H:%M:%S')"
