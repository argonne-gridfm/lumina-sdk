#!/bin/bash
# Frontier data pre-processing job — heterogeneous OPFDataset (HGT/RGAT/HEAT)
#
# Distributes 40 tasks (2 cases × 20 groups) across SLURM tasks.
# Each task downloads the raw tar.gz for its assigned (case, group) from GCS,
# extracts JSON files, and writes group_<id>.pt to OPFData/processed/.
#
# Submit with:  sbatch job_submission_scripts/job-frontier_data_preprocess.sh
#
#SBATCH -A eng164
#SBATCH -J lumina_dataproc
#SBATCH -o dataproc-%j.log
#SBATCH -e dataproc-%j.log
#SBATCH -t 02:00:00
#SBATCH -q normal
#SBATCH -N 5
#SBATCH --ntasks-per-node=8       # 5 nodes × 8 tasks = 40 tasks = 2 cases × 20 groups
#SBATCH --cpus-per-task=7         # joblib parallel JSON processing per task

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
FRONTIER_VENV_BIN=${FRONTIER_VENV_BIN:-${REPO_ROOT}/lumina-frontier-rocm711-install/.venv/bin}

# --- proxy (required on Frontier login/compute for outbound HTTPS) ---
export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'

# --- modules ---
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
echo "Nodes: ${SLURM_JOB_NUM_NODES}, Tasks: ${SLURM_NTASKS}"

# --- preprocessing: download + process all (case, group) pairs ---
srun --ntasks=${SLURM_NTASKS:-40} \
     --ntasks-per-node=${SLURM_NTASKS_PER_NODE:-8} \
     --cpus-per-task=${SLURM_CPUS_PER_TASK:-7} \
     python scripts/data_process_frontier.py

echo "Preprocessing done: $(date '+%Y-%m-%d %H:%M:%S')"

ROOT="${REPO_ROOT}/"

# --- clean up shared temp directory now that all tasks are finished ---
TMP_DIR="${ROOT}OPFData/raw/dataset_release_1/gridopt-dataset-tmp"
if [[ -d "${TMP_DIR}" ]]; then
    echo "Removing temp dir: ${TMP_DIR}"
    rm -rf "${TMP_DIR}"
fi

# --- build shard manifests (single process, reads the .pt files just written) ---

for CASE in pglib_opf_case500_goc pglib_opf_case2000_goc; do
    echo "Building shards for ${CASE} ..."
    python scripts/opf_build_shards.py \
        --root "${ROOT}" \
        --case-name "${CASE}" \
        --group-ids $(seq -s ' ' 0 19) \
        --train-split 0.8 \
        --link-mode hardlink \
        --overwrite
    echo "Shards done for ${CASE}"
done

echo "All done: $(date '+%Y-%m-%d %H:%M:%S')"
