#!/bin/bash
# Small debug default:
#   sbatch -A <PROJECT> job_submission_scripts/job-frontier_large_rocm711.sh
# Large run example:
#   sbatch -A <PROJECT> -q normal -N 64 -t 06:00:00 job_submission_scripts/job-frontier_large_rocm711.sh
#
# Required: LUMINA_ROOT must contain the prepared OPFData/on_disk dataset.
# Optional: FRONTIER_VENV_BIN, LUMINA_CONFIG, LUMINA_MODEL_CONFIG,
#           LUMINA_LOGGING_DIR, LUMINA_CHECKPOINT_DIR, LUMINA_CASES,
#           LUMINA_GROUP_IDS, RESUME_CHECKPOINT.
#SBATCH -J lumina-frontier
#SBATCH -o frontier-%j.log
#SBATCH -e frontier-%j.log
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --network=disable_rdzv_get

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
FRONTIER_VENV_BIN=${FRONTIER_VENV_BIN:-${REPO_ROOT}/lumina-frontier-rocm711-install/.venv/bin}
LUMINA_ROOT=${LUMINA_ROOT:?Set LUMINA_ROOT to the directory containing OPFData/on_disk}
LUMINA_LOGGING_DIR=${LUMINA_LOGGING_DIR:-${REPO_ROOT}/logs/frontier}
LUMINA_CHECKPOINT_DIR=${LUMINA_CHECKPOINT_DIR:-${REPO_ROOT}/ddp-checkpoints/frontier}
LUMINA_CASES=${LUMINA_CASES:-case2000}
LUMINA_GROUP_IDS=${LUMINA_GROUP_IDS:-0}
LUMINA_CONFIG=${LUMINA_CONFIG:-configs/config.frontier.rocm711.debug.yaml}
LUMINA_MODEL_CONFIG=${LUMINA_MODEL_CONFIG:-configs/model/config_rank1_hgt.yaml}

source "${SCRIPT_DIR}/module-to-load-frontier-rocm711.sh"
export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}
export PATH="${FRONTIER_VENV_BIN}:${PATH}"
export LUMINA_ROOT LUMINA_LOGGING_DIR LUMINA_CHECKPOINT_DIR

cd "${REPO_ROOT}"
mkdir -p "${LUMINA_LOGGING_DIR}" "${LUMINA_CHECKPOINT_DIR}"

export MASTER_ADDR=$(hostname -i | awk '{print $1}')
export MASTER_PORT=${MASTER_PORT:-29500}
export OMP_NUM_THREADS=7
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export MIOPEN_USER_DB_PATH=/tmp/miopen-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
mkdir -p "${MIOPEN_USER_DB_PATH}"

if (( SLURM_JOB_NUM_NODES == 1 )); then
    export NCCL_NET=Socket
else
    export NCCL_NET=OFI
fi

RESUME_ARGS=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    RESUME_ARGS=(--resume_checkpoint "${RESUME_CHECKPOINT}")
fi

echo "Job ${SLURM_JOB_ID}: nodes=${SLURM_JOB_NUM_NODES}, tasks=${SLURM_NTASKS}, master=${MASTER_ADDR}:${MASTER_PORT}"
echo "Python: $(which python3)"
echo "Data: ${LUMINA_ROOT}"

srun --ntasks="${SLURM_NTASKS}" \
     --ntasks-per-node=8 \
     --cpus-per-task=7 \
     --gpus-per-task=1 \
     --gpu-bind=closest \
     python3 -u example/opf/train_opf_ddp_frontier.py \
     --config "${LUMINA_CONFIG}" \
     --hetero_model_config "${LUMINA_MODEL_CONFIG}" \
     --cases ${LUMINA_CASES} \
     --group_ids ${LUMINA_GROUP_IDS} \
     --model_type HGT \
     --loss_type mse \
     "${RESUME_ARGS[@]}"
