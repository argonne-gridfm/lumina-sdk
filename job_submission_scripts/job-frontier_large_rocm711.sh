#!/bin/bash

#SBATCH -A lrn087
#SBATCH -J lumina-64n
#SBATCH -o frontier-%j.log
#SBATCH -e frontier-%j.log
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -t 02:00:00
#SBATCH -N 64
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --network=disable_rdzv_get

set -euo pipefail

#SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
#REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
REPO_ROOT=/lustre/orion/lrn087/proj-shared/emon/lumina-sdk
SCRIPT_DIR="${REPO_ROOT}/job_submission_scripts"

export FRONTIER_VENV_BIN=/lustre/orion/lrn087/proj-shared/emon/environment/lumina-frontier-rocm711/.venv/bin
export LUMINA_ROOT=/lustre/orion/lrn087/proj-shared/emon/datasets
export LUMINA_LOGGING_DIR=/lustre/orion/lrn087/proj-shared/emon/logs/lumina-384n-${SLURM_JOB_ID}
export LUMINA_CHECKPOINT_DIR=/lustre/orion/lrn087/proj-shared/emon/checkpoints/lumina-384n-${SLURM_JOB_ID}

export LUMINA_CONFIG=configs/config.frontier.rocm711.large.yaml
export LUMINA_MODEL_CONFIG=configs/model/config_rank1_hgt.yaml
export LUMINA_CASES="case2000"
export LUMINA_GROUP_IDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"

# Leave empty for a new run. Set an absolute checkpoint path to resume.
export RESUME_CHECKPOINT=""
#export RESUME_CHECKPOINT=/lustre/orion/lrn087/proj-shared/emon/checkpoints/lumina-512n/last-pglib_opf_case2000_goc.pt
source "${SCRIPT_DIR}/module-to-load-frontier-rocm711.sh"
source "${SCRIPT_DIR}/setting.sh"
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
export PATH="${FRONTIER_VENV_BIN}:${PATH}"

cd "${REPO_ROOT}"
mkdir -p "${LUMINA_LOGGING_DIR}" "${LUMINA_CHECKPOINT_DIR}"

export MASTER_ADDR
MASTER_ADDR=$(hostname -i | awk '{print $1}')
export MASTER_PORT=29500
export OMP_NUM_THREADS=7
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

export MIOPEN_USER_DB_PATH="/tmp/miopen-${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_USER_DB_PATH}"
mkdir -p "${MIOPEN_USER_DB_PATH}"

if (( SLURM_JOB_NUM_NODES == 1 )); then
    export NCCL_NET=Socket
else
    export NCCL_NET=OFI
fi

RESUME_ARGS=()
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    RESUME_ARGS=(--resume_checkpoint "${RESUME_CHECKPOINT}")
fi

echo "Job ${SLURM_JOB_ID}: ${SLURM_JOB_NUM_NODES} nodes, ${SLURM_NTASKS} GPU ranks"

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
     --wandb \
     --wandb_project=lumina_frontier \
     --wandb_run_name 671M_300k_64 \
     "${RESUME_ARGS[@]}"
