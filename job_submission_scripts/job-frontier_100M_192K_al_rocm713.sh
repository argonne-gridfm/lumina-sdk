#!/bin/bash
#SBATCH -A lrn087
#SBATCH -J lumina-100M-192K-al
#SBATCH -o frontier-%j.log
#SBATCH -e frontier-%j.log
#SBATCH -p batch
#SBATCH -q normal
#SBATCH -t 06:00:00
#SBATCH -N 384
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --network=disable_rdzv_get

set -euo pipefail

# Clear any conda environment inherited from the login node submission shell
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_EXE CONDA_PYTHON_EXE 2>/dev/null || true

REPO_ROOT=/lustre/orion/lrn087/proj-shared/mlupopa/lumina-sdk
SCRIPT_DIR="${REPO_ROOT}/job_submission_scripts"

export FRONTIER_VENV_BIN=/lustre/orion/lrn087/proj-shared/mlupopa/lumina-frontier-rocm713-install/.venv/bin
export LUMINA_ROOT=/lustre/orion/proj-shared/lrn087/emon/datasets
export LUMINA_LOGGING_DIR=/lustre/orion/lrn087/proj-shared/mlupopa/logs/lumina-100M-192K-al-${SLURM_JOB_ID}
export LUMINA_CHECKPOINT_DIR=/lustre/orion/lrn087/proj-shared/mlupopa/checkpoints/lumina-100M-192K-al-${SLURM_JOB_ID}
export LUMINA_CONFIG=configs/config.frontier.rocm713.al.100M.yaml
export LUMINA_MODEL_CONFIG=configs/model/heterognn_100M.yaml
export LUMINA_CASES="case2000"
export LUMINA_GROUP_IDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"

# Set an absolute checkpoint path to resume continual AL training.
# Leave empty for initial training from the MSE checkpoint.
export RESUME_CHECKPOINT=""

source "${SCRIPT_DIR}/module-to-load-frontier-rocm713.sh"
source "${SCRIPT_DIR}/setting.sh"

export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
export PATH="${FRONTIER_VENV_BIN}:${PATH}"

cd "${REPO_ROOT}"

mkdir -p "${LUMINA_LOGGING_DIR}" "${LUMINA_CHECKPOINT_DIR}"

export MASTER_ADDR="$(scontrol show hostnames "${SLURM_NODELIST}" | head -n 1)"
export MASTER_PORT=29500

echo "MASTER_ADDR: ${MASTER_ADDR}, MASTER_PORT: ${MASTER_PORT}"

export OMP_NUM_THREADS=7
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTHONWARNINGS="ignore"
export TMPDIR=/tmp
export TORCH_MULTIPROCESSING_SHARING_STRATEGY=file_descriptor

export ROCM_HOME=${ROCM_PATH}
export GPU_MAX_HW_QUEUES=2
export MIOPEN_DISABLE_CACHE=1

export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
# The aws-ofi-nccl (rccl-net-plugin) fails CXI domain creation on this stack
# (RC -38 ENOSYS) and crashes RCCL. Disable the external net plugin so RCCL uses
# its built-in transports: intra-node xGMI/SHM, inter-node TCP sockets over the
# HSN NICs. Validated single-node; multi-node uses the TCP fallback.
export NCCL_NET_PLUGIN=none

export MIOPEN_USER_DB_PATH="/tmp/miopen-${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_USER_DB_PATH}"

mkdir -p "${MIOPEN_USER_DB_PATH}"

RESUME_ARGS=()
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    RESUME_ARGS=(--resume_checkpoint "${RESUME_CHECKPOINT}")
fi

echo "Job ${SLURM_JOB_ID}: ${SLURM_JOB_NUM_NODES} nodes, ${SLURM_NTASKS} GPU ranks"

srun \
    --ntasks="${SLURM_NTASKS}" \
    --ntasks-per-node=8 \
    --cpus-per-task=7 \
    --gpus-per-task=1 \
    --gpu-bind=none \
    bash -c 'export ROCR_VISIBLE_DEVICES=${SLURM_LOCALID}; exec python3 -u example/opf/train_opf_ddp_frontier.py "$@"' -- \
        --config "${LUMINA_CONFIG}" \
        --hetero_model_config "${LUMINA_MODEL_CONFIG}" \
        --cases ${LUMINA_CASES} \
        --group_ids ${LUMINA_GROUP_IDS} \
        --model_type HGT \
        --loss_type augmented_lagrangian \
        --wandb \
        --wandb_project=lumina-frontier \
        --wandb_run_name 100M_192K_al \
        "${RESUME_ARGS[@]}"
