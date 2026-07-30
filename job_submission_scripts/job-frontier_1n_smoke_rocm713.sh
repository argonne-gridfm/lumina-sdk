#!/bin/bash
#SBATCH -A eng164
#SBATCH -J lumina_smoke_1n
#SBATCH -o smoke-1n-%j.log
#SBATCH -e smoke-1n-%j.log
#SBATCH -t 00:20:00
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
FRONTIER_VENV_BIN=${FRONTIER_VENV_BIN:-${REPO_ROOT}/lumina-frontier-rocm713-install/.venv/bin}

source "${SCRIPT_DIR}/module-to-load-frontier-rocm713.sh"
export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}
export PATH="${FRONTIER_VENV_BIN}:${PATH}"

cd "${REPO_ROOT}"

echo "python3: $(which python3)"
echo "job: ${SLURM_JOB_ID}"

NHOSTS=${SLURM_JOB_NUM_NODES}
NGPU_PER_HOST=${SLURM_GPUS_ON_NODE:-2}
NGPUS="$((NHOSTS * NGPU_PER_HOST))"
export MASTER_ADDR=$(srun --overlap -N 1 -n 1 --nodelist=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1) hostname -I | awk '{print $1}')
export MASTER_PORT=${MASTER_PORT:-29500}

export OMP_NUM_THREADS=7
export PYTHONWARNINGS="ignore"
export TMPDIR=/tmp
export TORCH_MULTIPROCESSING_SHARING_STRATEGY=file_descriptor

export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
# The aws-ofi-nccl (rccl-net-plugin) fails CXI domain creation on this stack
# (RC -38 ENOSYS) and crashes RCCL. Disable the external net plugin so RCCL uses
# its built-in transports: intra-node xGMI/SHM, inter-node TCP sockets over the
# HSN NICs. Validated single-node; multi-node uses the TCP fallback.
export NCCL_NET_PLUGIN=none

export GPU_MAX_HW_QUEUES=2
export MIOPEN_DISABLE_CACHE=1
export ROCM_HOME=${ROCM_PATH}

srun -N ${NHOSTS} \
    --ntasks=${NGPUS} \
    --ntasks-per-node=${NGPU_PER_HOST} \
    -c7 \
    --gpus-per-task=1 \
    --gpu-bind=none \
    "${SCRIPT_DIR}/launch_rank_frontier.sh" \
    python3 example/opf/train_opf_ddp_frontier.py \
    --config configs/config.frontier.rocm713.smoke.yaml \
    --cases case500 \
    --group_ids 0 \
    --model_type=HGT \
    --loss_type=mse
