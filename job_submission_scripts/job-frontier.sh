#!/bin/bash
#SBATCH -A LRN078
#SBATCH -J LUMINA
#SBATCH -o job-luma-%j.out
#SBATCH -e job-luma-%j.out
#SBATCH -t 01:00:00
#SBATCH --gpus-per-node=8        # set to GPUs per node on your system
#SBATCH --cpus-per-task=8        # dataloader/OS threads; tune as needed
#SBATCH --ntasks-per-node=1      # one task; torchrun will spawn per-GPU
###SBATCH --exclusive
#SBATCH -p batch 
#SBATCH -q debug
#SBATCH -N 1 #16 
##SBATCH -S 1

set -euo pipefail
 
export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'
 
# Load conda environemnt
source /lustre/orion/lrn070/world-shared/mlupopa/KibaekKim/lumina-core/module-to-load-frontier-rocm640.sh
source activate /lustre/orion/lrn070/world-shared/mlupopa/KibaekKim/lumina-core/install/frontier/lumina-frontier-install/.venv/
 
which python
python -c "import numpy; print(numpy.__version__)"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONWARNINGS="ignore"

# Derive rendezvous and world size
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
MASTER_PORT=${MASTER_PORT:-29500}
NODE_RANK=${SLURM_NODEID:-0}
NODES=${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-1}}

GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-}
if [[ -z "$GPUS_PER_NODE" && -n "${SLURM_JOB_GPUS:-}" ]]; then
  GPUS_PER_NODE=$(python - <<'PY'
import os
print(len(os.environ["SLURM_JOB_GPUS"].split(",")))
PY
)
fi
if [[ -z "$GPUS_PER_NODE" ]]; then
  GPUS_PER_NODE=$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)
fi

WORLD_SIZE=$((NODES * GPUS_PER_NODE))

echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NODE_RANK=${NODE_RANK}"
echo "NODES=${NODES}"
echo "GPUS_PER_NODE=${GPUS_PER_NODE}"
echo "WORLD_SIZE=${WORLD_SIZE}"

export MASTER_ADDR MASTER_PORT WORLD_SIZE RANK=${NODE_RANK} LOCAL_RANK=0

# Optional NCCL/RCCL debugging
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1    # often safer on Cray/ROCm; drop if using IB
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3  # adjust or unset if unknown

torchrun \
  --nnodes "${NODES}" \
  --nproc_per_node "${GPUS_PER_NODE}" \
  --node_rank "${NODE_RANK}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
  example/opf/train_opf_ddp.py \
  --config configs/config.frontier.ddp.yaml \
    --case case2000 \
    --group_id 0 \
    --model_type HeteroGNN \
    --loss_type mse
