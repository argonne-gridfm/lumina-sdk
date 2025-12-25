#!/bin/bash
set -euo pipefail

# This script is invoked by wandb agent for each run.
# It uses the SLURM allocation already granted to the job.

export SLURM_CPU_BIND="cores"
export MASTER_PORT=${MASTER_PORT:-29500} # default from torch launcher
export MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)}
export OMP_NUM_THREADS=32
export WANDB_API_KEY=1cf4939bab84febb2f103a53740597f912f531f0

module load conda
conda activate ${CFS}/amsc004/conda_envs/lumina

srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 \
    python -m torch.distributed.run \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    example/opf/train_opf_ddp.py \
    --cases case14 \
    --config=configs/config.perlmutter.ddp.yaml \
    --group_ids 0 \
    --wandb \
    "$@"
