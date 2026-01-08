#!/bin/bash
set -euo pipefail

# This script is invoked by wandb agent for each run.
# It uses the SLURM allocation already granted to the job.

export SLURM_CPU_BIND="cores"
export MASTER_PORT=${MASTER_PORT:-29500} # default from torch launcher
export MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)}
export OMP_NUM_THREADS=32
# export WANDB_API_KEY=

module load conda
conda activate ${CFS}/amsc004/conda_envs/lumina

srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 \
    python -m torch.distributed.run \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    example/opf/train_opf_ddp.py \
    --config configs/sweeps/iclr/config.yaml \
    --cases case30 \
    --group_ids 0 1 \
    --model_type=HEAT \
    --loss_type mse \
    --wandb \
    "$@"
