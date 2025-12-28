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
    --cases case14 case30 case57 case118 case500 \
    --config configs/sweeps/stage1/config.yaml \
    --group_ids 0 1 \
    --model_type=GAT \
    --homo_model_config configs/sweeps/stage1/model.yaml \
    --loss_type mse \
    --wandb \
    "$@"
