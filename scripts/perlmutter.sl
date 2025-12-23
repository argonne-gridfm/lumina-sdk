#!/bin/bash
#SBATCH -A amsc004
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 0:30:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --gpus-per-task=4
#SBATCH --gpu-bind=none
# -------------------------------------------------------------------------------------------------------------------
# To submit this script on Polaris:
# sbatch scripts/perlmutter.sl
# salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 4
# -------------------------------------------------------------------------------------------------------------------

export SLURM_CPU_BIND="cores"
export MASTER_PORT=29500 # default from torch launcher
export MASTER_ADDR=$(hostname)
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
    --config=configs/config.perlmutter.ddp.yaml \
    --group_id=9 \
    --loss_type=mse
