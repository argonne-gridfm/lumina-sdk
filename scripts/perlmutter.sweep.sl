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
# salloc --nodes 1 --qos interactive --time 00:30:00 --constraint gpu --gpus 4 --account amsc004
# -------------------------------------------------------------------------------------------------------------------

export SLURM_CPU_BIND="cores"
export MASTER_PORT=29500 # default from torch launcher
export MASTER_ADDR=$(hostname)
export OMP_NUM_THREADS=32
# export WANDB_API_KEY=

module load conda
conda activate ${CFS}/amsc004/conda_envs/lumina

wandb agent kibaek-kim-argonne-national-laboratory/lumina-core/cw6v7253
