#!/bin/bash
#SBATCH -A amsc004
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 0:30:00
#SBATCH -N 8
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
# -------------------------------------------------------------------------------------------------------------------
# To submit this script on Polaris:
# sbatch scripts/perlmutter.sl
# salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 4
# -------------------------------------------------------------------------------------------------------------------

export SLURM_CPU_BIND="cores"

module load conda
conda activate ${CFS}/amsc004/conda_envs/lumina

srun python example/opf/train_opf.py --num_nodes=8 --devices=4 --loss_type=augmented_lagrangian
