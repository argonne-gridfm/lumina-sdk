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

srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 --gpus-per-task=$SLURM_GPUS_ON_NODE --gpu-bind=none \
    python -m torch.distributed.run \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    example/opf/train_opf_ddp.py \
    --config=configs/config.perlmutter.ddp.yaml \
    --cases case14 case30 case57 case118 case500 case2000 case4661 \
    --group_ids 0 1 \
    --model_type=HGT \
    --homo_model_config configs/model/homognn.yaml \
    --hetero_model_config configs/model/heterognn.yaml \
    --loss_type=mse \
    --wandb \
    --wandb_group_name=debug \
    --wandb_run_name=hgt_corecases_bs8_gbs128_4gpus_sqlite

    # --cases case14 case30 case57 case118 case500 case2000 case4661 case6470 case10000 case13659 \