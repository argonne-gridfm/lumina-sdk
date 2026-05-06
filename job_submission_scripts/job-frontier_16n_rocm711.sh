#!/bin/bash
#SBATCH -A eng164
#SBATCH -J lumina_16n
#SBATCH -o gpu_16n-%j.log
#SBATCH -e gpu_16n-%j.log
#SBATCH -t 02:00:00
#SBATCH -q normal
#SBATCH -N 16
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none

module reset
ml cpe/24.07
ml cce/18.0.0
ml rocm/7.1.1
ml amd-mixed/7.1.1
ml craype-accel-amd-gfx90a
ml PrgEnv-gnu
ml miniforge3/23.11.0-0
module unload darshan-runtime || true
export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}

source activate /lustre/orion/eng164/proj-shared/lumina-core/lumina-frontier-rocm711-install/.venv/

cd /lustre/orion/eng164/proj-shared/lumina-core

echo python3: $(which python3)

TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job started at: ${TSTAMP}"

NHOSTS=${SLURM_JOB_NUM_NODES}
NGPU_PER_HOST=${SLURM_GPUS_ON_NODE:-8}
NGPUS="$((${NHOSTS} * ${NGPU_PER_HOST}))"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=29500
echo "NHOSTS: ${NHOSTS}, NGPU_PER_HOST: ${NGPU_PER_HOST}, NGPUS: ${NGPUS}, MASTER_ADDR: ${MASTER_ADDR}, MASTER_PORT: ${MASTER_PORT}"

export OMP_NUM_THREADS=8
export PYTHONWARNINGS="ignore"

export TMPDIR=/tmp
export TORCH_MULTIPROCESSING_SHARING_STRATEGY=file_descriptor

# RCCL / network settings for Frontier (Slingshot-11)
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3
export FI_CXI_ATS=0
export FI_MR_CACHE_MONITOR=disabled
export FI_CXI_RX_MATCH_MODE=hybrid
export FI_CXI_OFLOW_BUF_SIZE=8388608
export FI_CXI_DEFAULT_CQ_SIZE=1048576
export FI_CXI_CQ_FILL_PERCENT=30

export MIOPEN_DISABLE_CACHE=1
export ROCM_HOME=${ROCM_PATH}

ulimit -c unlimited

srun -N ${NHOSTS} \
    --ntasks=${NGPUS} \
    --ntasks-per-node=${NGPU_PER_HOST} \
    --gpus-per-task=1 \
    python example/opf/train_opf_ddp_frontier.py \
    --config configs/config.frontier.rocm711.yaml \
    --hetero_model_config configs/model/config_rank1_hgt.yaml \
    --cases case500 case2000 \
    --group_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
    --model_type=HGT \
    --loss_type=mse \
    --wandb \
    --wandb_project=lumina_frontier \
    --wandb_run_name=hgt_case500_case2000_16n
