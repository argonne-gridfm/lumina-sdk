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

source /lustre/orion/eng164/proj-shared/lumina-core/job_submission_scripts/module-to-load-frontier-rocm711.sh
export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}

export PATH="/lustre/orion/eng164/proj-shared/lumina-core/lumina-frontier-rocm711-install/.venv/bin:${PATH}"

cd /lustre/orion/eng164/proj-shared/lumina-core

echo python3: $(which python3)

TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job started at: ${TSTAMP}"

NHOSTS=${SLURM_JOB_NUM_NODES}
NGPU_PER_HOST=${SLURM_GPUS_ON_NODE:-8}
NGPUS="$((${NHOSTS} * ${NGPU_PER_HOST}))"
# Get first node's IP directly via srun to avoid hostname resolution issues
export MASTER_ADDR=$(srun --overlap -N 1 -n 1 --nodelist=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1) hostname -I | awk '{print $1}')
export MASTER_PORT=29500
echo "NHOSTS: ${NHOSTS}, NGPU_PER_HOST: ${NGPU_PER_HOST}, NGPUS: ${NGPUS}, MASTER_ADDR: ${MASTER_ADDR}, MASTER_PORT: ${MASTER_PORT}"

export OMP_NUM_THREADS=7
export PYTHONWARNINGS="ignore"

export TMPDIR=/tmp
export TORCH_MULTIPROCESSING_SHARING_STRATEGY=file_descriptor

# NCCL / RCCL — match the working HydraGNN OPF configuration (no AWS OFI plugin)
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export NCCL_PROTO=Simple
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=hsn0

# ROCm / GPU tuning
export GPU_MAX_HW_QUEUES=2
export MIOPEN_DISABLE_CACHE=1
export ROCM_HOME=${ROCM_PATH}

ulimit -c unlimited

srun -N ${NHOSTS} \
    --ntasks=${NGPUS} \
    --ntasks-per-node=${NGPU_PER_HOST} \
    -c7 \
    --gpus-per-task=1 \
    --gpu-bind=none \
    job_submission_scripts/launch_rank_frontier.sh \
    python example/opf/train_opf_ddp_frontier.py \
    --config configs/config.frontier.rocm711.yaml \
    --hetero_model_config configs/model/config_rank1_hgt.yaml \
    --cases case500 case2000 \
    --group_ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
    --model_type=HGT \
    --loss_type=mse
