#!/bin/bash -l
# The first 15 characters of the job name are displayed in the qstat output:
#PBS -A GridFM
#PBS -N lumina
#PBS -l filesystems=home:grand:eagle
#PBS -j oe
# -------------------------------------------------------------------------------------------------------------------
# To submit this script on Polaris:
# qsub -V -q debug -l select=2 -l walltime=00:30:00 scripts/polaris.sh
# qsub -V -q prod -l select=64 -l walltime=00:10:00 scripts/polaris.sh
# qsub -V -q debug -l select=1 -l walltime=01:00:00 -I
# qsub -V -q debug-scaling -l select=8 -l walltime=00:30:00 -I
# -------------------------------------------------------------------------------------------------------------------
echo Working directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR

module use /soft/modulefiles
module load conda
conda activate base

source /eagle/GridFM/conda_envs/lumina/bin/activate
echo python3: $(which python3)

TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
echo "Job ID: ${PBS_JOBID}"
echo "Job started at: ${TSTAMP}"

NHOSTS=$(wc -l <"${PBS_NODEFILE}")
NGPU_PER_HOST=$(nvidia-smi -L | wc -l)
NGPUS="$((${NHOSTS} * ${NGPU_PER_HOST}))"
MASTER_ADDR=$(head -n1 "$PBS_NODEFILE")
MASTER_PORT=29501  # pick an unused port
echo "NHOSTS: ${NHOSTS}, NGPU_PER_HOST: ${NGPU_PER_HOST}, NGPUS: ${NGPUS}, MASTER_ADDR: ${MASTER_ADDR}, MASTER_PORT: ${MASTER_PORT}"

mkdir -p $PBS_O_WORKDIR/logs

SCRATCH=/eagle/GridFM
export HF_HOME=$SCRATCH/cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_MODULES_CACHE=$HF_HOME/modules
export TRITON_CACHE_DIR=$SCRATCH/cache/triton

# export WANDB_API_KEY=

export PYTHONPATH="$PBS_O_WORKDIR:$PYTHONPATH"

RDZV_ID=${PBS_JOBID:-$RANDOM}

mpiexec -n ${NHOSTS} --ppn 1 --depth=64 --cpu-bind depth \
    python -m torch.distributed.run \
    --nnodes ${NHOSTS} \
    --nproc_per_node ${NGPU_PER_HOST} \
    --rdzv_backend c10d \
    --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
    --rdzv_id ${RDZV_ID} \
    example/opf/train_opf.py \
    --case case14 \
    --group_id 0 \
    --config configs/config.polaris.yaml \
    --model_type HeteroGNN \
    --loss_type mse \
    --minmax_scaling \
    --accelerator auto \
    --devices ${NGPU_PER_HOST} \
    --num_nodes ${NHOSTS}
