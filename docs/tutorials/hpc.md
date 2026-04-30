# HPC Training

This guide covers running LUMINA on Argonne's Polaris and NERSC's Perlmutter supercomputers.

## Polaris (ALCF)

### Single-node script

Use the pre-built Polaris config:

```bash
#!/bin/bash
#PBS -l select=1:system=polaris
#PBS -l walltime=02:00:00
#PBS -q prod
#PBS -A account_name

module load conda
conda activate /path/to/lumina-env
cd /path/to/lumina-sdk

export MASTER_ADDR=$(hostname).hsn.cm.polaris.alcf.anl.gov
export MASTER_PORT=29500

mpiexec -n 1 -ppn 4 \
 python example/opf/train_opf_ddp.py \
  --config configs/config.polaris.ddp.yaml \
  --cases case14 case30 case118 \
  --group_ids 0 1 2 3 4
```

### Multi-node DDP script

Use the pre-built Polaris config:

```bash
#!/bin/bash
#PBS -l select=2:system=polaris
#PBS -l walltime=02:00:00
#PBS -q prod
#PBS -A account_name

module load conda
conda activate /path/to/lumina-env
cd /path/to/lumina-sdk

NNODES=$(cat $PBS_NODEFILE | sort | uniq | wc -l)
NGPUS_PER_NODE=4
NTOTGPUS=$((NNODES * NGPUS_PER_NODE))

export MASTER_ADDR=$(hostname).hsn.cm.polaris.alcf.anl.gov
export MASTER_PORT=29500

mpiexec -n ${NTOTGPUS} -ppn ${NGPUS_PER_NODE} \
  python example/opf/train_opf_ddp.py \
  --config configs/config.polaris.ddp.yaml \
  --cases case14 case118 case2000 \
  --group_ids 0 1 2 3 4 5 6 7 8 9
```

## Perlmutter (NERSC)

### Single-node script

```bash
#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 8
#SBATCH -t 02:00:00
#SBATCH -q regular
#SBATCH -A m1234

module load pytorch
cd /path/to/lumina-sdk
pip install -e .

export SLURM_CPU_BIND="cores"
export MASTER_PORT=${MASTER_PORT:-29500} 
export MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)}
export OMP_NUM_THREADS=32

srun --ntasks-per-node 4 --gpus-per-task 1 \ 
  python example/opf/train_opf_ddp.py \
  --config configs/config.perlmutter.ddp.yaml \
  --cases case14 case118 \
  --group_ids 0 1
```

### Multi-node DDP script

```bash
#!/bin/bash
#SBATCH -N 2
#SBATCH -C gpu
#SBATCH -G 8
#SBATCH -t 02:00:00
#SBATCH -q regular
#SBATCH -A m1234

module load pytorch
cd /path/to/lumina-sdk/
pip install -e .

export SLURM_CPU_BIND="cores"
export MASTER_PORT=${MASTER_PORT:-29500}
export MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)}
export OMP_NUM_THREADS=32

srun --ntasks-per-node 4 --gpus-per-task 1 \ # optional to specify -N if using a subset of nodes
  python example/opf/train_opf_ddp.py \
  --config configs/config.perlmutter.ddp.yaml \
  --cases case14 case118 case2000 \
  --group_ids 0 1 2 3 4
```

## Multi-Node Tips

- **Data staging**: Use `data.staging.root` in config to stage datasets to node-local storage (e.g., `$TMPDIR`)
- **Gradient accumulation**: Set `training.accumulate_grad_batches` to simulate larger batch sizes
- **Sharded datasets**: For large cases, pre-build shards with `scripts/opf_build_shards.py`
- **W&B logging**: Only rank 0 logs to W&B; use `--wandb` flag

## Existing HPC Documentation

Additional system-specific docs:

- [W&B sweeps on Perlmutter](../wandb_sweep_perlmutter.md)
- [HuggingFace integration](../huggingface.md)
