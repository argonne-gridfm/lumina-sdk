# HPC Training

This guide covers running LUMINA on Argonne's Polaris, NERSC's Perlmutter, and OLCF's Frontier supercomputers.

## Polaris (ALCF)

### Setup

```bash
module load conda
conda activate /path/to/lumina-env
cd /eagle/projects/GridFM/lumina-core
```

### Config

Use the pre-built Polaris config:

```bash
torchrun --standalone --nproc_per_node=4 \
  example/opf/train_opf_ddp.py \
  --config configs/config.polaris.ddp.yaml \
  --cases case14 case30 case118 \
  --group_ids 0 1 2 3 4
```

### Job Script

```bash
#!/bin/bash
#PBS -l select=2:system=polaris
#PBS -l walltime=02:00:00
#PBS -q prod
#PBS -A GridFM

cd $PBS_O_WORKDIR

NNODES=$(cat $PBS_NODEFILE | sort | uniq | wc -l)
NGPUS_PER_NODE=4
NTOTGPUS=$((NNODES * NGPUS_PER_NODE))

torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=$NGPUS_PER_NODE \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$(head -1 $PBS_NODEFILE):29500 \
  example/opf/train_opf_ddp.py \
  --config configs/config.polaris.ddp.yaml \
  --cases case14 case118 case2000 \
  --group_ids 0 1 2 3 4 5 6 7 8 9
```

## Perlmutter (NERSC)

### Setup

```bash
module load pytorch
cd $SCRATCH/lumina-core
pip install -e .
```

### Config

```bash
torchrun --standalone --nproc_per_node=4 \
  example/opf/train_opf_ddp.py \
  --config configs/config.perlmutter.ddp.yaml \
  --cases case14 case118 \
  --group_ids 0 1
```

### Job Script

```bash
#!/bin/bash
#SBATCH -N 2
#SBATCH -C gpu
#SBATCH -G 8
#SBATCH -t 02:00:00
#SBATCH -q regular
#SBATCH -A m1234

srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=4 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1):29500 \
  example/opf/train_opf_ddp.py \
  --config configs/config.perlmutter.ddp.yaml \
  --cases case14 case118 case2000 \
  --group_ids 0 1 2 3 4
```

## Frontier (OLCF)

### Setup

See `install/frontier/` for system-specific installation scripts.

### Config

```bash
torchrun --standalone --nproc_per_node=8 \
  example/opf/train_opf_ddp.py \
  --config configs/config.frontier.ddp.yaml \
  --cases case14 case118 \
  --group_ids 0 1
```

## Multi-Node Tips

- **Data staging**: Use `data.staging.root` in config to stage datasets to node-local storage (e.g., `$TMPDIR`)
- **Gradient accumulation**: Set `training.accumulate_grad_batches` to simulate larger batch sizes
- **Sharded datasets**: For large cases, pre-build shards with `scripts/opf_build_shards.py`
- **W&B logging**: Only rank 0 logs to W&B; use `--wandb` flag

## Existing HPC Documentation

Additional system-specific docs:

- [Polaris setup](../polaris.md)
- [Perlmutter setup](../perlmutter.md)
- [W&B sweeps on Perlmutter](../wandb_sweep_perlmutter.md)
- [HuggingFace integration](../huggingface.md)
