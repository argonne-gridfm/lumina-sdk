# HPC Training

This guide covers running LUMINA on Argonne's Polaris and NERSC's Perlmutter supercomputers.

!!! note "Substitute the UPPERCASE placeholders for your environment"
    The job scripts below contain `<UPPERCASE>` placeholders that you must replace before submitting:

    - `<HPC_ACCOUNT>` — your allocation ID
    - `<CONDA_ENV_PATH>` — path to your conda env
    - `<LUMINA_REPO_PATH>` — your local clone of `lumina-sdk`

## Polaris (ALCF)

### Single-node script

Use the pre-built Polaris config:

```bash
#!/bin/bash
#PBS -l select=1:system=polaris
#PBS -l walltime=02:00:00
#PBS -q prod
#PBS -A <HPC_ACCOUNT>

module load conda
conda activate <CONDA_ENV_PATH>
cd <LUMINA_REPO_PATH>

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
#PBS -A <HPC_ACCOUNT>

module load conda
conda activate <CONDA_ENV_PATH>
cd <LUMINA_REPO_PATH>

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
#SBATCH -A <HPC_ACCOUNT>

module load pytorch
cd <LUMINA_REPO_PATH>
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
#SBATCH -A <HPC_ACCOUNT>

module load pytorch
cd <LUMINA_REPO_PATH>/
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

## Device Visibility Notes

- **Perlmutter / Polaris**: each rank typically sees all node GPUs, so the trainer uses `LOCAL_RANK` to select the local device.
- **Frontier**: when using `job_submission_scripts/launch_rank_frontier.sh`, each rank gets one visible GPU via `ROCR_VISIBLE_DEVICES=$SLURM_LOCALID`, so the selected device index is `0` inside that process.

## Frontier (OLCF)

Frontier helper scripts are provided as templates under `job_submission_scripts/`.

1. Create or activate a ROCm 7.1.1 environment:

```bash
bash install/frontier/setup_env_frontier_rocm711.sh
```

2. (Optional) preprocess heterogeneous OPF data:

```bash
sbatch job_submission_scripts/job-frontier_data_preprocess.sh
```

3. Launch multi-node DDP training:

```bash
sbatch job_submission_scripts/job-frontier_16n_rocm711.sh
```

Set `FRONTIER_VENV_BIN` (and path overrides below) to match your allocation and filesystem layout before submission.

## Frontier Path Overrides

Use placeholders and override site-specific paths at runtime instead of committing cluster-specific absolute paths:

```bash
python example/opf/train_opf_ddp_frontier.py \
  --config configs/config.frontier.rocm711.yaml \
  --root <DATA_ROOT> \
  --logging_dir <LOG_DIR> \
  --checkpoint_dir <CKPT_DIR>
```

You can also use env vars:

```bash
export LUMINA_ROOT=<DATA_ROOT>
export LUMINA_LOGGING_DIR=<LOG_DIR>
export LUMINA_CHECKPOINT_DIR=<CKPT_DIR>
```

## Existing HPC Documentation

Additional system-specific docs:

- [W&B sweeps on Perlmutter](../wandb_sweep_perlmutter.md)
<!-- - [HuggingFace integration](../huggingface.md) -->
