# W&B Sweep on Perlmutter

This guide shows the pattern we use for a W&B sweep of LUMINA training on NERSC Perlmutter:

1. A sweep config (`opf_<model>.yaml`) that defines the hyper-parameter grid and the per-run command.
2. A per-run launcher script that the W&B agent invokes once per trial — it sets up DDP and calls `example/opf/train_opf_ddp.py` with the sweep overrides.
3. A SLURM job script that allocates nodes and runs one or more `wandb agent` processes.

The example files mentioned below (`configs/sweeps/...`, `scripts/perlmutter/...`) are **not shipped with the repo** — they're git-ignored because they contain account-specific settings. Use the templates in this guide as a starting point.

!!! note "Substitute the UPPERCASE placeholders for your environment"
    The snippets below use `<UPPERCASE>` placeholders that you must replace before submitting:

    - `<HPC_ACCOUNT>` — your NERSC allocation (e.g. `m1234`)
    - `<CONDA_ENV_PATH>` — your conda env on `${CFS}` or `${PSCRATCH}`
    - `<LUMINA_REPO_PATH>` — your local clone of `lumina-sdk`
    - `<WANDB_ENTITY>`, `<WANDB_PROJECT>`, `<SWEEP_ID>` — your W&B identifiers

## Step 1: Define the sweep

Create a YAML under `configs/sweeps/` (gitignored). The `command` block is what the W&B agent runs for each trial — it points at your per-run launcher script.

```yaml
# configs/sweeps/opf_hgt.yaml
program: scripts/perlmutter/sweep/launch.hgt.sh
method: bayes
metric:
  name: val/loss/total
  goal: minimize
command:
  - ${env}
  - ${program}
  - ${args}
parameters:
  models.HGT.hidden_channels:
    values: [128, 256, 512]
  models.HGT.num_layers:
    values: [4, 6, 8]
  models.HGT.num_heads:
    values: [2, 4, 8]
  models.HGT.dropout:
    values: [0.0, 0.1, 0.2]
  optimizer.AdamW.lr:
    values: [1.0e-4, 5.0e-4, 1.0e-3]
  wandb_project:
    value: <WANDB_PROJECT>
```

Register the sweep on a login node:

```bash
export WANDB_API_KEY=<YOUR_KEY>
wandb sweep configs/sweeps/opf_hgt.yaml
# prints a sweep path like <WANDB_ENTITY>/<WANDB_PROJECT>/<SWEEP_ID>
```

## Step 2: Per-run launcher (`launch.hgt.sh`)

This script is invoked by the W&B agent for each trial. It uses the SLURM allocation already granted to the agent's job and forwards W&B-injected hyperparameter overrides via `"$@"`.

```bash
#!/bin/bash
set -euo pipefail

export SLURM_CPU_BIND="cores"
export MASTER_PORT=${MASTER_PORT:-29500}
export MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)}
export OMP_NUM_THREADS=32

module load conda
conda activate <CONDA_ENV_PATH>

srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 \
    python -m torch.distributed.run \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    example/opf/train_opf_ddp.py \
    --config configs/config.perlmutter.ddp.yaml \
    --cases case30 \
    --group_ids 0 1 \
    --model_type HGT \
    --hetero_model_config configs/model/heterognn.yaml \
    --loss_type mse \
    --wandb \
    "$@"
```

Make it executable: `chmod +x scripts/perlmutter/sweep/launch.hgt.sh`.

## Step 3: SLURM agent job (`launch.sweep.sl`)

Each SLURM array task runs one `wandb agent`, which in turn invokes the launcher above for each trial pulled from the sweep queue.

```bash
#!/bin/bash
#SBATCH -A <HPC_ACCOUNT>
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 2:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --gpus-per-task=4
#SBATCH --gpu-bind=none
#SBATCH --array=0-49

export SLURM_CPU_BIND="cores"
export MASTER_PORT=29500
export MASTER_ADDR=$(hostname)
export OMP_NUM_THREADS=32

module load conda
conda activate <CONDA_ENV_PATH>

# Total runs = (array size) * --count
wandb agent --count 1 <WANDB_ENTITY>/<WANDB_PROJECT>/<SWEEP_ID>
```

Submit:

```bash
sbatch scripts/perlmutter/sweep/launch.sweep.sl
```

## Monitoring

- `sqs` — your job queue status
- W&B UI — sweep dashboard under `<WANDB_ENTITY>/<WANDB_PROJECT>`
- Logs — `slurm-<jobid>_<arrayid>.out` in the submission directory

## Common pitfalls

- **`WANDB_API_KEY` not set on compute nodes.** Either export it in your job script or store it in `~/.netrc` so the agent picks it up.
- **Sweep config path mismatch.** The `program:` line in the sweep YAML is interpreted relative to the working directory of the agent, not the sweep config file. Use a path from `<LUMINA_REPO_PATH>` (e.g. `scripts/perlmutter/sweep/launch.hgt.sh`).
- **Idle agents.** If `wandb agent --count 1` finishes faster than expected, check that the trial actually ran (look at `slurm-*.out`). Sweep config syntax errors cause agents to exit immediately without running anything.
- **Disk locality.** For large cases, stage the dataset to `$PSCRATCH` and point `root` in the perlmutter config at it; reading raw OPFData over CFS will bottleneck training.
