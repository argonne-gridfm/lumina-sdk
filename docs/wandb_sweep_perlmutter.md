# W&B Sweep on Perlmutter (HGT example)

This guide shows how to run a W&B sweep for the HGT model on Perlmutter using:
- `configs/sweeps/iclr/opf_hgt.yaml` (sweep definition)
- `scripts/perlmutter/sweep/iclr/launch.hgt.sh` (per-run entrypoint)
- `scripts/perlmutter/sweep/iclr/launch.sweep.sl` (SLURM job that runs W&B agents)

## How the pieces connect

1) `wandb sweep configs/sweeps/iclr/opf_hgt.yaml` registers the sweep.
2) The sweep config's `command` calls `launch.hgt.sh` for each run.
3) `launch.hgt.sh` starts DDP training with `example/opf/train_opf_ddp.py` and forwards sweep overrides via `"$@"`.
4) `launch.sweep.sl` allocates Perlmutter resources and runs one or more `wandb agent` processes.

## Step 1: Create the sweep (login node)

```bash
# Make sure W&B can authenticate.
export WANDB_API_KEY=YOUR_KEY

# Register the sweep and get the sweep ID.
wandb sweep configs/sweeps/iclr/opf_hgt.yaml
```

The command prints a sweep path like `ENTITY/PROJECT/SWEEP_ID`. You will use it in the agent command.

Note: `opf_hgt.yaml` defines HGT hyperparameters (layers, hidden size, heads, dropout, etc.) and sets
`wandb_project: lumina-sweep-ICLR-hgt-stage1` as a training config parameter. If you create the sweep
under a different W&B project, keep those aligned.

## Step 2: Configure the agent job

Edit `scripts/perlmutter/sweep/iclr/launch.sweep.sl` to include the sweep ID. The script currently shows
examples as comments; add the active line you want to run.

```bash
# Example: one run per agent
wandb agent --count 1 ENTITY/PROJECT/SWEEP_ID
```

If you want multiple parallel agents, keep the SLURM array (e.g., `#SBATCH --array=0-24`) and adjust
`--count`. Total runs = array size * count.

Make sure `WANDB_API_KEY` is set (either in your environment or in the script). The script and
`launch.hgt.sh` currently export `WANDB_API_KEY`; update or remove those lines to use your key.

## Step 3: Submit the Perlmutter job

```bash
sbatch scripts/perlmutter/sweep/iclr/launch.sweep.sl
```

Each array task runs a `wandb agent` that pulls a sweep config and invokes the command from
`opf_hgt.yaml`, which in turn runs `launch.hgt.sh`.

## What `launch.hgt.sh` runs

`launch.hgt.sh` uses the SLURM allocation and starts distributed training via `srun`:

- Base config: `configs/sweeps/iclr/config.yaml`
- Model: `--model_type=HGT`
- DDP settings derive from `SLURM_JOB_NUM_NODES` and `SLURM_GPUS_ON_NODE`
- Sweep overrides are passed in from W&B as CLI args (`"$@"`)

If you change node or GPU counts in `launch.sweep.sl`, `launch.hgt.sh` will pick them up automatically.

## Monitoring

```bash
sqs
```

Track runs in the W&B UI under the sweep project.
