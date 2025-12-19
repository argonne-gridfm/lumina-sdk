# W&B Sweep on HPC (Polaris/Summit/etc.)

This guide explains how to run W&B hyperparameter sweeps on HPC systems where compute nodes don't have internet access.

## Architecture

- **Login Node**: Runs W&B agent (has internet) → Submits SLURM jobs
- **Compute Nodes**: Run training (no internet) → Log to W&B offline mode
- **After Jobs Complete**: Sync offline logs back to W&B servers

## Setup

### 1. On Login Node (with internet)

```bash
# Activate your environment
source venv/bin/activate

# Create sweep (one time)
wandb sweep configs/sweeps/opf_default.yaml

# This will output a sweep ID like: zeesh-an-emory-university/lumina-core/abc123xyz
```

### 2. Start W&B Agent

```bash
# Run agent on login node
wandb agent zeesh-an-emory-university/lumina-core/<sweep-id>
```

The agent will:
- Pull hyperparameter configs from W&B
- Call `train_opf.py` with those configs
- `train_opf.py` detects it's running under agent (checks `WANDB_SWEEP_ID` env var)
- Submits a SLURM job for each run
- Training happens on compute nodes in offline mode

### 3. Training on Compute Nodes

Each SLURM job:
- Runs on a compute node (no internet)
- Sets `WANDB_MODE=offline`
- Trains the model
- Logs metrics to local `wandb/offline-run-*` directories

### 4. Sync Offline Logs (After Jobs Complete)

After your SLURM jobs finish, sync the offline logs:

```bash
# Option 1: Use the helper script
./scripts/sync_wandb_offline.sh

# Option 2: Manual sync
find wandb -name "offline-run-*" -type d | while read dir; do
    wandb sync "$dir"
done
```

## SLURM Job Configuration

The SLURM job template is generated automatically in `train_opf.py`. You can customize it by editing the `submit_slurm_job()` function.

Default settings:
- `--time=24:00:00`
- `--nodes=1`
- `--gres=gpu:1`
- `--partition=gpuA100x4` (adjust for your system)

To customize, edit these lines in `train_opf.py`:
```python
#SBATCH --partition=YOUR_PARTITION
#SBATCH --time=YOUR_TIME_LIMIT
#SBATCH --gres=gpu:YOUR_GPU_COUNT
```

## Troubleshooting

### Agent keeps submitting jobs but they fail
- Check SLURM logs: `slurm_logs/opf_train_*.out`
- Verify your partition name is correct
- Check that modules (python, cuda) load correctly

### Offline logs not syncing
- Make sure you're on login node (has internet)
- Check that `wandb sync` command works: `wandb sync wandb/offline-run-*/`

### Jobs stuck in queue
- Check queue status: `squeue -u $USER`
- Verify partition exists: `sinfo`
- Adjust time limit or partition in SLURM template

## Manual Testing

Test the SLURM submission without W&B agent:

```bash
# Set environment variables to simulate agent
export WANDB_SWEEP_ID=test
export WANDB_RUN_ID=test123

# Run script (should submit SLURM job)
python example/opf/train_opf.py --wandb --config configs/config.yaml --case case14
```

Check the submitted job:
```bash
squeue -u $USER
```
