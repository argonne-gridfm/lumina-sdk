# Tutorials

Step-by-step guides for common workflows, from beginner to advanced.

## Beginner

| Tutorial | Description |
|----------|-------------|
| [Training Your First Model](training.md) | Load case14 data, configure training, launch with torchrun |
| [Evaluation](evaluation.md) | Load checkpoints, run inference, check constraint violations |

## Advanced

| Tutorial | Description |
|----------|-------------|
| [Multi-Case Training](multi-case.md) | Train across multiple grid topologies with DDP |
| [Custom Models](custom-models.md) | Add new GNN backends, register in ModelFactory |
| [HPC Training](hpc.md) | Polaris, Perlmutter setup and job scripts |

## Notebooks

Runnable Jupyter notebooks walking through end-to-end workflows on case14.

| Notebook | Description |
|----------|-------------|
| [Quickstart (case14)](notebooks/01_quickstart_case14.ipynb) | Smallest end-to-end run: load, build, train, plot |
| [Dataset exploration](notebooks/02_dataset_exploration.ipynb) | Inspect node/edge types, schemas, target distributions |
| [Evaluation walkthrough](notebooks/03_evaluation_walkthrough.ipynb) | Load a checkpoint and measure constraint violations |
| [Physics-informed loss](notebooks/04_physics_informed_loss.ipynb) | MSE vs. MSE + power-balance penalty, side by side |
| [DDP on local GPUs](notebooks/05_ddp_local_gpus.ipynb) | Multi-GPU training on a single workstation via `torchrun` |
| [Polaris (HPC)](notebooks/06_polaris_hpc.ipynb) | PBS job script and rendezvous setup for ALCF Polaris |
