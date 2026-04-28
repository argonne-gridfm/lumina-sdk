# LUMINA - A Large-scale Unified Model for Intelligent Grid Applications

Lumina Core is the core package for LUMINA, a large-scale unified model for intelligent grid applications. It provides essential functionalities for data processing, model architecture, evaluator and more. 

## Key Features

- **Multiple Loss Functions**: Support for standard ML losses (MSE, RMSE, MAE, MAPE, SmoothL1) and physics-informed training
- **Flexible Architecture**: Heterogeneous and homogeneous GNN models for power grid applications
- **Scalable Training**: PyTorch Lightning integration for distributed training
- **Comprehensive Evaluation**: Built-in evaluators for OPF and other grid optimization tasks

## Quick Start

Run distributed training with `example/opf/train_opf_ddp.py` (works on single node or multi-node via `torchrun`):
```bash
torchrun --standalone --nproc_per_node=4 \
  example/opf/train_opf_ddp.py \
  --config configs/config.yaml \
  --cases case14 \
  --group_ids 0 \
  --model_type HeteroGNN \
  --loss_type mse
```

## Installation

### Prerequisites
- Python 3.10+
- PyTorch and PyTorch Geometric installed for your platform (CPU/CUDA/ROCm)

### Install from source
```shell
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install PyTorch and PyG using their official instructions:
# https://pytorch.org/get-started/locally/
# https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
pip install torch torch-geometric

# Install lumina-core with common extras
pip install -e ".[acopf,hps,benchmark]"

# Or install everything
pip install -e ".[all]"
```

Check `Makefile` for additional optional dependency targets:
```shell
make help
```

If you use a shared HPC environment, see the activation notes in `docs/perlmutter.md` or `docs/polaris.md`.

## OPFData dataset

OPFData is distributed as raw `.tar.gz` files (15,000 samples per group). Lumina expects the following layout under your dataset root:
```text
<root>/OPFData/raw/<release>/*.tar.gz
<root>/OPFData/processed/<release>/<case_name>/group_<id>.pt
```

Set `root` in your config to the parent directory that contains `OPFData`. For example:
```yaml
root: "/path/to/datasets"
```

On NERSC Perlmutter, the raw and processed datasets are available at `$CFS/amsc004/datasets/OPFData` (set `root: $CFS/amsc004/datasets`).

## Data preprocessing (recommended)

`scripts/data_process.py` converts raw OPFData groups into `.pt` files under `OPFData/processed`, and also writes homogeneous variants. It is CPU/IO heavy, so run it as a separate preprocessing job before training.

Before running, edit:
- `root` to point at your dataset directory
- `case_mapping` and `group_ids` for the cases/groups you want

Example:
```shell
python scripts/data_process.py
# or launch multiple ranks with MPI/SLURM for faster preprocessing
```

## Large cases (use on-disk backend)

For large cases, use the on-disk dataset backend to avoid loading full groups into memory. Set the following in your config:
```yaml
data:
  backend: "on_disk"
  on_disk_backend: "sqlite"
```

This writes per-group databases under `<root>/OPFData/on_disk` on first access.

## Distributed training on OPFData (DDP)

`example/opf/train_opf_ddp.py` trains on OPFData with PyTorch DDP. It accepts short case names (e.g., `case14`) or full PGLIB names and expects preprocessed groups under `OPFData/processed`.

Example (single node):
```shell
torchrun --standalone --nproc_per_node=4 \
  example/opf/train_opf_ddp.py \
  --config configs/config.yaml \
  --cases case14 case30 \
  --group_ids 0 1 \
  --model_type HGT \
  --loss_type mse
```

<!-- 
## Instrall for other repo

# Install the latest from main (good for dev, risky for prod)
git+ssh://git@github.com/argonne-gridmf/lumina-core.git

# Install a specific tag (Recommended for stability)
git+ssh://git@github.com/argonne-gridmf/lumina-core.git@v0.1.0

[project]
dependencies = [
    "lumina-core @ git+ssh://git@github.com/argonne-gridmf/lumina-core.git"
] 
-->

## LICENSE

Copyright (c) 2025, Argonne National Laboratory  
All rights reserved.
