# Installation

## Prerequisites

- Python 3.10+
- PyTorch >= 2.0 (with CUDA support for GPU training)
- PyTorch Geometric >= 2.7.0

!!! warning "Install PyTorch first"
    PyTorch and PyTorch Geometric must be installed **before** installing LUMINA, as they are not auto-installed by pip.

## Step 1: Install PyTorch

Follow the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) for your platform and CUDA version.

```bash
# Example: PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Step 2: Install PyTorch Geometric

```bash
pip install torch-geometric
```

## Step 3: Install LUMINA

### Development install (recommended)

```bash
git clone https://github.com/argonne-gridfm/lumina-core.git
cd lumina-core
make dev
```

This installs LUMINA in editable mode (`pip install -e .`).

### With optional dependencies

```bash
# Testing
make install-test       # or: pip install -e ".[test]"

# ACOPF evaluation (pandapower, pypower)
make install-acopf      # or: pip install -e ".[acopf]"

# Hyperparameter search (wandb, optuna)
make install-hps        # or: pip install -e ".[hps]"

# Documentation
pip install -e ".[doc]"

# Everything
make install-all        # or: pip install -e ".[test,acopf,hps]"
```

## Verify Installation

```python
import lumina
from lumina.dataset.opf import OPFDataset
from lumina.model.opf.losses import OPFLossManager

print("LUMINA installed successfully!")
```

Check detected versions:

```bash
make info
```

## HPC Systems

For Polaris or Perlmutter, see the [HPC Training guide](../tutorials/hpc.md) for system-specific setup instructions.
