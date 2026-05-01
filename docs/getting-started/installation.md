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
# Example: PyTorch with CUDA 12.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## Step 2: Install PyTorch Geometric

```bash
pip install torch-geometric
```

## Step 3: Install LUMINA

### Development install (recommended)

```bash
git clone https://github.com/argonne-gridfm/lumina-sdk.git
cd lumina-sdk
make dev
```

This installs LUMINA in editable mode (`pip install -e .`).

!!! important "Minimum supported install includes `[acopf]`"
    The base install above is **not sufficient** to load any OPF dataset — `lumina/dataset/opf/utils.py` unconditionally imports `pandapower`, which only ships with the `[acopf]` extra. Importing `OPFDataset` will raise `ModuleNotFoundError: pandapower` otherwise.

    For most workflows the minimum supported install is:

    ```bash
    pip install -e ".[acopf]"
    # or equivalently:
    make install-acopf
    ```

### With optional dependencies

```bash
# Testing
make install-test       # or: pip install -e ".[test]"

# Required for OPFDataset loading + ACOPF evaluation (pandapower, pypower)
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
from lumina.dataset.opf.opf_dataset import OPFDataset
from lumina.model.opf.losses import OPFLossManager

print(f"LUMINA installed successfully! Version {lumina.__version__}")
```

Check detected versions:

```bash
make info
```

## HPC Systems

For Polaris or Perlmutter, see the [HPC Training guide](../tutorials/hpc.md) for system-specific setup instructions.
