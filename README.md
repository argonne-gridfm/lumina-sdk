# LUMINA - A Large-scale Unified Model for Intelligent Grid Applications

Lumina Core is the core package for LUMINA, a large-scale unified model for intelligent grid applications. It provides essential functionalities for data processing, model architecture, evaluator and more. 

## Key Features

- **Multiple Loss Functions**: Support for standard ML losses, Augmented Lagrangian, and Violated Lagrangian for physics-informed training
- **Flexible Architecture**: Heterogeneous and homogeneous GNN models for power grid applications
- **Scalable Training**: PyTorch Lightning integration for distributed training
- **Comprehensive Evaluation**: Built-in evaluators for OPF and other grid optimization tasks

## Quick Start

### Basic Training (MSE Loss)
```bash
python example/opf/train_opf.py --case case14 --loss_type mse
```

### Physics-Informed Training (Augmented Lagrangian)
```bash
python example/opf/train_opf.py --case case14 --loss_type augmented_lagrangian
```

### Violation-Based Training (Violated Lagrangian)
```bash
python example/opf/train_opf.py --case case14 --loss_type violated_lagrangian
```

For detailed information about loss functions and training strategies, see [Loss Functions Documentation](docs/LOSS_FUNCTIONS.md). 

## Install Instructions for Lumina Core

### ALCF Polaris

```shell
bash install/polaris/create_envs.sh
```

See `docs/polaris.md`.

### Generis systems

1. create and activate a virtual environment (recommended)

```
python -m venv .venv
source .venv/bin/activate
```

2. install Lumina Core package and optional dependencies

- install the general package in editable mode
```
pip install -e . 
```

- install optional dependencies as needed, e.g., for cuGraph support
```
pip install -e .[cuGraph] -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
```

- (Recommended) install all optional dependencies
```
pip install -e .[all] -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
```

- checkout `Makefile` for other optional dependencies installation commands.
```
make help
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
