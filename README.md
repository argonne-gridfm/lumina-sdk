# LUMINA — Large-scale Unified Model for Intelligent Grid Applications

[![Version](https://img.shields.io/badge/version-0.1.0rc1-blue.svg)](CHANGELOG.md)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.9-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-%E2%89%A52.7-3C2179.svg)](https://pytorch-geometric.readthedocs.io/)
[![Docs](https://img.shields.io/badge/docs-mkdocs--material-526CFE.svg?logo=materialformkdocs&logoColor=white)](https://argonne-gridfm.github.io/lumina-sdk/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

`lumina-sdk` is a PyTorch / PyTorch Geometric framework for training graph neural networks on power grid problems, including AC Optimal Power Flow (ACOPF) and related power-grid optimization tasks.

📖 **Docs:** <https://argonne-gridfm.github.io/lumina-sdk/>

## Key Features

- **Datasets** — `OPFDataset` (in-memory), `OPFOnDiskDataset` (SQLite/RocksDB), `OPFShardedIterableDataset` (sharded for HPC).
- **Models** — Heterogeneous (`OPFHeteroGNN` with SAGE/GCN/GIN/GAT backends, `RGAT`, `HEAT`, `HGT`) and homogeneous GNN architectures.
- **Losses** — Standard ML losses (MSE, RMSE, MAE, MAPE, SmoothL1) plus a `PhysicsInformedLoss` skeleton for power-balance penalties.
- **Trainer** — Plain PyTorch DDP via MPI (`mpirun` / `mpiexec` / `srun`); single-process script for local smoke tests; W&B logging, checkpointing, gradient clipping, sample-based scheduling.
- **Evaluator** — `ACOPFConstraintEvaluator` for voltage / generation / power-balance / thermal-limit violation checks against solver targets.

## Installation

Requires Python 3.10+ and a working PyTorch install.

```bash
# 1. Install PyTorch + PyG following their official instructions for your CUDA version
pip install torch torch-geometric

# 2. Install lumina-sdk in editable mode with the acopf extra (required for OPFDataset)
pip install -e ".[acopf]"

# Or pull in everything (test, dev, hps, doc, benchmark):
pip install -e ".[all]"
```

See `Makefile` for shortcut targets (`make dev`, `make install-acopf`, ...).

## Quick start

Local single-process smoke test (no DDP, no MPI):

```bash
python example/opf/train_opf_simple.py
```

Multi-GPU DDP via MPI (HPC-style):

```bash
mpirun -np 4 python example/opf/train_opf_ddp.py \
  --config configs/config.yaml \
  --cases case30 \
  --group_ids 0 \
  --model_type HeteroGNN \
  --loss_type mse
```

For `torchrun` and HPC-specific (Polaris / Perlmutter) job-script templates, see the [HPC training guide](https://argonne-gridfm.github.io/lumina-sdk/tutorials/hpc/).

## Dataset layout

`OPFDataset` looks for raw archives + processed groups under your dataset root:

```
<root>/OPFData/raw/<release>/<case>_<group>.tar.gz
<root>/OPFData/processed/<release>/<case>/group_<id>.pt
```

If the raw archive is missing, the dataset downloads it from Google Cloud Storage on first use. To pre-process many cases / groups in bulk, use `python scripts/data_process.py --root <DATASET_ROOT>` (MPI-aware).

## License

Copyright © 2026 UChicago Argonne, LLC.

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for details.
