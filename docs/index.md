# LUMINA-SDK

**Large-scale Unified Model for Intelligent Grid Applications**

LUMINA-SDK is a PyTorch/PyTorch Geometric framework for training AI models (e.g., GNNs) for power grid problems (e.g., AC Optimal Power Flow). It provides a complete pipeline from data loading through distributed training to constraint-aware evaluation.

## Key Features

- **Multiple GNN Architectures** — HeteroGNN with SAGE, GAT, GCN, GIN, HGT, HEAT, and Transformer backends, plus homogeneous model variants
- **Flexible Loss Functions** — MSE, RMSE, MAE, MAPE, SmoothL1 with per-node-type weighting and physics-informed penalties
- **Scalable Training** — PyTorch DDP with multi-GPU, multi-node support on Polaris and Perlmutter HPC systems
- **Large-Scale Data** — In-memory, SQLite/RocksDB on-disk, and sharded iterable datasets for cases from 14 to 13,659 buses
- **Constraint Evaluation** — Voltage bounds, generation limits, and power balance violation checking
- **Experiment Tracking** — Weights & Biases integration with metric logging and checkpointing

## Architecture

```
lumina/
├── dataset/opf/     # Data loading: in-memory, on-disk, sharded
├── model/
│   ├── base/        # Generic hetero/homo GNN shells
│   └── opf/         # OPF-specific models and losses
├── trainer/opf/     # DDP training loops with checkpointing
├── evaluator/opf/   # Constraint violation checking
├── loader/opf/      # Custom PyG DataLoader wrappers
└── utils/           # Graph construction, I/O, throughput
```

## Quick Links

- [Installation](getting-started/installation.md) — Get up and running
- [Quickstart](getting-started/quickstart.md) — Train your first model in minutes
- [API Reference](api/dataset.md) — Full API documentation
- [Tutorials](tutorials/training.md) — Step-by-step guides


## LICENSE

Copyright (c) 2026, Argonne National Laboratory