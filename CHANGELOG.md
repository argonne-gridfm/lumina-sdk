# Changelog

All notable releases of `lumina-sdk` are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to [PEP 440](https://peps.python.org/pep-0440/).

## [Unreleased]

### Added

- Added smoke tests for Frontier preprocessing helpers in `scripts/data_process_frontier.py` (list parsing and SLURM rank/size env resolution).

### Changed

- Improved cross-platform DDP GPU binding so Perlmutter/Polaris use `LOCAL_RANK` when all GPUs are visible, while Frontier remains compatible with per-rank `ROCR_VISIBLE_DEVICES` launch wrappers.
- Frontier training now supports runtime path overrides via CLI flags (`--root`, `--logging_dir`, `--checkpoint_dir`) or env vars (`LUMINA_ROOT`, `LUMINA_LOGGING_DIR`, `LUMINA_CHECKPOINT_DIR`), removing site-specific hardcoded paths from shipped configs.
- Removed unreleased Lagrangian settings from Frontier-facing configs and removed Lagrangian loss options from the Frontier training CLI.

## [v0.1.0rc1] — 2026-04-30

First public release candidate.

### Added

- **Datasets** — `OPFDataset` (in-memory), `OPFOnDiskDataset` (SQLite/RocksDB), `OPFShardedIterableDataset`, plus a `to_float32` PyG transform for fp64 → fp32 casting.
- **Models** — Heterogeneous GNN architectures: `OPFHeteroGNN` (with `sage` / `gcn` / `gin` / `gat` backends), `RGAT`, `HEAT`, `HGT`; homogeneous variants under `OPFHomoGNN`.
- **Losses** — `OPFLossManager` with MSE / RMSE / MAE / MAPE / SmoothL1; `PhysicsInformedLoss` skeleton with quadratic / absolute / log-barrier penalties.
- **Trainer** — Plain PyTorch DDP via MPI; gradient clipping; non-finite loss handling; W&B integration; sample-based scheduling.
- **Evaluator** — `ACOPFConstraintEvaluator` for voltage / generation / power-balance / thermal-limit violations; `Modeler` for checkpoint loading and end-to-end prediction.
- **Examples** — `train_opf_simple.py` (single-process smoke test), `train_opf_ddp.py` (multi-GPU DDP), `evaluate_opf_constraint.py` (constraint metrics).
- **Backends** — Tested on CPU and NVIDIA CUDA; HPC job templates for Polaris (PBS / MPICH) and Perlmutter (SLURM).
- **Docs** — Material for MkDocs site at <https://argonne-gridfm.github.io/lumina-sdk/> with quickstart, training tutorial, evaluation, multi-case, HPC, and 6 runnable Jupyter notebooks.
