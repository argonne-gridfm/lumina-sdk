# Multi-Case Training

Train a single model across multiple grid topologies simultaneously using `MultiCaseOPFTrainer`.

## Overview

Multi-case training learns a unified GNN that generalizes across different power grid sizes — from 14-bus to 13,659-bus systems. Each case has its own dataset, and the trainer interleaves batches from all cases.

## Available Cases

| Case | Buses | Generators | Lines |
|------|-------|------------|-------|
| `case14` | 14 | 5 | 20 |
| `case30` | 30 | 6 | 41 |
| `case57` | 57 | 7 | 80 |
| `case118` | 118 | 54 | 186 |
| `case500` | 500 | 90 | 733 |
| `case2000` | 2,000 | 543 | 3,206 |
| `case4661` | 4,661 | 593 | 5,997 |
| `case6470` | 6,470 | 1,399 | 9,005 |
| `case10000` | 10,000 | 2,488 | 13,046 |
| `case13659` | 13,659 | 4,092 | 20,467 |

## Launch Multi-Case Training

```bash
torchrun --standalone --nproc_per_node=4 \
  example/opf/train_opf_ddp.py \
  --config configs/config.yaml \
  --cases case14 case30 case57 case118 \
  --group_ids 0 1 2 3 \
  --model_type HeteroGNN \
  --loss_type mse
```

- `--cases`: Space-separated list of case names
- `--group_ids`: Data groups to load for each case (each group = 15,000 samples)

## Data Groups

Each case has 20 groups (0-19), with ~15,000 samples each. For large-scale training:

```bash
torchrun --standalone --nproc_per_node=8 \
  example/opf/train_opf_ddp.py \
  --config configs/config.yaml \
  --cases case14 case118 case2000 \
  --group_ids 0 1 2 3 4 5 6 7 8 9
```

## Sharded Datasets

For very large datasets that don't fit in memory, use the sharded backend:

### Build shards first

```bash
python scripts/opf_build_shards.py \
  --root /path/to/data \
  --case-name pglib_opf_case2000_goc \
  --group-ids 0 1 2 3 4 5
```

### Configure sharded loading

```yaml
data:
  dataset_backend: "sharded"
  sharded_manifest_name: "manifest.json"
```

## On-Disk Datasets

For cases too large for memory but not using sharding:

```yaml
data:
  dataset_backend: "on_disk"
  on_disk_backend: "sqlite"  # or "rocksdb"
```

The `OPFOnDiskDataset` stores individual samples in a SQLite/RocksDB database, loading them on demand.

## Configuration for Multi-Case

Key config settings for multi-case training:

```yaml
training:
  max_global_samples: 2000000   # Total samples across all cases
  global_batch_size: 8192       # Effective batch size
  val_every_n_samples: 81920    # Validate every N samples

data:
  multi_case:
    case_config:
      - "pglib_opf_case14_ieee"
      - "pglib_opf_case118_ieee"
      - "pglib_opf_case2000_goc"
    group_ids: [0, 0, 0]
```

## Next Steps

- [Evaluation](evaluation.md) — Evaluate multi-case models
- [HPC Training](hpc.md) — Scale to HPC clusters
