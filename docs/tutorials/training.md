# Training Your First Model

This tutorial walks through training a heterogeneous GNN on the IEEE 14-bus ACOPF problem.

## 1. Prepare Data

LUMINA automatically downloads and processes OPFData when you create a dataset:

```python
from lumina.dataset.opf.opf_dataset import OPFDataset

dataset = OPFDataset(
    root='./opf_data',
    case_name='pglib_opf_case14_ieee',
    group_id=0,  # Each group has ~15,000 samples
)
print(f"Loaded {len(dataset)} samples")
```

Each sample is a `HeteroData` graph:

```python
sample = dataset[0]
print(sample)
# HeteroData(
#   bus={ x=[14, 4], y=[14, 2] },
#   generator={ x=[5, 11], y=[5, 2] },
#   load={ x=[11, 2] },
#   shunt={ x=[1, 2] },
#   (bus, ac_line, bus)={ edge_index=[2, 20], edge_attr=[20, 9] },
#   ...
# )
```

- **bus.x**: Bus features (base_kv, bus_type, vmin, vmax)
- **bus.y**: Bus targets (voltage angle, voltage magnitude)
- **generator.y**: Generator targets (active power, reactive power)

## 2. Configure Training

Create or modify a YAML config file. The default `configs/config.yaml` works for small cases:

```yaml
root: "./opf_data/"
checkpoint_dir: "./checkpoints/"

optimizer:
  AdamW:
    lr: 1.0e-03
    weight_decay: 0.01

training:
  max_epochs: 50
  gradient_clip_val: 1.0
  global_batch_size: 256

loader:
  batch_size: 32
  num_workers: 4
```

## 3. Launch Training

### Using torchrun (recommended for multi-GPU)

```bash
torchrun --standalone --nproc_per_node=4 \
  example/opf/train_opf_ddp.py \
  --config configs/config.yaml \
  --cases case14 \
  --group_ids 0 \
  --model_type HeteroGNN \
  --loss_type mse
```

### Single GPU

```bash
torchrun --standalone --nproc_per_node=1 \
  example/opf/train_opf_ddp.py \
  --config configs/config.yaml \
  --cases case14 \
  --group_ids 0
```

## 4. Model Types

LUMINA supports several GNN architectures:

| Model | Backend | Description |
|-------|---------|-------------|
| `HeteroGNN` | SAGE/GAT/GCN/GIN | General heterogeneous message passing |
| `RGAT` | Relational GAT | Multi-relational graph attention |
| `HEAT` | HEAT | Heterogeneous edge attribute transformer |
| `HGT` | HGT | Heterogeneous graph transformer |

Set the backend in the model config:

```yaml
models:
  HeteroGNN:
    hidden_channels: 64
    num_layers: 4
    backend: "SAGE"   # or "GAT", "GCN", "GIN"
```

## 5. Loss Functions

Available loss types:

| Loss Type | Description |
|-----------|-------------|
| `mse` | Mean Squared Error (default) |
| `rmse` | Root Mean Squared Error |
| `mae` | Mean Absolute Error |
| `mape` | Mean Absolute Percentage Error |
| `smooth_l1` | Smooth L1 / Huber Loss |

## 6. Monitor Training

### With Weights & Biases

```bash
torchrun --standalone --nproc_per_node=4 \
  example/opf/train_opf_ddp.py \
  --config configs/config.yaml \
  --cases case14 \
  --group_ids 0 \
  --wandb \
  --wandb_project "lumina-training"
```

### Checkpoints

Checkpoints are saved to `checkpoint_dir` and include:
- `best.pt` — Best model by validation score
- `last.pt` — Most recent model
- `epoch_N.pt` — Periodic checkpoints (if configured)

## Next Steps

- [Multi-Case Training](multi-case.md) — Train across multiple grid topologies
- [Evaluation](evaluation.md) — Evaluate model predictions against physical constraints
