# Quickstart

Train a GNN model on AC Optimal Power Flow in under 5 minutes.

## Load a Dataset

```python
from lumina.dataset.opf.opf_dataset import OPFDataset

# Download and load the IEEE 14-bus case (group 0 = 15,000 samples)
dataset = OPFDataset(root='./opf_data', case_name='pglib_opf_case14_ieee', group_id=0)

print(f"Samples: {len(dataset)}")
print(f"Node types: {dataset[0].node_types}")
print(f"Edge types: {dataset[0].edge_types}")
```

Each sample is a `HeteroData` graph with node types `bus`, `generator`, `load`, `shunt` and edge types `ac_line`, `transformer`, `generator_link`, `load_link`, `shunt_link`.

## Train with DDP

The main training entry point uses PyTorch DDP via `torchrun`:

```bash
# Single-node, 4 GPUs
torchrun --standalone --nproc_per_node=4 \
  example/opf/train_opf_ddp.py \
  --config configs/config.yaml \
  --cases case14 \
  --group_ids 0 \
  --model_type HeteroGNN \
  --loss_type mse
```

## Train Programmatically

```python
from lumina.dataset.opf.opf_dataset import OPFDataset
from lumina.model.opf.losses import OPFLossManager
from lumina.loader.opf.opf_loader import DataLoader

# Load data
dataset = OPFDataset(root='./opf_data', case_name='pglib_opf_case14_ieee')
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Create loss manager
loss_manager = OPFLossManager(loss_type='mse')

# Training loop (simplified)
for batch in loader:
    predictions = model(batch.x_dict, batch.edge_index_dict)
    loss, info = loss_manager.compute_loss(predictions, batch)
    loss.backward()
    # optimizer.step() ...
```

## Evaluate

```python
from lumina.evaluator.opf.evaluator import ACOPFConstraintEvaluator

evaluator = ACOPFConstraintEvaluator(
    voltage_limits={'vmin': vmin_tensor, 'vmax': vmax_tensor},
    generation_limits={'pmin': pmin, 'pmax': pmax, 'qmin': qmin, 'qmax': qmax},
)

violations = evaluator.evaluate_all_constraints(
    predictions=predictions,
    batch_data=batch,
)
summary = evaluator.get_violation_summary(violations)
```

## Configuration

All training parameters are controlled via YAML configs. See `configs/config.yaml` for the full reference with defaults.

Key sections:

| Section | Controls |
|---------|----------|
| `optimizer` | AdamW learning rate, weight decay |
| `scheduler` | Cosine/step LR scheduling |
| `training` | Epochs, patience, gradient clipping, batch size |
| `loader` | Batch size, workers, shuffling |
| `checkpointing` | Save frequency, monitored metric |

See the [Configuration Reference](../configuration.md) for details.

## Next Steps

- [Training Tutorial](../tutorials/training.md) — Full walkthrough with explanations
- [Multi-Case Training](../tutorials/multi-case.md) — Train across multiple grid topologies
- [API Reference](../api/dataset.md) — Complete API documentation
