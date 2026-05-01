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

# Expected OUTPUT:
# Samples: 15000
# Node types: ['bus', 'generator', 'load', 'shunt']
# Edge types: [('bus', 'ac_line', 'bus'), ('bus', 'transformer', 'bus'), ('generator', 'generator_link', 'bus'), 
#   ('bus', 'generator_link', 'generator'), ('load', 'load_link', 'bus'), ('bus', 'load_link', 'load'), 
#   ('shunt', 'shunt_link', 'bus'), ('bus', 'shunt_link', 'shunt')]
```

Each sample is a `HeteroData` graph with node types `bus`, `generator`, `load`, `shunt` and edge types `ac_line`, `transformer`, `generator_link`, `load_link`, `shunt_link`.

## Train on a local machine

For a quick smoke test on a single workstation, use the minimal single-process script. It reads `configs/config.yaml` (loader / optimizer / training) and the `HeteroGNN` section of `configs/model/heterognn.yaml`, but skips DDP / MPI entirely:

```bash
python example/opf/train_opf_simple.py
```

Defaults: case14 group 0; CUDA if available, else CPU. Override case / group / device with CLI flags:

```bash
python example/opf/train_opf_simple.py --case pglib_opf_case30_ieee --group_id 0 --device cpu
```

Per-epoch output looks like:

```
epoch  1/10 | train_loss=0.0421 | val_loss=0.0357
epoch  2/10 | train_loss=0.0289 | val_loss=0.0271
...
```

Nothing is checkpointed or logged — the script is purely for validating the install + dataset + model wiring. Model architecture (`hidden_channels`, `num_layers`, `backend`) comes from `configs/model/heterognn.yaml`; the shipped defaults are GAT with `hidden_channels=2048`, `num_layers=8`. Edit that file to shrink the model if CPU iteration is too slow.

## Train with DDP (multi-GPU on HPC)

The full DDP entry point `example/opf/train_opf_ddp.py` is intended for HPC clusters (Polaris, NERSC Perlmutter) where the launcher environment provides the rendezvous and rank variables. It requires NVIDIA GPUs (the process group is hardcoded to NCCL).

```bash
torchrun --standalone --nproc_per_node=4 \
  example/opf/train_opf_ddp.py \
  --config configs/config.yaml \
  --cases case14 \
  --group_ids 0 \
  --model_type HeteroGNN \
  --loss_type mse
```

For Polaris- or Perlmutter-specific `mpiexec` / `srun` invocations and job-script templates, see the [HPC training guide](../tutorials/hpc.md).

!!! info "Local workstation users"
    On a non-HPC machine, the DDP launcher above won't work out of the box (it relies on env vars typically set by the cluster job launcher). Use the [single-process script](#train-on-a-local-machine) above for local smoke testing.

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

| Section         | Controls                                                     |
| --------------- | ------------------------------------------------------------ |
| `optimizer`     | AdamW learning rate, weight decay                            |
| `scheduler`     | Cosine/step LR scheduling                                    |
| `training`      | Epochs, patience, gradient clipping, sample-based scheduling |
| `loader`        | Batch size, workers, shuffling                               |
| `checkpointing` | Save frequency, monitored metric                             |

See the [Configuration Reference](../configuration.md) for details.

## Next Steps

- [Training Tutorial](../tutorials/training.md) — Full walkthrough with explanations
- [Multi-Case Training](../tutorials/multi-case.md) — Train across multiple grid topologies
- [API Reference](../api/dataset.md) — Complete API documentation
