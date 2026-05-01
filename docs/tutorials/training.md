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

The shipped `configs/config.yaml` is sized for HPC runs (e.g. `global_batch_size: 8192`, `max_epochs: 10`, `batch_size: 16`). For a small first-time run on a workstation you'll typically want a lighter override. Save the snippet below as `configs/config.case14.yaml` and pass it via `--config`:

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

!!! note "These are tutorial-friendly values, not the shipped defaults"
    See `configs/config.yaml` for the canonical values used for full-scale runs.

## 3. Launch Training

### Single process (smoke test)

For a one-GPU sanity check on a non-HPC machine, use the minimal single-process script:

```bash
python example/opf/train_opf_simple.py
```

It reads the same `configs/config.yaml` (loader / optimizer / training) and the `HeteroGNN` section of `configs/model/heterognn.yaml`. CLI flags expose only the dynamic bits:

```bash
python example/opf/train_opf_simple.py --case pglib_opf_case14_ieee --group_id 0 --device cpu
```

No DDP, MPI, checkpointing, or W&B — purely for validating your install + dataset + model wiring.

### DDP training

For real training, use `example/opf/train_opf_ddp.py`. NVIDIA GPUs are required (the process group is hardcoded to NCCL). Pick the launcher matching your environment:

=== "Local workstation (torchrun)"

    Single-machine multi-GPU. `torchrun` handles process spawning and rendezvous — no manual `MASTER_ADDR` / `MASTER_PORT` needed.

    ```bash
    torchrun --standalone --nproc_per_node=4 \
      example/opf/train_opf_ddp.py \
      --config configs/config.yaml \
      --cases case14 \
      --group_ids 0 \
      --model_type HeteroGNN \
      --loss_type mse
    ```

    Use `--nproc_per_node=1` for a single-GPU run.

=== "HPC single-node"

    Polaris (PBS + MPICH): `mpiexec -n 1 -ppn 4` to claim 4 GPUs on one node. Perlmutter (SLURM): `srun --ntasks-per-node 4`. Full job-script templates and Perlmutter `srun` invocations live in the [HPC training guide](hpc.md).

    ```bash
    # Polaris (single node, 4 GPUs)
    export MASTER_ADDR=$(hostname).hsn.cm.polaris.alcf.anl.gov
    export MASTER_PORT=29500

    mpiexec -n 1 -ppn 4 \
      python example/opf/train_opf_ddp.py \
      --config configs/config.polaris.ddp.yaml \
      --cases case14 \
      --group_ids 0
    ```

=== "HPC multi-node"

    Across multiple nodes — the full PBS / SLURM job scripts are in the [HPC training guide](hpc.md).

    ```bash
    # Polaris (NNODES nodes × 4 GPUs)
    NNODES=$(cat $PBS_NODEFILE | sort | uniq | wc -l)
    NGPUS_PER_NODE=4
    NTOTGPUS=$((NNODES * NGPUS_PER_NODE))

    export MASTER_ADDR=$(hostname).hsn.cm.polaris.alcf.anl.gov
    export MASTER_PORT=29500

    mpiexec -n ${NTOTGPUS} -ppn ${NGPUS_PER_NODE} \
      python example/opf/train_opf_ddp.py \
      --config configs/config.polaris.ddp.yaml \
      --cases case14 case118 \
      --group_ids 0 1 2 3
    ```

## 4. Model Types

LUMINA supports several GNN architectures. `--model_type` selects the architecture; `--backend` is a sub-option that only applies to `HeteroGNN` (it picks the per-relation convolution layer used inside `HeteroConv`).

| `--model_type` | `--backend` (HeteroGNN only) | Description |
|----------------|------------------------------|-------------|
| `HeteroGNN`    | `sage` / `gcn` / `gin` / `gat` | General heterogeneous message passing with a selectable conv backend |
| `RGAT`         | N/A                          | Multi-relational graph attention (per-relation `GATConv`) |
| `HEAT`         | N/A                          | Heterogeneous edge-attribute transformer |
| `HGT`          | N/A                          | Heterogeneous graph transformer |

Set the backend in the model config. The snippet below is a lightweight override suitable for case14 — the shipped `configs/model/heterognn.yaml` defaults to a much larger model (`hidden_channels: 2048`, `num_layers: 8`, `backend: "gat"`) for production runs:

```yaml
models:
  HeteroGNN:
    hidden_channels: 64
    num_layers: 4
    backend: "sage"   # or "gat", "gcn", "gin" — lowercase only
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
  --wandb_project <YOUR_WANDB_PROJECT_NAME>
```

!!! note "W&B logging requires `wandb` extra"
    The `wandb` package is not included in the base install. To enable W&B logging, install the `[hps]` extra:

    ```bash
    pip install -e ".[hps]"
    # or equivalently:
    make install-hps
    ```

### Checkpoints

Checkpoints are saved to `checkpoint_dir` and include:
- `best.pt` — Best model by validation score
- `last.pt` — Most recent model
- `epoch_N.pt` — Periodic checkpoints (if configured)

## Next Steps

- [Multi-Case Training](multi-case.md) — Train across multiple grid topologies
- [Evaluation](evaluation.md) — Evaluate model predictions against physical constraints
