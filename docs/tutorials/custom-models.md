# Custom Models

This guide covers adding new GNN architectures and loss functions to LUMINA.

## Model Architecture

LUMINA models follow a two-layer structure:

1. **Base models** (`lumina/model/base/`) — Generic GNN shells (HeteroGNN, HomoGNN)
2. **OPF models** (`lumina/model/opf/`) — Task-specific wrappers with OPF output heads

## Adding a New Heterogeneous GNN Backend

### Step 1: Define the Backend

Add your GNN layer to `lumina/model/base/hetero_gnn.py`:

```python
from torch_geometric.nn import HeteroConv, SAGEConv

class MyCustomGNN(HGNN_Base):
    """Custom heterogeneous GNN backend."""
    
    def __init__(self, metadata, in_channels, hidden_channels, num_layers, **kwargs):
        super().__init__()
        self.convs = nn.ModuleList()
        
        for i in range(num_layers):
            conv_dict = {}
            for edge_type in metadata[1]:
                src_channels = in_channels if i == 0 else hidden_channels
                conv_dict[edge_type] = SAGEConv(src_channels, hidden_channels)
            self.convs.append(HeteroConv(conv_dict))
    
    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        return x_dict
```

### Step 2: Register in ModelFactory

```python
# In lumina/model/base/hetero_gnn.py
class ModelFactory:
    @staticmethod
    def create_model(model_type, metadata, in_channels, hidden_channels, ...):
        if model_type == "MyCustomGNN":
            return MyCustomGNN(metadata, in_channels, hidden_channels, ...)
        # ... existing backends
```

### Step 3: Add OPF Wrapper (optional)

If your model needs OPF-specific output processing, add a wrapper in `lumina/model/opf/hetero_model.py`:

```python
class OPFMyCustomGNN(nn.Module):
    def __init__(self, metadata, in_channels_dict, hidden_channels, num_layers, ...):
        super().__init__()
        self.encoder = MyCustomGNN(metadata, in_channels_dict, hidden_channels, num_layers)
        # Add output heads for bus and generator predictions
        self.bus_head = nn.Linear(hidden_channels, 2)      # va, vm
        self.gen_head = nn.Linear(hidden_channels, 2)      # pg, qg
```

### Step 4: Use in Training

```bash
torchrun --standalone --nproc_per_node=4 \
  example/opf/train_opf_ddp.py \
  --model_type MyCustomGNN \
  --config configs/config.yaml \
  --cases case14
```

## Adding a Custom Loss Function

### Extend ACOPFLossFunction

```python
from lumina.model.opf.losses import ACOPFLossFunction

class WeightedMSELoss(ACOPFLossFunction):
    """MSE with per-feature weighting."""
    
    def __init__(self, feature_weights=None, **kwargs):
        super().__init__(loss_type='mse', **kwargs)
        self.feature_weights = feature_weights
    
    def _compute_single_loss(self, predictions, targets):
        diff = (predictions - targets) ** 2
        if self.feature_weights is not None:
            diff = diff * self.feature_weights
        return diff.mean()
```

### Use with OPFLossManager

The `OPFLossManager` wraps any `ACOPFLossFunction`:

```python
manager = OPFLossManager(loss_type='mse', node_weights={'bus': 2.0, 'generator': 1.0})
loss, info = manager.compute_loss(predictions, batch)
```

## Homogeneous Models

For homogeneous graph models, see `lumina/model/base/homo_gnn.py`. The key classes are:

- `GNNBase` — Base class for all homogeneous GNNs
- `GNNPool` — Readout/pooling layer
- `GNN_basic`, `GAT`, `GCN`, `GIN`, `TRANSFORMER` — Concrete implementations

Use with `OPFHomogeneousDataset` and the homogeneous data pipeline in `lumina/utils/graph_utils.py`.

## Next Steps

- [HPC Training](hpc.md) — Deploy on HPC clusters
- [Model API Reference](../api/model.md) — Full model API
