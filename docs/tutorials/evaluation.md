# Evaluation

This tutorial covers evaluating trained models against physical constraint violations.

## Overview

LUMINA provides two levels of evaluation:

1. **Loss-based** — MSE/RMSE between predictions and solver solutions
2. **Constraint-based** — Physical feasibility checks (voltage bounds, generation limits)

## Load a Trained Model

The canonical way is to point `Modeler.load_model_from_training_checkpoint` at a checkpoint produced by `example/opf/train_opf_ddp.py` (or `train_opf_simple.py`):

```python
import torch
from lumina.evaluator.opf.utils import Modeler

modeler = Modeler(device=torch.device('cpu'))
model = modeler.load_model_from_training_checkpoint("checkpoints/best.pt")
```

`Modeler.__init__` is keyword-only beyond `device`. The defaults are usually fine; the knobs that matter for downstream constraint evaluation are:

```python
modeler = Modeler(
    device=torch.device('cpu'),
    fail_on_missing=False,    # raise if a constraint input is missing
    verbose=True,
    base_mva=100.0,
    slack_bus_indices='0',    # comma-separated string
)
```

For lower-level control, `Modeler.load_model` accepts a config dict + state dict directly (e.g., when loading from a non-checkpoint source) and returns the constructed model alongside the parsed config:

```python
model, config = modeler.load_model(config_data=config_dict, state_dict=state_dict)
```

Both `load_model_from_training_checkpoint` and `load_model` cache the constructed model on `modeler.model`, so subsequent calls like `run_predictions` pick it up implicitly.

## Run Predictions

```python
from lumina.dataset.opf.opf_dataset import OPFDataset
from lumina.dataset.opf.transforms import to_float32
from lumina.loader.opf.opf_loader import DataLoader

dataset = OPFDataset(root='./opf_data', case_name='pglib_opf_case14_ieee', transform=to_float32)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

pred_batch_pairs = modeler.run_predictions(loader, max_batches=10)
```

`run_predictions(loader, max_batches=None, minmax_scaling=True)` uses the model previously loaded by `load_model*`. Set `minmax_scaling=False` to skip the per-bus voltage / per-generator power scaling applied to model outputs.

## Evaluate Constraint Violations

### Bound Constraints

Check voltage magnitude and generation limit violations:

```python
from lumina.evaluator.opf.evaluator import ACOPFConstraintEvaluator

evaluator = ACOPFConstraintEvaluator(
    voltage_limits={'vmin': vmin, 'vmax': vmax},
    generation_limits={'pmin': pmin, 'pmax': pmax, 'qmin': qmin, 'qmax': qmax},
    device=torch.device('cpu'),
)

violations = evaluator.evaluate_all_constraints(
    predictions=predictions,
    batch_data=batch,
    normalize=True,
    return_individual=False,
)

summary = evaluator.get_violation_summary(violations)
for name, value in summary.items():
    print(f"{name}: {value:.6f}")
```

### Using Modeler for End-to-End Evaluation

The `Modeler` class provides a streamlined evaluation pipeline:

```python
stats = modeler.evaluate_from_predictions(
    pred_batch_pairs,
    normalize=True,
    cache_key="case14",
)

for metric, values in stats.items():
    print(f"{metric}: mean={values['mean']:.6f}, var={values['var']:.6f}")
```

## Extract Network Parameters

Helper functions extract limits directly from batch data:

```python
from lumina.evaluator.opf.utils import (
    extract_voltage_and_generation_limits_from_batch,
    extract_network_parameters_from_batch,
)

voltage_limits, generation_limits = extract_voltage_and_generation_limits_from_batch(batch)
network_params = extract_network_parameters_from_batch(batch)
```

## Loss-Based Evaluation

```python
from lumina.model.opf.losses import OPFLossManager

loss_manager = OPFLossManager(loss_type='mse')
loss_manager.eval()

with torch.no_grad():
    for batch in loader:
        predictions = model(batch.x_dict, batch.edge_index_dict)
        loss, info = loss_manager.compute_loss(predictions, batch)
        print(f"Loss: {loss.item():.6f}")
```

## Next Steps

- [Custom Models](custom-models.md) — Build your own GNN architecture
- [API Reference](../api/evaluator.md) — Full evaluator API
