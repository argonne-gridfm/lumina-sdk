# Evaluation

This tutorial covers evaluating trained models against physical constraint violations.

## Overview

LUMINA provides two levels of evaluation:

1. **Loss-based** — MSE/RMSE between predictions and solver solutions
2. **Constraint-based** — Physical feasibility checks (voltage bounds, generation limits)

## Load a Trained Model

```python
from lumina.evaluator.opf.utils import Modeler

modeler = Modeler(
    device=torch.device('cpu'),
    repo_id="argonne-gridfm/lumina-case14",
    token="hf_your_token",
)
model, config = modeler.load_model()
```

Or from a local checkpoint:

```python
modeler = Modeler(device=torch.device('cpu'))
model = modeler.load_model_from_training_checkpoint("checkpoints/best.pt")
```

## Run Predictions

```python
from lumina.dataset.opf.opf_dataset import OPFDataset
from lumina.loader.opf.opf_loader import DataLoader

dataset = OPFDataset(root='./opf_data', case_name='pglib_opf_case14_ieee')
loader = DataLoader(dataset, batch_size=32, shuffle=False)

pred_batch_pairs = modeler.run_predictions(model, loader, max_batches=10)
```

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
