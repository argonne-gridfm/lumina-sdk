# Configuration Reference

All training parameters are controlled via YAML config files in `configs/`. The canonical reference is `configs/config.yaml`.

## Config Structure

```yaml
root: "./opf_data/"              # Dataset root directory
logging_dir: "./logs/"           # Log output directory
checkpoint_dir: "./ddp_checkpoints/"  # Checkpoint save directory
```

## Optimizer

```yaml
optimizer:
  AdamW:
    lr: 1.0e-03           # Learning rate
    betas: [0.9, 0.999]   # Adam beta parameters
    eps: 1.0e-08           # Numerical stability
    weight_decay: 0.01     # L2 regularization
```

## Scheduler

```yaml
scheduler:
  type: "cosine"     # Options: "cosine", "step", "plateau"
  t_max: null        # Max iterations for cosine (null = max_epochs)
  eta_min: 0.0       # Minimum learning rate
```

## Data

```yaml
train_split: 0.8     # Fraction of data for training
val_split: 0.1       # Fraction for validation (rest is test)

loader:
  batch_size: 16
  shuffle: true
  num_workers: 0     # DataLoader workers (0 = main process)
```

## Training

```yaml
training:
  max_epochs: 10
  patience: 100000              # Early stopping patience
  max_global_samples: 500000    # Stop after this many samples
  global_batch_size: 8192       # Effective batch size across GPUs
  gradient_clip_val: 1.0        # Gradient norm clipping
  gradient_clip_algorithm: "norm"
  fail_on_nonfinite: false      # Raise on NaN loss (false = skip)
  accumulate_grad_batches: 1    # Gradient accumulation steps
  log_every_n_steps: 0          # W&B log interval (0 = every step)
  log_every_n_samples: 8192     # Log every N samples
  val_every_n_epochs: 0         # Validate every N epochs (0 = sample-based)
  val_every_n_samples: 40960    # Validate every N samples
  throughput_enabled: false     # Enable throughput measurement
```

## Checkpointing

```yaml
checkpointing:
  monitor: "val/score"     # Metric to monitor for best model
  save_last: true          # Always save last checkpoint
  run_scoped: true         # Scope checkpoints by run name
  every_n_epochs: 0        # Save every N epochs (0 = disabled)
  every_n_samples: 0       # Save every N samples (0 = disabled)
```

## Model Configuration

Model configs are in `configs/model/`. Each model type has its own section:

```yaml
models:
  HeteroGNN:
    hidden_channels: 64
    num_layers: 4
    backend: "SAGE"       # Options: SAGE, GAT, GCN, GIN
    dropout: 0.0
```

## HPC Configs

Pre-built configs for HPC systems:

| System             | Config File                          |
| ------------------ | ------------------------------------ |
| Polaris (ALCF)     | `configs/config.polaris.ddp.yaml`    |
| Perlmutter (NERSC) | `configs/config.perlmutter.ddp.yaml` |

## Sweep Configs

Hyperparameter sweep configs for Optuna/W&B are in `configs/sweeps/`.
