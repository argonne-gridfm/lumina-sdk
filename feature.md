# LUMINA Feature Inventory

## Data

| # | Feature | Description |
|---|---------|-------------|
| D1 | **In-Memory Dataset** | `OPFDataset` — loads full dataset into memory from JSON/HDF5 → `.pt` files |
| D2 | **On-Disk Dataset** | `OPFOnDiskDataset` — SQLite/RocksDB backend for large cases, row-by-row access |
| D3 | **Sharded Iterable Dataset** | `OPFShardedIterableDataset` — manifest-based streaming for distributed training |
| D4 | **Multi-Case Dataset** | `OPFMultiDataset` — combines multiple cases/groups with factory methods |
| D5 | **Homogeneous Graph Conversion** | `OPFHomogeneousDataset` — converts hetero graphs to homo with node/edge type embeddings |
| D6 | **HDF5 Format Support** | Loads HDF5 data with automatic schema alignment between JSON/HDF5 formats |
| D7 | **Contingency (N-1/N-k) Support** | Topological perturbations: branch removal, generator removal, slack generator injection |
| D8 | **OPF Data Schema** | Pydantic-based schemas for bus, generator, load, shunt, ac_line, transformer with alignment maps |
| D9 | **Data Staging** | Copy datasets to fast local storage (SSD/tmpdir) for HPC with file locking |
| D10 | **Case ID Tagging** | `CaseTaggedDataset` attaches case identifiers to samples for multi-case training |
| D11 | **Target Sanitization** | Detects/replaces NaN/Inf targets with masking or zero-fill |

## Model

| # | Feature | Description |
|---|---------|-------------|
| M1 | **Heterogeneous GNN** | `OPFHeteroGNN` with configurable backends: SAGE, GCN, GIN, GAT |
| M2 | **HGT** | Heterogeneous Graph Transformer with type-aware attention |
| M3 | **HEAT** | Heterogeneous Edge-Attributed Transformer (v1 and v2) |
| M4 | **RGAT** | Relational Graph Attention Network |
| M5 | **Homogeneous GNNs** | GCN, GAT, GIN, Transformer for homogeneous graphs |
| M6 | **Model Factory** | `ModelFactory` for config-driven model instantiation |
| M7 | **Standard Losses** | MSE, RMSE, MAE, MAPE, SmoothL1 with per-node-type weighting |
| M12 | **Physics-Informed Loss** | `PhysicsInformedLoss` with quadratic/absolute/log-barrier penalty methods |

## Training

| # | Feature | Description |
|---|---------|-------------|
| T1 | **DDP Training** | Full `DistributedDataParallel` with synchronized metrics, barriers, and distributed samplers |
| T2 | **Multi-Case Training** | `MultiCaseOPFTrainer` — simultaneous training on multiple power flow cases with case mixing |
| T3 | **Sample-Based Scheduling** | Validation, checkpointing, and stopping triggers by sample count (not just epochs) |
| T4 | **Gradient Clipping** | Norm-based or value-based clipping with configurable thresholds |
| T5 | **Non-Finite Loss Handling** | Detects NaN/Inf in loss and gradients; skip or fail modes |
| T6 | **Checkpointing** | Periodic (epoch/sample), best-validation, with full state (model, optimizer, config) |
| T7 | **W&B Integration** | Weights & Biases logging for loss components, LR, constraint violations, throughput |
| T8 | **Cosine LR Scheduler** | CosineAnnealingLR with automatic T_max inference |
| T9 | **Gradient Accumulation** | Configurable accumulation steps for effective large batch sizes |
| T10 | **Early Stopping** | Patience-based stopping on validation loss |
| T11 | **Throughput Tracking** | `ThroughputTracker` — samples/sec measurement with warmup phase and metadata |
| T12 | **Validation Subset Sampling** | Seeded subset of validation set for faster evaluation |

## Evaluation

| # | Feature | Description |
|---|---------|-------------|
| E1 | **Bound Constraint Evaluator** | `ACOPFConstraintEvaluator` — voltage magnitude, generation (P/Q) limit violations |
| E2 | **Power Flow Constraints** | AC power balance (P, Q at each bus) via Y-bus matrix computation |
| E3 | **Line Flow Constraints** | Thermal limit violations on transmission lines (apparent power) |
| E4 | **Normalized Violation Metrics** | RMS-based metrics for P-balance, Q-balance, and line limits |
| E5 | **Modeler Utility** | Post-training evaluation: checkpoint loading, batch prediction, aggregate statistics |
| E7 | **Constraint EMA Tracking** | Exponential moving average of violations for convergence monitoring |
