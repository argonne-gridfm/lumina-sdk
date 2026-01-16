---
tags:
- power-systems
- optimal-power-flow
- graph-neural-network
- pytorch
- heterogeneous-graph
library_name: pytorch
pipeline_tag: other
---

# LUMINA-1B

## Model Description

**LUMINA-1B** is a 1-billion parameter heterogeneous graph neural network designed for **fast, high-fidelity prediction of AC Optimal Power Flow (ACOPF) solutions**.
The model encodes power system components—buses, generators, loads, and transmission lines—as distinct node and edge types, capturing structural and physical heterogeneity in electric grids.

Trained on ACOPF datasets spanning **multiple network topologies**, LUMINA-1B generalizes across diverse grid configurations and operating conditions. It provides efficient approximations of OPF variables such as voltages, power injections, and line flows, serving as a scalable surrogate for traditional optimization solvers.

## Model Details

- **Model Architecture**: HGT
- **Total Parameters**: 1,006,940,164 (1.0B)
- **Trainable Parameters**: 1,006,940,164 (1.0B)
- **Training Case**: OPFData-All
- **Training Data Size**: 15,000 samples per case
- **Training Date**: 20251112_193637
- **Final Validation Loss**: 0.082726
- **Training Epochs**: 20

## Input Channels
- **Bus**: 7 features
- **Generator**: 11 features
- **Load**: 2 features
- **Shunt**: 2 features
- **ac_line**: 6 features
- **transformer**: 11 features

## Model Files

This repository contains the model in multiple formats:
- `config.json` - Model configuration and metadata
- `model.pt` - Complete PyTorch checkpoint with metadata and config
- `model.safetensors` - Model weights in SafeTensors format (recommended)
- `requirements.txt` - Required Python libraries

## Lumina Installation

1. Clone or pull latest `lumina-core` repository:

   ```
   git clone git@github.com:argonne-gridfm/lumina-core.git
   ```

2. Install project as package from local directory in editable mode

   ```bash
   cd lumina-core
   pip install -e .
   ```

## Basic Usage

### Model Setup

```python
import json

import torch
from huggingface_hub import hf_hub_download

from lumina.dataset.opf.opf_dataset import OPFDataset
from lumina.loader.opf.opf_loader import DataLoader
from lumina.evaluator.opf.utils import Modeler

# Download model
config_path = hf_hub_download(repo_id="argonne/LUMINA-1B", filename="config.json")

# Load config
with open(config_path, "r") as f:
    config_data = json.load(f)

# Recreate model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modeler = Modeler(device, slack_bus_indices="0,1")
```

### Load Weights Option 1: Using SafeTensors (Recommended)

```python
...

# Load weights from SafeTensors
safetensors_path = hf_hub_download(repo_id="argonne/LUMINA-1B", filename="model.safetensors")
state_dict = load_file(safetensors_path)
modeler.load_model(config_data, state_dict)
...
```

### Load Weights Option 2: Using PyTorch Checkpoint

```python
...

# Load weights from PyTorch Checkpoint on CPU
model_path = hf_hub_download(repo_id="argonne/LUMINA-1B", filename="model.pt")
checkpoint = torch.load(model_path, map_location="cpu")
state_dict = checkpoint.get("model_state_dict")
modeler.load_model(config_data, state_dict)
...
```

### Load Data and Run Model

```python
...
# Simple argument defaults
case_name = config_data.get("case_name", "pglib_opf_case14_ieee")
batch_size = 1
max_batches = 50
total_batches = None

# Loading OPF dataset
dataset = OPFDataset(root="./opf_data", case_name=case_name)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Run model across data batches
preds = modeler.run_predictions(loader, max_batches=max_batches)

# Evaluate from predictions
stats = modeler.evaluate_from_predictions(preds, cache_key="pglib_opf_case14_ieee")
```


## Data Source

Google DeepMind. (2024). OPFData: Large-scale datasets for AC optimal power flow. Google Cloud Storage. Retrieved November 12, 2025, from storage.googleapis.com.


## Acknowledgements

This work was supported by the U.S. Department of Energy, Office of Science, Advanced Scientific Computing Research, and Laboratory Directed Research and Development (LDRD) funding from Argonne National Laboratory, under Contract DE-AC02-06CH11357. This research used resources of the Argonne Leadership Computing Facility at Argonne National Laboratory, which is supported by the Office of Science of the U.S. Department of Energy under contract DE-AC02-06CH11357. This research used resources of the Argonne Leadership Computing Facility and the National Energy Research Scientific Computing Center (NERSC), which are U.S. Department of Energy Office of Science User Facilities.


## License

Copyright (c) 2025, Argonne National Laboratory  
All rights reserved.
