import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

##LuminaCore
from lumina.dataset.scuc import SCUCDataset
from lumina.evaluator.scuc import SCUCConstraintViolations
from lumina.model.scuc import HGNNEncoder, HGTEncoder, SCUCTransformerHead, SCUCLSTMHead


# ======================
# Configuration
# ======================
SCUC_HEAD_TYPE = "lstm"  # Options: "transformer" or "lstm"

# ======================
# Training Log Setup
# ======================
LOG_DIR = "training_logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Generate unique run identifier
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_id = f"{timestamp}_{os.urandom(4).hex()}"
log_filename = f"training_log_{run_id}.txt"
log_filepath = os.path.join(LOG_DIR, log_filename)

# Open log file
log_file = open(log_filepath, 'w', buffering=1)  # Line buffered

def log_print(message):
    """Print to both console and log file."""
    print(message)
    log_file.write(message + '\n')
    log_file.flush()

log_print("="*80)
log_print("SCUC TRAINING LOG")
log_print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log_print(f"Run ID: {run_id}")
log_print(f"Log file: {log_filepath}")
log_print("="*80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_print(f"\n🔧 Device: {device}")

# Encoder types: hgnn, hgt
ENCODER = "hgt"

log_print(f"\n{'='*80}")
log_print("CONFIGURATION")
log_print(f"{'='*80}")
log_print(f"SCUC Head Type: {SCUC_HEAD_TYPE}")
log_print(f"Encoder Type: {ENCODER}")

# ======================
# Load SCUC datasets
# ======================
scuc_cases = [
    "case14",
    "case30",
    "case57",
    "case118",
    "case300"
    # Add more cases as needed
]

SCUC_CASE_TO_ID = {case: idx for idx, case in enumerate(scuc_cases)}
SCUC_CASE_ID_TO_NAME = {idx: case for case, idx in SCUC_CASE_TO_ID.items()}
log_print(f"SCUC Cases: {scuc_cases}")

root_scuc = "/eagle/projects/GridFM/datasets/scuc"

# Load SCUC datasets
scuc_datasets = [
    SCUCDataset(
        root=root_scuc,
        case_name=case,
        problem_root=root_scuc,
        sol_root=root_scuc,
        force_reload=False
    )
    for case in scuc_cases
]

# Attach numeric identifiers so batches know which case they came from.
for case, dataset in zip(scuc_cases, scuc_datasets):
    dataset.case_id = SCUC_CASE_TO_ID[case]

log_print(f"\n✅ Loaded {len(scuc_datasets)} SCUC dataset(s)")
log_print(f"   Sample data: {scuc_datasets[0][0]}")

# Concatenate datasets
scuc_dataset = ConcatDataset(scuc_datasets)
log_print(f"   Total samples: {len(scuc_dataset)}")

# DataLoader
batch_size = 64
scuc_loader = DataLoader(scuc_dataset, batch_size=batch_size, shuffle=True, 
                        num_workers=4, pin_memory=True, persistent_workers=True)

# Calculate actual batches per epoch
actual_batches_per_epoch = len(scuc_loader)
log_print(f"   Batches per epoch: {actual_batches_per_epoch}")
log_print(f"   Samples per batch: {batch_size}")
log_print(f"   Total batches in dataset: {actual_batches_per_epoch}")

# Set max steps per epoch (use full dataset or cap at 300)
max_steps_per_epoch = min(actual_batches_per_epoch, 300)
if actual_batches_per_epoch < 300:
    log_print(f"\n⚠️  WARNING: Dataset only has {actual_batches_per_epoch} batches")
    log_print(f"   Will use all {actual_batches_per_epoch} batches per epoch")
    log_print(f"   Consider: Adding more data OR reducing batch_size to increase batches")
else:
    log_print(f"\n✅ Using {max_steps_per_epoch} batches per epoch (dataset has {actual_batches_per_epoch})")

# ======================
# Metadata
# ======================
sample_scuc = scuc_dataset[0]
log_print(f"\nSample SCUC data: {sample_scuc}")

# Get metadata
all_node_types = sorted(set(sample_scuc.node_types))
all_edge_types = sorted(set(sample_scuc.edge_types))

metadata = (all_node_types, all_edge_types)
log_print(f"Metadata: {metadata}")

def get_input_channels(data):
    return {ntype: data[ntype].x.size(-1) for ntype in data.node_types}

scuc_input = get_input_channels(sample_scuc)
input_channels = scuc_input
log_print(f"Input channels: {input_channels}")

# ======================
# Models
# ======================
if ENCODER == "hgnn":
    encoder = HGNNEncoder(
        metadata=metadata,
        input_channels=input_channels,
        hidden_channels=64,
        num_layers=4,
        backend="sage"
    ).to(device)
elif ENCODER == "hgt":
    encoder = HGTEncoder(
        metadata=metadata,
        input_channels=input_channels,
        hidden_channels=64,
        num_layers=8,
        heads=8
    ).to(device)

# SCUC Head (choose between Transformer and LSTM)
log_print(f"\n🔧 Initializing SCUC Head: {SCUC_HEAD_TYPE.upper()}")

if SCUC_HEAD_TYPE == "transformer":
    scuc_head = SCUCTransformerHead(
        d_emb=64,
        d_time=16,
        n_time=36,
        hidden_dim=128,
        n_layers=2,
        n_heads=4
    ).to(device)
    log_print("   ✅ Using Transformer-based head with self-attention")
    log_print(f"   Parameters: d_emb=64, d_time=16, n_time=36, hidden_dim=128, n_layers=2, n_heads=4")
elif SCUC_HEAD_TYPE == "lstm":
    scuc_head = SCUCLSTMHead(
        d_emb=64,
        d_time=16,
        n_time=36,
        hidden_dim=128,
        n_layers=2,
        dropout=0.1,
        bidirectional=True
    ).to(device)
    log_print("   ✅ Using LSTM-based head (bidirectional)")
    log_print(f"   Parameters: d_emb=64, d_time=16, n_time=36, hidden_dim=128, n_layers=2, bidirectional=True")
else:
    raise ValueError(f"Unknown SCUC_HEAD_TYPE: {SCUC_HEAD_TYPE}. Use 'transformer' or 'lstm'")

# ======================
# Model Size Analysis
# ======================
def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size_mb(model):
    """Calculate the model size in MB (assuming float32, 4 bytes per parameter)."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size
    return total_size / (1024 * 1024)  # Convert bytes to MB

# Calculate model sizes
encoder_params = count_parameters(encoder)
scuc_head_params = count_parameters(scuc_head)
total_params = encoder_params + scuc_head_params

encoder_size_mb = get_model_size_mb(encoder)
scuc_head_size_mb = get_model_size_mb(scuc_head)
total_size_mb = encoder_size_mb + scuc_head_size_mb

# Print model size information
log_print("\n" + "="*80)
log_print("📊 MODEL SIZE ANALYSIS")
log_print("="*80)
log_print(f"🔧 Encoder ({ENCODER.upper()}):")
log_print(f"   Parameters: {encoder_params:,}")
log_print(f"   Size: {encoder_size_mb:.2f} MB")
log_print(f"\n🔧 SCUC Head ({SCUC_HEAD_TYPE.upper()}):")
log_print(f"   Parameters: {scuc_head_params:,}")
log_print(f"   Size: {scuc_head_size_mb:.2f} MB")
log_print(f"\n📦 Total Model:")
log_print(f"   Total Parameters: {total_params:,}")
log_print(f"   Total Size: {total_size_mb:.2f} MB ({total_size_mb/1024:.3f} GB)")
log_print("="*80)

# Optimizer
optimizer_scuc = optim.Adam(
    list(encoder.parameters()) + list(scuc_head.parameters()),
    lr=5e-4, weight_decay=1e-5
)

# Constraint violation tracker
constraint_violations = SCUCConstraintViolations(hard_binary=False)
log_print(f"\n🔧 Constraint Violation Tracker initialized (soft binary mode)")

# ======================
# Compute SCUC Pg Normalization Stats
# ======================
@torch.no_grad()
def compute_scuc_pg_stats(dataset):
    """Compute mean and std of Pg targets for normalization."""
    vals = []
    for i in range(len(dataset)):
        d = dataset[i]
        # generator.y: [num_gens, n_time, 2] where [:,:,1] is Pg target
        pg = d['generator'].y[:, :, 1].reshape(-1).float()
        vals.append(pg)
    allv = torch.cat(vals)
    mu = allv.mean().item()
    std = allv.std(unbiased=False).item()
    return mu, max(std, 1e-8)

log_print("\n🔄 Computing SCUC normalization statistics...")
scuc_pg_mean, scuc_pg_std = compute_scuc_pg_stats(scuc_dataset)
log_print(f"✅ SCUC Pg stats → mean: {scuc_pg_mean:.6f}, std: {scuc_pg_std:.6f}")

# Convert to tensors for fast arithmetic
pg_mean_t = torch.tensor(scuc_pg_mean, device=device)
pg_std_t = torch.tensor(scuc_pg_std, device=device)
SCUC_EPS = 1e-8

# Loss functions
pos_weight_val = 3987 / 1773  # Adjust based on your data (on/off ratio)
pos_weight_tensor = torch.tensor(pos_weight_val, device=device)
bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
mse_loss = nn.MSELoss()
smooth_l1 = nn.SmoothL1Loss(beta=0.5)

# ======================
# Utility Functions
# ======================
def prepare_x_dict(data, input_channels):
    """Prepare input dictionary with padding if needed."""
    out = {}
    for ntype, dim in input_channels.items():
        if ntype in data.node_types:
            x = data[ntype].x
            if x.size(-1) < dim:
                pad = torch.zeros(x.size(0), dim - x.size(-1), device=x.device)
                x = torch.cat([x, pad], dim=-1)
            out[ntype] = x.to(device)
        else:
            out[ntype] = torch.zeros((0, dim), device=device)
    return out


def init_case_metric_store():
    """Create a fresh accumulator for per-case SCUC metrics."""
    return {
        case: {
            'pg_loss_sum': 0.0,
            'unit_loss_sum': 0.0,
            'total_loss_sum': 0.0,
            'accuracy_sum': 0.0,
            'violation_fraction_sum': 0.0,
            'violations_per_generator_sum': 0.0,
            'violations_per_time_sum': 0.0,
            'mw_violations_per_generator_sum': 0.0,
            'avg_mw_violation_sum': 0.0,
            'samples': 0,
        }
        for case in scuc_cases
    }


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0

# ======================
# Model Saving Setup
# ======================
model_save_dir = "/eagle/projects/GridFM/trained_models"
os.makedirs(model_save_dir, exist_ok=True)
log_print(f"\n✅ Model save directory: {model_save_dir}")

best_total_loss = float('inf')
wandb_run_name = f"scuc_{SCUC_HEAD_TYPE}_{int(time.time())}"
best_model_path = os.path.join(model_save_dir, f"best_scuc_{SCUC_HEAD_TYPE}_model_{wandb_run_name}.pth")
log_print(f"✅ Best model will be saved as: {best_model_path}")
log_print(f"✅ W&B Run name: {wandb_run_name}")

# ======================
# WANDB Setup
# ======================
def _try_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return None

wandb_config = {
    "project": "grid-foundation",
    "entity": None,
    "run_name": wandb_run_name,
    "datasets": {
        "SCUC_cases": scuc_cases,
        "len_scuc": len(scuc_dataset),
        "batch_size": batch_size,
        "shuffle": True,
        "total_batches_available": actual_batches_per_epoch,
        "max_steps_per_epoch": max_steps_per_epoch  # Actual steps used
    },
    "model": {
        "encoder": {
            "arch": ENCODER,
            "hidden_channels": 64,
            "num_layers": 8 if ENCODER == "hgt" else 4,
            "heads": 8 if ENCODER == "hgt" else None,
            "backend": "sage" if ENCODER == "hgnn" else None,
            "input_channels_per_ntype": input_channels,
            "metadata_node_types": list(metadata[0]),
            "metadata_edge_types": [str(e) for e in metadata[1]],
        },
        "scuc_head": {
            "type": SCUC_HEAD_TYPE,
            "d_emb": 64,
            "d_time": 16,
            "n_time": 36,
            "hidden_dim": 128,
            "n_layers": 2,
            "n_heads": 4 if SCUC_HEAD_TYPE == "transformer" else None,
            "bidirectional": True if SCUC_HEAD_TYPE == "lstm" else None
        }
    },
    "optim": {
        "optimizer": "Adam",
        "lr": 5e-4,
        "num_epochs": 150,
        "losses": {
            "scuc_pg": "MSELoss (normalized Pg)",
            "scuc_unit": "BCEWithLogitsLoss (binary commitment, pos_weight adjusted)"
        },
        "normalization": {
            "scuc_pg_mean": scuc_pg_mean,
            "scuc_pg_std": scuc_pg_std
        }
    },
    "env": {
        "device": str(device),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "platform": platform.platform(),
        "git_commit": _try_git_commit(),
    },
}

# Log full configuration to file
log_print(f"\n{'='*80}")
log_print("FULL CONFIGURATION DETAILS")
log_print(f"{'='*80}")
log_print(json.dumps(wandb_config, indent=2))
log_print(f"{'='*80}")

# Start W&B run
log_print(f"\n🌐 Initializing Weights & Biases...")
wandb.init(
    project=wandb_config["project"],
    name=wandb_config["run_name"],
    config=wandb_config,
)
log_print(f"✅ W&B initialized: Project='{wandb_config['project']}', Run='{wandb_config['run_name']}'")

# ======================
# Training Loop
# ======================
num_epochs = 150

# Track losses and accuracy
pg_loss_curve = []
unit_loss_curve = []
total_loss_curve = []
accuracy_curve = []

encoder.train()
scuc_head.train()

log_print("\n" + "="*80)
log_print("🚀 Starting SCUC Training")
log_print("="*80)
log_print(f"Epochs: {num_epochs}")
log_print(f"Batch Size: {batch_size}")

# Use actual dataset size or cap at 300 (whichever is smaller)
max_steps_per_epoch = min(actual_batches_per_epoch, 300)
log_print(f"Steps per Epoch: {max_steps_per_epoch} (dataset has {actual_batches_per_epoch} batches)")

if actual_batches_per_epoch < 300:
    log_print(f"⚠️  WARNING: Dataset only has {actual_batches_per_epoch} batches (< 300)")
    log_print(f"   Consider adding more data or reducing batch size")

log_print("="*80)

# Track total training time
training_start_time = time.time()

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    epoch_pg_loss = 0.0
    epoch_unit_loss = 0.0
    total_loss_epoch = 0.0
    epoch_accuracy = 0.0
    batches = 0
    commitment_acc = 0.0
    
    # Constraint violation tracking - need to track batch-level metrics properly
    epoch_violation_fraction = 0.0  # Average violation rate across batches
    epoch_violations_per_gen = 0.0  # Average violations per generator across batches  
    epoch_violations_per_time = 0.0  # Average violations per time step across batches
    epoch_mw_violations_per_gen = 0.0  # Average MW violations per generator across batches
    epoch_avg_mw_violation = 0.0  # Average MW per violation across batches
    epoch_total_batches = 0  # Count batches for proper averaging

    # Per-case metric accumulation
    case_metrics = init_case_metric_store()
    
    # Log epoch start (every 10 epochs)
    if epoch % 10 == 0:
        log_print(f"\n📅 Starting Epoch {epoch+1}/{num_epochs} (processing {max_steps_per_epoch} batches)")
    
    # Progress bar
    pbar = tqdm(enumerate(scuc_loader), total=max_steps_per_epoch,
                desc=f"Epoch {epoch+1}/{num_epochs}",
                leave=False, ncols=100)
    
    for step, scuc_batch in pbar:
        if step >= max_steps_per_epoch:
            break
        
        scuc_batch = scuc_batch.to(device)
        
        # Clear gradients
        optimizer_scuc.zero_grad()
        
        # === SCUC Forward Pass (matching main_new.py) ===
        x_dict_scuc = prepare_x_dict(scuc_batch, input_channels)
        edge_index_dict_scuc = {e: scuc_batch[e].edge_index for e in scuc_batch.edge_types}
        edge_attr_dict_scuc = {e: scuc_batch[e].edge_attr for e in scuc_batch.edge_types 
                              if 'edge_attr' in scuc_batch[e]}
        
        # Encode graph
        h_scuc = encoder(x_dict_scuc, edge_index_dict_scuc, edge_attr_dict_scuc)
        h_bus = h_scuc["bus"]
        h_gen = h_scuc["generator"]
        
        # Concatenate bus and generator embeddings
        h_all = torch.cat([h_bus, h_gen], dim=0)
        
        # Prepare temporal load profiles
        lp_bus = scuc_batch['bus'].temp_load.to(device)  # [num_bus, n_time]
        lp_gen = torch.zeros((h_gen.size(0), lp_bus.size(1)), device=device)
        lp_all = torch.cat([lp_bus, lp_gen], dim=0)
        
        # Create generator mask
        gen_mask = torch.cat([
            torch.zeros(h_bus.size(0), dtype=torch.bool, device=device),
            torch.ones(h_gen.size(0), dtype=torch.bool, device=device)
        ], dim=0)
        
        # Get SCUC labels (temporal)
        scuc_labels = scuc_batch['generator'].y.to(device)  # [num_gens, n_time, 2]
        pg_target = scuc_labels[:, :, 1]      # [num_gens, n_time] - Pg in MW
        unit_target = scuc_labels[:, :, 0]    # [num_gens, n_time] - 0/1 commitment
        
        # Debug on first iteration
        if step == 0 and epoch == 0:
            log_print(f"\n🔍 Debug Info:")
            log_print(f"   h_all shape: {h_all.shape}")
            log_print(f"   lp_all shape: {lp_all.shape}")
            log_print(f"   gen_mask shape: {gen_mask.shape}")
            log_print(f"   scuc_labels shape: {scuc_labels.shape}")
            log_print(f"   pg_target shape: {pg_target.shape}")
            log_print(f"   unit_target shape: {unit_target.shape}")
        
        # Forward through SCUC head
        Pg, UnitLabel = scuc_head(h_all, lp_all, gen_mask)  # Pg in MW, UnitLabel as logits
        
        Pg_norm = (Pg - pg_mean_t) / (pg_std_t + SCUC_EPS)
        pg_target_norm = (pg_target - pg_mean_t) / (pg_std_t + SCUC_EPS)

        # === Constraint Violation Tracking ===
        with torch.no_grad():
            # Extract generator features for constraint parameters
            gen_features = scuc_batch['generator'].x.to(device)  # [num_gens, feature_dim]
            gen_params = constraint_violations.extract_generator_params(gen_features)
            
            # Compute violations with production curve limits now included
            # pmin/pmax are extracted from "Production cost curve (MW)" data at indices [19] and [20]
            violations = constraint_violations.compute_violations(
                Pg=Pg,
                ulogits=UnitLabel,
                pmin=gen_params['pmin'],  # From production curve [0] (minimum power)
                pmax=gen_params['pmax'],  # From production curve [-1] (maximum power)
                rup=gen_params['rup'],
                rdn=gen_params['rdn'],
                sup=gen_params['sup'],
                sdn=gen_params['sdn'],
                init_status_hours=gen_params['init_status_hours'],
                init_power=gen_params['init_power']
            )
            
            # Debug: Log first batch violations summary  
            if step == 0 and epoch == 0:
                log_print(f"\n🔍 Constraint Violation Debug (first batch):")
                log_print(f"   {violations['summary_str']}")
                log_print(f"   Capacity and ramping violations computed with actual constraints")
                log_print(f"   pmin/pmax from production curve data, ramp limits from generator specs")

            # ---- Per-case metric accumulation ----
            generator_batch_idx = scuc_batch['generator'].batch
            case_ids = scuc_batch.case_id.view(-1)
            time_steps = Pg.size(1)

            for graph_idx, case_idx_tensor in enumerate(case_ids):
                case_idx_val = int(case_idx_tensor.item())
                if case_idx_val < 0:
                    continue
                case_name = SCUC_CASE_ID_TO_NAME.get(case_idx_val)
                if case_name is None or case_name not in case_metrics:
                    continue

                gen_mask = (generator_batch_idx == graph_idx)
                if gen_mask.sum().item() == 0:
                    continue

                Pg_norm_sample = Pg_norm[gen_mask]
                pg_target_norm_sample = pg_target_norm[gen_mask]
                UnitLabel_sample = UnitLabel[gen_mask]
                unit_target_sample = unit_target[gen_mask]

                loss_pg_sample = F.mse_loss(Pg_norm_sample, pg_target_norm_sample, reduction='mean').item()
                loss_unit_sample = F.binary_cross_entropy_with_logits(
                    UnitLabel_sample, unit_target_sample,
                    pos_weight=pos_weight_tensor, reduction='mean'
                ).item()
                total_loss_sample = loss_pg_sample + loss_unit_sample

                preds_sample = (torch.sigmoid(UnitLabel_sample) >= 0.5).float()
                acc_sample = (preds_sample == unit_target_sample).float().mean().item()

                viol_cap_lower_sample = violations['viol_cap_lower'][gen_mask]
                viol_cap_upper_sample = violations['viol_cap_upper'][gen_mask]
                viol_ramp_up_sample = violations['viol_ramp_up'][gen_mask]
                viol_ramp_dn_sample = violations['viol_ramp_dn'][gen_mask]

                cap_lower_mask_sample = violations['cap_lower_mask'][gen_mask]
                cap_upper_mask_sample = violations['cap_upper_mask'][gen_mask]
                ramp_up_mask_sample = violations['ramp_up_mask'][gen_mask]
                ramp_dn_mask_sample = violations['ramp_dn_mask'][gen_mask]

                total_mw_viol_cap_lower = viol_cap_lower_sample.sum().item()
                total_mw_viol_cap_upper = viol_cap_upper_sample.sum().item()
                total_mw_viol_ramp_up = viol_ramp_up_sample.sum().item()
                total_mw_viol_ramp_dn = viol_ramp_dn_sample.sum().item()
                total_mw_viol = (
                    total_mw_viol_cap_lower + total_mw_viol_cap_upper +
                    total_mw_viol_ramp_up + total_mw_viol_ramp_dn
                )

                num_cap_lower = cap_lower_mask_sample.sum().item()
                num_cap_upper = cap_upper_mask_sample.sum().item()
                num_ramp_up = ramp_up_mask_sample.sum().item()
                num_ramp_dn = ramp_dn_mask_sample.sum().item()
                total_violations = num_cap_lower + num_cap_upper + num_ramp_up + num_ramp_dn

                num_generators_sample = gen_mask.sum().item()
                total_possible = num_generators_sample * time_steps * 4

                violation_fraction = safe_div(total_violations, total_possible)
                violations_per_generator = safe_div(total_violations, num_generators_sample)
                violations_per_time = safe_div(total_violations, time_steps)
                mw_violations_per_generator = safe_div(total_mw_viol, num_generators_sample)
                avg_mw_violation = safe_div(total_mw_viol, total_violations)

                metrics_entry = case_metrics[case_name]
                metrics_entry['pg_loss_sum'] += loss_pg_sample
                metrics_entry['unit_loss_sum'] += loss_unit_sample
                metrics_entry['total_loss_sum'] += total_loss_sample
                metrics_entry['accuracy_sum'] += acc_sample
                metrics_entry['violation_fraction_sum'] += violation_fraction
                metrics_entry['violations_per_generator_sum'] += violations_per_generator
                metrics_entry['violations_per_time_sum'] += violations_per_time
                metrics_entry['mw_violations_per_generator_sum'] += mw_violations_per_generator
                metrics_entry['avg_mw_violation_sum'] += avg_mw_violation
                metrics_entry['samples'] += 1
        # === Pg Loss (Normalized MSE) ===
        loss_pg = mse_loss(Pg_norm, pg_target_norm)
        
        # === Unit Commitment Loss (BCE with logits) ===
        loss_unit = bce_loss(UnitLabel, unit_target)
        
        # === Total SCUC Loss ===
        scuc_loss_total = loss_pg + loss_unit
        
        # Compute commitment accuracy
        with torch.no_grad():
            probs = torch.sigmoid(UnitLabel)
            preds = (probs >= 0.5).float()
            commitment_acc = (preds == unit_target).float().mean().item()
        
        # Backward pass
        scuc_loss_total.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(scuc_head.parameters(), max_norm=1.0)
        
        optimizer_scuc.step()
        
        # Accumulate losses and accuracy
        epoch_pg_loss += loss_pg.item()
        epoch_unit_loss += loss_unit.item()
        total_loss_epoch += scuc_loss_total.item()
        epoch_accuracy += commitment_acc
        
        # Accumulate constraint violations (average across batches)
        epoch_violation_fraction += violations['violation_fraction']
        epoch_violations_per_gen += violations['violations_per_generator']
        epoch_violations_per_time += violations['violations_per_time']
        epoch_mw_violations_per_gen += violations['mw_violations_per_generator']
        epoch_avg_mw_violation += violations['avg_mw_violation']
        epoch_total_batches += 1
        
        batches += 1
        
        # Update progress bar
        current_pg_loss = epoch_pg_loss / batches
        current_unit_loss = epoch_unit_loss / batches
        current_total_loss = total_loss_epoch / batches
        
        progress_info = {
            'Pg': f'{current_pg_loss:.4f}',
            'Unit': f'{current_unit_loss:.4f}',
            'Total': f'{current_total_loss:.4f}',
            'Acc': f'{commitment_acc:.3f}'
        }
        pbar.set_postfix(progress_info)
    
    pbar.close()
    
    # Calculate epoch timing
    epoch_time = time.time() - epoch_start_time
    
    # Average losses and accuracy
    epoch_pg_loss /= batches
    epoch_unit_loss /= batches
    total_loss_epoch /= batches
    epoch_accuracy /= batches
    
    # Average constraint violations across batches
    epoch_violation_fraction /= epoch_total_batches if epoch_total_batches > 0 else 1
    epoch_violations_per_gen /= epoch_total_batches if epoch_total_batches > 0 else 1
    epoch_violations_per_time /= epoch_total_batches if epoch_total_batches > 0 else 1
    epoch_mw_violations_per_gen /= epoch_total_batches if epoch_total_batches > 0 else 1
    epoch_avg_mw_violation /= epoch_total_batches if epoch_total_batches > 0 else 1
    
    # Log to wandb with interpretable constraint metrics
    wandb_payload = {
        "epoch": epoch + 1,
        "loss/scuc_pg": epoch_pg_loss,
        "loss/scuc_unit": epoch_unit_loss,
        "loss/total": total_loss_epoch,
        "metrics/commitment_accuracy": epoch_accuracy,
        "constraints/violation_rate": epoch_violation_fraction,
        "constraints/violations_per_generator": epoch_violations_per_gen,
        "constraints/violations_per_time": epoch_violations_per_time,
        "constraints/mw_violations_per_generator": epoch_mw_violations_per_gen,
        "constraints/avg_mw_per_violation": epoch_avg_mw_violation,
        "timing/epoch_time": epoch_time,
        "timing/batches_processed": batches
    }

    per_case_console_lines = []
    for case_name, metrics_entry in case_metrics.items():
        samples = metrics_entry['samples']
        if samples == 0:
            continue

        avg_pg_loss_case = metrics_entry['pg_loss_sum'] / samples
        avg_unit_loss_case = metrics_entry['unit_loss_sum'] / samples
        avg_total_loss_case = metrics_entry['total_loss_sum'] / samples
        avg_accuracy_case = metrics_entry['accuracy_sum'] / samples
        avg_violation_fraction_case = metrics_entry['violation_fraction_sum'] / samples
        avg_violations_per_gen_case = metrics_entry['violations_per_generator_sum'] / samples
        avg_violations_per_time_case = metrics_entry['violations_per_time_sum'] / samples
        avg_mw_viol_per_gen_case = metrics_entry['mw_violations_per_generator_sum'] / samples
        avg_mw_per_violation_case = metrics_entry['avg_mw_violation_sum'] / samples

        wandb_payload.update({
            f"scuc_case/{case_name}/loss_pg": avg_pg_loss_case,
            f"scuc_case/{case_name}/loss_unit": avg_unit_loss_case,
            f"scuc_case/{case_name}/loss_total": avg_total_loss_case,
            f"scuc_case/{case_name}/accuracy": avg_accuracy_case,
            f"scuc_case/{case_name}/violation_rate": avg_violation_fraction_case,
            f"scuc_case/{case_name}/violations_per_generator": avg_violations_per_gen_case,
            f"scuc_case/{case_name}/violations_per_time": avg_violations_per_time_case,
            f"scuc_case/{case_name}/mw_violations_per_generator": avg_mw_viol_per_gen_case,
            f"scuc_case/{case_name}/avg_mw_per_violation": avg_mw_per_violation_case,
            f"scuc_case/{case_name}/samples": samples,
        })

        per_case_console_lines.append(
            f"   [{case_name}] Loss: {avg_total_loss_case:.4f} (Pg: {avg_pg_loss_case:.4f}, "
            f"Unit: {avg_unit_loss_case:.4f}) | Acc: {avg_accuracy_case:.3f} | "
            f"Viol: {avg_violation_fraction_case:.1%}, {avg_violations_per_gen_case:.1f}/gen, "
            f"{avg_mw_per_violation_case:.1f} MW/viol"
        )

    wandb.log(wandb_payload)
    
    # Save to curves
    pg_loss_curve.append(epoch_pg_loss)
    unit_loss_curve.append(epoch_unit_loss)
    total_loss_curve.append(total_loss_epoch)
    accuracy_curve.append(epoch_accuracy)
    
    # Detailed logging with interpretable constraint violations
    log_msg = (f"[Epoch {epoch+1}/{num_epochs}] "
               f"Time: {epoch_time:.1f}s, "
               f"Loss: {total_loss_epoch:.4f} "
               f"(Pg: {epoch_pg_loss:.4f}, Unit: {epoch_unit_loss:.4f}), "
               f"Acc: {epoch_accuracy:.3f}, "
               f"Constr: {epoch_violation_fraction:.1%} rate, "
               f"{epoch_violations_per_gen:.1f} viol/gen, "
               f"{epoch_avg_mw_violation:.0f} MW/viol")
    log_print(log_msg)
    if per_case_console_lines:
        log_print("   Per-case breakdown:")
        for line in per_case_console_lines:
            log_print(line)
    
    # ======================
    # Model Saving
    # ======================
    if total_loss_epoch < best_total_loss:
        best_total_loss = total_loss_epoch
        
        torch.save({
            'epoch': epoch + 1,
            'encoder_state_dict': encoder.state_dict(),
            'scuc_head_state_dict': scuc_head.state_dict(),
            'optimizer_state_dict': optimizer_scuc.state_dict(),
            'total_loss': total_loss_epoch,
            'pg_loss': epoch_pg_loss,
            'unit_loss': epoch_unit_loss,
            'commitment_accuracy': epoch_accuracy,
            'scuc_pg_mean': scuc_pg_mean,
            'scuc_pg_std': scuc_pg_std,
            'loss_curves': {
                'pg_loss': pg_loss_curve,
                'unit_loss': unit_loss_curve,
                'total_loss': total_loss_curve,
                'accuracy': accuracy_curve
            },
            'model_config': {
                'encoder_type': ENCODER,
                'scuc_head_type': SCUC_HEAD_TYPE,
                'hidden_channels': 64,
                'num_layers': 8 if ENCODER == "hgt" else 4,
                'heads': 8 if ENCODER == "hgt" else None,
                'input_channels': input_channels,
                'metadata': metadata,
                'n_time': 36,
                'd_time': 16,
            }
        }, best_model_path)
        log_print(f"   ✅ Saved NEW best model (loss: {total_loss_epoch:.4f}, acc: {epoch_accuracy:.3f})")
    
    # Save checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        checkpoint_path = os.path.join(model_save_dir, 
                                      f"scuc_{SCUC_HEAD_TYPE}_checkpoint_epoch_{epoch+1}_{wandb_run_name}.pth")
        torch.save({
            'epoch': epoch + 1,
            'encoder_state_dict': encoder.state_dict(),
            'scuc_head_state_dict': scuc_head.state_dict(),
            'optimizer_state_dict': optimizer_scuc.state_dict(),
            'total_loss': total_loss_epoch,
            'pg_loss': epoch_pg_loss,
            'unit_loss': epoch_unit_loss,
            'commitment_accuracy': epoch_accuracy,
            'scuc_pg_mean': scuc_pg_mean,
            'scuc_pg_std': scuc_pg_std,
            'loss_curves': {
                'pg_loss': pg_loss_curve,
                'unit_loss': unit_loss_curve,
                'total_loss': total_loss_curve,
                'accuracy': accuracy_curve
            },
            'model_config': {
                'encoder_type': ENCODER,
                'scuc_head_type': SCUC_HEAD_TYPE,
                'hidden_channels': 64,
                'num_layers': 8 if ENCODER == "hgt" else 4,
                'heads': 8 if ENCODER == "hgt" else None,
                'input_channels': input_channels,
                'metadata': metadata,
                'n_time': 36,
                'd_time': 16,
            }
        }, checkpoint_path)
        log_print(f"   💾 Saved checkpoint (epoch {epoch+1})")

# ======================
# Final Model Save
# ======================
final_model_path = os.path.join(model_save_dir, f"final_scuc_{SCUC_HEAD_TYPE}_model_{wandb_run_name}.pth")
torch.save({
    'epoch': num_epochs,
    'encoder_state_dict': encoder.state_dict(),
    'scuc_head_state_dict': scuc_head.state_dict(),
    'optimizer_state_dict': optimizer_scuc.state_dict(),
    'total_loss': total_loss_epoch,
    'pg_loss': epoch_pg_loss,
    'unit_loss': epoch_unit_loss,
    'commitment_accuracy': epoch_accuracy,
    'scuc_pg_mean': scuc_pg_mean,
    'scuc_pg_std': scuc_pg_std,
    'loss_curves': {
        'pg_loss': pg_loss_curve,
        'unit_loss': unit_loss_curve,
        'total_loss': total_loss_curve,
        'accuracy': accuracy_curve
    },
    'model_config': {
        'encoder_type': ENCODER,
        'scuc_head_type': SCUC_HEAD_TYPE,
        'hidden_channels': 64,
        'num_layers': 8 if ENCODER == "hgt" else 4,
        'heads': 8 if ENCODER == "hgt" else None,
        'input_channels': input_channels,
        'metadata': metadata,
        'n_time': 36,
        'd_time': 16,
    }
}, final_model_path)
log_print(f"\n✅ Saved final model to {final_model_path}")

# ======================
# Plotting
# ======================
log_print(f"\n{'='*80}")
log_print("GENERATING TRAINING PLOTS")
log_print(f"{'='*80}")
os.makedirs("plots_scuc", exist_ok=True)

EPS = 1e-12

def plot_and_save(values, title, filename, ylabel="Loss", logy=False):
    vals = [max(v, EPS) if logy else v for v in values]
    plt.figure()
    plt.plot(range(1, len(vals)+1), vals, marker='o')
    if logy:
        plt.yscale("log")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel + (" (log)" if logy else ""))
    plt.grid(True, which='both' if logy else 'major')
    plt.tight_layout()
    plt.savefig(os.path.join("plots_scuc", filename))
    plt.close()

# Individual plots
plot_and_save(total_loss_curve, "SCUC Total Loss", "scuc_total_loss.png")
plot_and_save(pg_loss_curve, "SCUC Pg Loss (log)", 
             "scuc_pg_loss_log.png", logy=True)
plot_and_save(unit_loss_curve, "SCUC Unit Commitment Loss (log)", 
             "scuc_unit_loss_log.png", logy=True)

# Accuracy plot (linear, not log)
plot_and_save(accuracy_curve, "SCUC Commitment Accuracy", 
             "scuc_accuracy.png", ylabel="Accuracy", logy=False)

# Loss + Accuracy combined plot (dual y-axis)
fig, ax1 = plt.subplots(figsize=(10, 6))
epochs_list = list(range(1, len(total_loss_curve)+1))

color_loss = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Total Loss', color=color_loss)
ax1.plot(epochs_list, total_loss_curve, marker='o', color=color_loss, 
         label='Total Loss', linewidth=2)
ax1.tick_params(axis='y', labelcolor=color_loss)
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
color_acc = 'tab:blue'
ax2.set_ylabel('Commitment Accuracy', color=color_acc)
ax2.plot(epochs_list, accuracy_curve, marker='s', color=color_acc, 
         label='Accuracy', linewidth=2)
ax2.tick_params(axis='y', labelcolor=color_acc)
ax2.set_ylim([0, 1])  # Accuracy is 0-1

plt.title('SCUC Training: Loss and Accuracy over Epochs')
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=2)
plt.tight_layout()
plt.savefig(os.path.join("plots_scuc", "scuc_loss_and_accuracy.png"), dpi=300)
plt.close()
print("   ✅ Saved loss + accuracy combined plot")

# Combined plot
epochs = range(1, num_epochs+1)

def normalize_curve(curve):
    m = max(curve) if max(curve) != 0 else 1.0
    return [v / m for v in curve]

norm_pg = normalize_curve(pg_loss_curve)
norm_unit = normalize_curve(unit_loss_curve)
norm_total = normalize_curve(total_loss_curve)

plt.figure(figsize=(10,6))
plt.plot(epochs, norm_pg, marker='o', label='Pg Loss (MSE)')
plt.plot(epochs, norm_unit, marker='s', label='Unit Commitment Loss (BCE)')
plt.plot(epochs, norm_total, marker='^', label='Total Loss')
plt.title("SCUC Training Losses (Normalized)")
plt.xlabel("Epoch")
plt.ylabel("Normalized Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join("plots_scuc", "scuc_losses_combined.png"))
plt.close()

# 4-panel comprehensive plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('SCUC Training Metrics Overview', fontsize=16, fontweight='bold')

# Panel 1: Pg Loss
axes[0, 0].plot(epochs, pg_loss_curve, marker='o', color='tab:blue', linewidth=2)
axes[0, 0].set_title('Pg Loss (Dispatch)')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('MSE Loss')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_yscale('log')

# Panel 2: Unit Commitment Loss
axes[0, 1].plot(epochs, unit_loss_curve, marker='s', color='tab:orange', linewidth=2)
axes[0, 1].set_title('Unit Commitment Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('BCE Loss')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_yscale('log')

# Panel 3: Total Loss
axes[1, 0].plot(epochs, total_loss_curve, marker='^', color='tab:red', linewidth=2)
axes[1, 0].set_title('Total Loss')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].grid(True, alpha=0.3)

# Panel 4: Accuracy
axes[1, 1].plot(epochs, accuracy_curve, marker='D', color='tab:green', linewidth=2)
axes[1, 1].axhline(y=0.85, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Target: 0.85')
axes[1, 1].set_title('Commitment Accuracy')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].set_ylim([0, 1])
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(os.path.join("plots_scuc", "scuc_training_overview.png"), dpi=300, bbox_inches='tight')
plt.close()

log_print("\n✅ Saved SCUC plots to 'plots_scuc/' folder:")
log_print("   📊 scuc_total_loss.png - Total loss over epochs")
log_print("   📊 scuc_pg_loss_log.png - Pg loss (log scale)")
log_print("   📊 scuc_unit_loss_log.png - Unit commitment loss (log scale)")
log_print("   📊 scuc_accuracy.png - Commitment accuracy over epochs")
log_print("   📊 scuc_loss_and_accuracy.png - Loss + Accuracy dual-axis plot")
log_print("   📊 scuc_losses_combined.png - All losses normalized")
log_print("   📊 scuc_training_overview.png - 4-panel comprehensive view")

# Log plots to wandb
plots_art = wandb.Artifact("scuc-training-plots", type="plots")
for fname in os.listdir("plots_scuc"):
    fpath = os.path.join("plots_scuc", fname)
    if os.path.isfile(fpath):
        plots_art.add_file(fpath)
wandb.log_artifact(plots_art)

log_print("\n" + "="*80)
log_print("🎉 SCUC Training Completed Successfully!")
log_print("="*80)

# Calculate total training time
total_training_time = time.time() - training_start_time
hours = int(total_training_time // 3600)
minutes = int((total_training_time % 3600) // 60)
seconds = int(total_training_time % 60)

log_print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log_print(f"Run ID: {run_id}")
log_print(f"🔧 SCUC Head Type: {SCUC_HEAD_TYPE.upper()}")
log_print(f"\n⏱️  Training Time:")
log_print(f"   Total: {hours}h {minutes}m {seconds}s ({total_training_time:.1f}s)")
log_print(f"   Avg per epoch: {total_training_time/num_epochs:.1f}s")
log_print(f"   Batches per epoch: {max_steps_per_epoch}")
log_print(f"   Total batches processed: {num_epochs * max_steps_per_epoch}")
log_print(f"\n📁 Models saved in: {model_save_dir}")
log_print(f"📊 Plots saved in: plots_scuc/")
log_print(f"🔍 Best model: {best_model_path}")
log_print(f"   Best Loss: {best_total_loss:.4f}")
log_print(f"   Final Accuracy: {accuracy_curve[-1]:.3f}" if accuracy_curve else "")
log_print(f"🏷️  W&B Run name: {wandb_run_name}")
log_print(f"\n📈 Training Summary:")
log_print(f"   Pg Loss: {pg_loss_curve[0]:.4f} → {pg_loss_curve[-1]:.4f} ({((pg_loss_curve[-1]-pg_loss_curve[0])/pg_loss_curve[0]*100):+.1f}%)")
log_print(f"   Unit Loss: {unit_loss_curve[0]:.4f} → {unit_loss_curve[-1]:.4f} ({((unit_loss_curve[-1]-unit_loss_curve[0])/unit_loss_curve[0]*100):+.1f}%)")
log_print(f"   Accuracy: {accuracy_curve[0]:.3f} → {accuracy_curve[-1]:.3f} ({((accuracy_curve[-1]-accuracy_curve[0])/accuracy_curve[0]*100):+.1f}%)")
log_print(f"   Total Loss: {total_loss_curve[0]:.4f} → {total_loss_curve[-1]:.4f} ({((total_loss_curve[-1]-total_loss_curve[0])/total_loss_curve[0]*100):+.1f}%)")
log_print("="*80)
log_print(f"📝 Full training log saved to: {log_filepath}")
log_print("="*80)

# Close log file
log_file.close()

