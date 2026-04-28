"""
Hyperparameter Tuning Script using Optuna for ACOPF Training
============================================================

This script uses Optuna (Bayesian optimization) to automatically find the best
hyperparameters for training OPF models. It's an alternative to W&B sweeps that
runs locally without requiring internet or W&B accounts.

HOW TO RUN:
-----------

Basic usage (50 trials):
    python example/opf/tune_opf_optuna.py --case case14 --model_type HeteroGNN --loss_type mse --n_trials 50

Advanced usage:
    python example/opf/tune_opf_optuna.py \\
        --case case30 \\
        --model_type RGAT \\
        --loss_type mse \\
        --n_trials 100 \\
        --max_epochs 100 \\
        --pruning \\
        --storage sqlite:///optuna_study.db

With parallel execution (4 workers):
    python example/opf/tune_opf_optuna.py \\
        --case case57 \\
        --n_trials 100 \\
        --n_jobs 4

Resume a previous study:
    python example/opf/tune_opf_optuna.py \\
        --case case14 \\
        --n_trials 100 \\
        --storage sqlite:///optuna_study.db \\
        --study_name opf_case14_HeteroGNN


HYPERPARAMETERS OPTIMIZED:
--------------------------

ALWAYS OPTIMIZED (all models):
    1. Learning Rate (optimizer.Adam.lr)
       - Range: 1e-5 to 1e-3
       - Distribution: Log-uniform (better for learning rates)
    
    2. Weight Decay (optimizer.Adam.weight_decay)
       - Range: 1e-6 to 1e-4
       - Distribution: Log-uniform
    
    3. Batch Size (loader.batch_size)
       - Values: [16, 32, 64]
       - Type: Categorical
    
    4. Max Epochs (trainer.max_epochs)
       - Range: 50 to 100
       - Type: Integer
       - Note: Only optimized if --max_epochs is not specified

FOR HETERO MODELS (HeteroGNN, RGAT, HEAT, HGT):
    5. Hidden Channels (models.hidden_channels)
       - Values: [64, 128, 256]
       - Type: Categorical
    
    6. Number of Layers (models.num_layers)
       - Range: 2 to 5
       - Type: Integer
    
    7. Number of Heads (RGAT, HGT only)
       - Range: 1 to 8
       - Type: Integer
    
    8. Attention Heads (HEAT only)
       - Range: 1 to 8
       - Type: Integer

FOR HOMO MODELS (GCN, GAT, GIN, Transformer):
    5. Hidden Dimension (models.hidden_dim)
       - Values: [64, 128, 256]
       - Type: Categorical
    
    6. Number of Layers (models.num_layers)
       - Range: 2 to 5
       - Type: Integer
    
    7. Dropout (models.dropout)
       - Range: 0.0 to 0.5
       - Type: Float


HOW IT WORKS:
-------------

1. SEARCH SPACE DEFINITION:
   - Optuna defines ranges for each hyperparameter (see above)
   - Uses intelligent sampling (TPE - Tree-structured Parzen Estimator)
   - Not exhaustive - samples promising regions more

2. TRIAL EXECUTION:
   For each trial:
   a) Optuna suggests hyperparameter values
   b) Model is trained with those values
   c) Validation loss is recorded
   d) Optuna uses results to suggest better hyperparameters next time

3. PRUNING (Early Stopping):
   - Unpromising trials are stopped early (saves time)
   - Uses MedianPruner: stops if trial is worse than median of previous trials
   - Can be disabled with --no-pruning flag

4. RESULTS:
   - Best hyperparameters printed to console
   - Results saved to: optuna_results/optuna_*.json
   - Visualizations generated (if plotly installed):
     * optimization_history_*.html - Shows loss over trials
     * param_importances_*.html - Shows which hyperparameters matter most


OUTPUT FILES:
-------------

Results JSON:
    optuna_results/optuna_{case}_{model}_{timestamp}.json
    Contains: best_trial, best_value, best_params, n_trials

Visualizations (if plotly installed):
    optuna_results/optimization_history_{case}_{model}.html
    optuna_results/param_importances_{case}_{model}.html


COMMAND-LINE ARGUMENTS:
-----------------------

Required:
    --case: OPF case name (case14, case30, case57, etc.)

Optional:
    --model_type: Model architecture (default: HeteroGNN)
    --loss_type: Loss function (default: mse)
    --n_trials: Number of optimization trials (default: 50)
    --max_epochs: Max epochs per trial (default: None = optimize 50-100)
    --study_name: Optuna study name for resuming (default: auto-generated)
    --storage: Storage URL for persistence (e.g., sqlite:///optuna.db)
    --direction: minimize or maximize (default: minimize)
    --pruning: Enable early stopping (default: True)
    --n_jobs: Number of parallel jobs (default: 1)
    --timeout: Total timeout in seconds (default: None)


INSTALLATION:
-------------

Install Optuna:
    pip install optuna

Or with project extras:
    pip install -e .[hps]

For visualizations:
    pip install plotly


TIPS:
-----

1. Start with 20-30 trials to get a sense of the search space
2. Use --pruning to save time (stops bad trials early)
3. Use --storage to save progress and resume later
4. Use --n_jobs > 1 for parallel execution (faster but uses more resources)
5. Check optuna_results/ directory for intermediate results


COMPARISON WITH W&B SWEEPS:
----------------------------

Optuna:
    + Simple Python script (no YAML config)
    + Works offline (no internet needed)
    + Built-in pruning
    + Easy to resume with storage
    - No cloud dashboard
    - Manual parallel execution setup

W&B Sweeps:
    + Cloud dashboard for monitoring
    + Automatic distributed execution
    + Easy sharing of results
    - Requires internet
    - More complex setup
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from optuna.integration import PyTorchLightningPruningCallback

try:
    import optuna
    from optuna.integration import PyTorchLightningPruningCallback
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️  Optuna not available. Install with: pip install optuna")

try:
    from lightning.pytorch.loggers import WandbLogger
except ImportError:
    WandbLogger = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lumina.dataset.opf.opf_dataset import OPFDataset
from lumina.model.opf.losses import OPFLossManager
from lumina.model.opf.homo_model import get_gnnNets
from lumina.model.opf.hetero_model import HEAT, HGT, RGAT, OPFHeteroGNN
from lumina.utils.graph_utils import HomoOPFDataset, convert_opf_to_homo
from example.opf.train_opf import OPFLightningModule, parse_case_name, initialize_model


def objective(trial, config, case_name, group_id, model_type, loss_type, max_epochs, n_trials_timeout=None):
    """
    Optuna objective function that defines the hyperparameter search space and runs training.
    
    This function is called once per trial. For each trial:
    1. Optuna suggests hyperparameter values
    2. Model is trained with those values
    3. Validation loss is returned
    4. Optuna uses the result to suggest better hyperparameters for next trial
    
    Args:
        trial: Optuna trial object (provides suggest_* methods)
        config: Base configuration dictionary
        case_name: OPF case name
        group_id: Dataset group ID
        model_type: Model architecture type
        loss_type: Loss function type
        max_epochs: Maximum training epochs (None = optimize, int = fixed)
        n_trials_timeout: Optional timeout in seconds
    
    Returns:
        Validation loss (to be minimized by Optuna)
    """
    # ========================================================================
    # HYPERPARAMETER SEARCH SPACE DEFINITION
    # ========================================================================
    # Optuna will intelligently sample from these ranges using TPE algorithm
    
    # 1. LEARNING RATE (always optimized)
    # Range: 1e-5 to 1e-3, log-uniform distribution
    # Log-uniform is better for learning rates (covers orders of magnitude)
    lr = trial.suggest_float('optimizer.Adam.lr', 1e-5, 1e-3, log=True)
    
    # 2. WEIGHT DECAY (always optimized)
    # Range: 1e-6 to 1e-4, log-uniform distribution
    weight_decay = trial.suggest_float('optimizer.Adam.weight_decay', 1e-6, 1e-4, log=True)
    
    # 3. BATCH SIZE (always optimized)
    # Categorical: chooses one of [16, 32, 64]
    batch_size = trial.suggest_categorical('loader.batch_size', [16, 32, 64])
    
    # 4. TRAINING EPOCHS (optimized if max_epochs not specified)
    # Range: 50 to 100 epochs
    # If --max_epochs is provided, this is fixed to that value
    epochs = trial.suggest_int('trainer.max_epochs', 50, 100) if max_epochs is None else max_epochs
    
    # ========================================================================
    # MODEL-SPECIFIC HYPERPARAMETERS
    # ========================================================================
    
    if model_type in ['HeteroGNN', 'RGAT', 'HEAT', 'HGT']:
        # For Heterogeneous GNN models:
        
        # 5. HIDDEN CHANNELS
        # Categorical: chooses one of [64, 128, 256]
        hidden_channels = trial.suggest_categorical('models.hidden_channels', [64, 128, 256])
        
        # 6. NUMBER OF LAYERS
        # Integer: 2 to 5 layers
        num_layers = trial.suggest_int('models.num_layers', 2, 5)
        
        # 7. ATTENTION HEADS (for RGAT and HGT only)
        if model_type in ['RGAT', 'HGT']:
            # Integer: 1 to 8 heads
            num_heads = trial.suggest_int('models.num_heads', 1, 8)
        
        # 8. ATTENTION HEADS (for HEAT only)
        elif model_type == 'HEAT':
            # Integer: 1 to 8 heads
            attention_heads = trial.suggest_int('models.attention_heads', 1, 8)
    else:
        # For Homogeneous GNN models (GCN, GAT, GIN, Transformer):
        
        # 5. HIDDEN DIMENSION
        # Categorical: chooses one of [64, 128, 256]
        hidden_dim = trial.suggest_categorical('models.hidden_dim', [64, 128, 256])
        
        # 6. NUMBER OF LAYERS
        # Integer: 2 to 5 layers
        num_layers = trial.suggest_int('models.num_layers', 2, 5)
        
        # 7. DROPOUT
        # Float: 0.0 to 0.5 (regularization)
        dropout = trial.suggest_float('models.dropout', 0.0, 0.5)
    
    # ========================================================================
    # APPLY HYPERPARAMETERS TO CONFIG
    # ========================================================================
    # Update the base config with the hyperparameters suggested by Optuna
    trial_config = copy.deepcopy(config)
    
    # Update optimizer config
    if 'optimizer' not in trial_config:
        trial_config['optimizer'] = {}
    if 'Adam' not in trial_config['optimizer']:
        trial_config['optimizer']['Adam'] = {}
    trial_config['optimizer']['Adam']['lr'] = lr
    trial_config['optimizer']['Adam']['weight_decay'] = weight_decay
    
    # Update loader config
    if 'loader' not in trial_config:
        trial_config['loader'] = {}
    trial_config['loader']['batch_size'] = batch_size
    
    # Update trainer config
    if 'trainer' not in trial_config:
        trial_config['trainer'] = {}
    trial_config['trainer']['max_epochs'] = epochs
    
    # Update model config
    if 'models' not in trial_config:
        trial_config['models'] = {}
    if model_type not in trial_config['models']:
        trial_config['models'][model_type] = {}
    
    model_config = trial_config['models'][model_type]
    if model_type in ['HeteroGNN', 'RGAT', 'HEAT', 'HGT']:
        model_config['hidden_channels'] = hidden_channels
        model_config['num_layers'] = num_layers
        if model_type in ['RGAT', 'HGT']:
            model_config['num_heads'] = num_heads
        elif model_type == 'HEAT':
            model_config['attention_heads'] = attention_heads
    else:
        model_config['hidden_dim'] = hidden_dim
        model_config['num_layers'] = num_layers
        model_config['dropout'] = dropout
    
    # Initialize model with trial hyperparameters
    model = OPFLightningModule(
        config=trial_config,
        case_name=case_name,
        group_id=group_id,
        model_type=model_type,
        loss_type=loss_type
    )
    
    # ========================================================================
    # SETUP TRAINER
    # ========================================================================
    trainer_config = copy.deepcopy(trial_config.get('trainer', {}))
    
    # Auto-detect GPU availability (same logic as train_opf.py)
    cuda_available = False
    gpu_name = None
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            cuda_available = True
        except (RuntimeError, AttributeError):
            cuda_available = False
            if 'CUDA_VISIBLE_DEVICES' not in os.environ:
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    if cuda_available and gpu_name:
        trainer_config['accelerator'] = 'gpu'
        trainer_config['devices'] = 1
        print(f"✓ Trial {trial.number}: Using GPU ({gpu_name})")
    else:
        trainer_config['accelerator'] = 'cpu'
        trainer_config['devices'] = 'auto'
        # Monkey-patch to prevent CUDA access
        original_is_available = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
        print(f"✓ Trial {trial.number}: Using CPU")
    
    trainer_config['num_nodes'] = 1
    trainer_config['precision'] = '32-true'
    trainer_config['strategy'] = 'auto'
    
    # ========================================================================
    # SETUP CALLBACKS
    # ========================================================================
    
    # Checkpoint callback (but don't save - we're just tuning)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename=f'trial-{trial.number}-{{epoch:02d}}-{{val_loss:.4f}}',
        save_top_k=0,  # Don't save checkpoints during tuning (saves disk space)
        mode='min',
    )
    
    # Early stopping callback (stops if validation loss doesn't improve)
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,  # Stop if no improvement for 10 epochs
        verbose=False,
        mode='min'
    )
    
    # Optuna pruning callback (stops unpromising trials early)
    # This is the key feature: if this trial is performing worse than previous
    # trials, Optuna will stop it early to save time
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor='val_loss')
    
    callbacks = [checkpoint_callback, early_stop_callback, pruning_callback]
    
    # Remove conflicting arguments from trainer_config to avoid duplicates
    # These are set explicitly below, so remove from config dict first
    trainer_config.pop('enable_progress_bar', None)
    trainer_config.pop('enable_model_summary', None)
    trainer_config.pop('logger', None)
    
    # Enable minimal progress bar for tuning (shows epoch progress)
    trainer = pl.Trainer(
        **trainer_config,
        callbacks=callbacks,
        enable_progress_bar=True,  # Show progress during tuning
        enable_model_summary=False,  # Disable model summary to reduce clutter
        logger=False,  # Disable default logging during tuning
    )
    
    # ========================================================================
    # TRAIN MODEL AND RETURN VALIDATION LOSS
    # ========================================================================
    try:
        # Train the model with the suggested hyperparameters
        print(f"\n🚀 Trial {trial.number}: Starting training...")
        print(f"   Hyperparameters:")
        print(f"     - Learning Rate: {lr:.6f}")
        print(f"     - Weight Decay: {weight_decay:.6f}")
        print(f"     - Batch Size: {batch_size}")
        print(f"     - Max Epochs: {epochs}")
        if model_type in ['HeteroGNN', 'RGAT', 'HEAT', 'HGT']:
            print(f"     - Hidden Channels: {hidden_channels}")
            print(f"     - Num Layers: {num_layers}")
            if model_type in ['RGAT', 'HGT']:
                print(f"     - Num Heads: {num_heads}")
            elif model_type == 'HEAT':
                print(f"     - Attention Heads: {attention_heads}")
        else:
            print(f"     - Hidden Dim: {hidden_dim}")
            print(f"     - Num Layers: {num_layers}")
            print(f"     - Dropout: {dropout:.3f}")
        print()
        
        # Train the model
        trainer.fit(model)
        
        # Get best validation loss from training
        # This is what Optuna will use to decide which hyperparameters are best
        best_val_loss = trainer.callback_metrics.get('val_loss', float('inf'))
        if best_val_loss is None or best_val_loss == float('inf'):
            # Fallback: get from early stopping callback
            if hasattr(early_stop_callback, 'best_score'):
                best_val_loss = early_stop_callback.best_score
            else:
                best_val_loss = float('inf')
        
        print(f"\n✅ Trial {trial.number}: Completed")
        print(f"   Best Validation Loss: {best_val_loss:.6f}")
        print("-" * 60)
        
        # Return validation loss (Optuna will minimize this)
        return float(best_val_loss)
    
    except optuna.TrialPruned:
        # Trial was pruned (stopped early by Optuna)
        # This is normal - Optuna stops bad trials to save time
        raise
    except Exception as e:
        # Trial failed due to error
        print(f"❌ Trial {trial.number} failed: {e}")
        return float('inf')  # Return worst possible value


def main():
    parser = argparse.ArgumentParser(description='OPF Hyperparameter Tuning with Optuna')
    parser.add_argument('--case', type=str, default='case14',
                        help='Case name (short form like case14, case2000 or full pglib name)')
    parser.add_argument('--group_id', type=int, default=0,
                        help='Group ID for dataset (default: 0)')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--model_type', type=str, default='HeteroGNN',
                        choices=['HeteroGNN', 'GCN', 'GAT', 'GIN', 'Transformer', 'RGAT', 'HEAT', 'HGT'],
                        help='Model type to train (default: HeteroGNN)')
    parser.add_argument('--loss_type', type=str, default='mse',
                        choices=['mse', 'rmse', 'mae', 'mape', 'smooth_l1'],
                        help='Loss function type (default: mse)')
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Number of Optuna trials to run (default: 50)')
    parser.add_argument('--max_epochs', type=int, default=None,
                        help='Maximum epochs per trial (default: from config or 50-100 range)')
    parser.add_argument('--study_name', type=str, default=None,
                        help='Optuna study name (for resuming studies)')
    parser.add_argument('--storage', type=str, default=None,
                        help='Optuna storage URL (e.g., sqlite:///optuna.db)')
    parser.add_argument('--direction', type=str, default='minimize',
                        choices=['minimize', 'maximize'],
                        help='Optimization direction (default: minimize)')
    parser.add_argument('--pruning', action='store_true', default=True,
                        help='Enable Optuna pruning (stop unpromising trials early)')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Number of parallel jobs (default: 1)')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Timeout in seconds for entire optimization (default: None)')
    
    args = parser.parse_args()
    
    if not OPTUNA_AVAILABLE:
        print("❌ Optuna is required but not installed.")
        print("   Install with: pip install optuna")
        sys.exit(1)
    
    # ========================================================================
    # MAIN OPTIMIZATION LOOP
    # ========================================================================
    
    print("🔍 Optuna Hyperparameter Tuning for OPF Training")
    print("=" * 60)
    
    # Load config
    config_path = args.config
    if not os.path.exists(config_path):
        parent_config = os.path.join(Path(__file__).parent.parent, 'config_files', 'single.yaml')
        if os.path.exists(parent_config):
            config_path = parent_config
    
    print(f"Loading config from: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Load model config if available
    config_dir = Path(config_path).parent
    model_config_path = config_dir / 'model' / 'heterognn.yaml'
    if not model_config_path.exists():
        model_config_path = Path(__file__).parent.parent / 'configs' / 'model' / 'heterognn.yaml'
    
    if model_config_path.exists():
        print(f"Loading model config from: {model_config_path}")
        with open(model_config_path, "r") as f:
            model_config = yaml.safe_load(f)
            if 'models' in model_config:
                if 'models' not in config:
                    config['models'] = {}
                config['models'].update(model_config['models'])
    
    # Ensure defaults
    if 'loader' not in config:
        config['loader'] = {'batch_size': 32, 'shuffle': True, 'num_workers': 4}
    if 'train_split' not in config:
        config['train_split'] = 0.8
    if 'val_split' not in config:
        config['val_split'] = 0.1
    
    case_name = parse_case_name(args.case)
    
    print(f"\n📊 Study Configuration:")
    print(f"   Case: {case_name}")
    print(f"   Model: {args.model_type}")
    print(f"   Loss: {args.loss_type}")
    print(f"   Trials: {args.n_trials}")
    print(f"   Direction: {args.direction}")
    print(f"   Pruning: {args.pruning}")
    print("=" * 60)
    
    # ========================================================================
    # CREATE OR LOAD OPTUNA STUDY
    # ========================================================================
    # A "study" is Optuna's container for all trials in one optimization run
    # If storage is provided, we can resume previous studies
    
    if args.storage:
        # Persistent study (saved to database/file)
        # Can resume later or share results
        study = optuna.create_study(
            study_name=args.study_name or f"opf_{case_name}_{args.model_type}",
            storage=args.storage,  # e.g., "sqlite:///optuna.db"
            load_if_exists=True,  # Resume if study already exists
            direction=args.direction,  # 'minimize' or 'maximize'
            pruner=optuna.pruners.MedianPruner() if args.pruning else None,
        )
        print(f"📁 Using storage: {args.storage}")
        print(f"   Study name: {study.study_name}")
    else:
        # In-memory study (lost when script ends)
        study = optuna.create_study(
            study_name=args.study_name or f"opf_{case_name}_{args.model_type}",
            direction=args.direction,
            pruner=optuna.pruners.MedianPruner() if args.pruning else None,
        )
        print("📁 Using in-memory study (no persistence)")
        print("   (Use --storage to save progress)")
    
    # ========================================================================
    # RUN OPTIMIZATION
    # ========================================================================
    # This is where the magic happens - Optuna will run n_trials trials,
    # each time calling the objective() function with suggested hyperparameters
    
    print(f"\n🚀 Starting optimization ({args.n_trials} trials)...")
    print("=" * 60)
    print("Each trial will:")
    print("  1. Sample hyperparameters from search space")
    print("  2. Train model with those hyperparameters")
    print("  3. Record validation loss")
    print("  4. Use results to suggest better hyperparameters for next trial")
    print("=" * 60)
    
    study.optimize(
        lambda trial: objective(
            trial, config, case_name, args.group_id, 
            args.model_type, args.loss_type, args.max_epochs
        ),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        timeout=args.timeout,
        show_progress_bar=True,
    )
    
    # ========================================================================
    # DISPLAY RESULTS
    # ========================================================================
    
    print("\n" + "=" * 60)
    print("✅ Optimization Complete!")
    print("=" * 60)
    print(f"\n📈 Best Trial:")
    print(f"   Trial Number: {study.best_trial.number}")
    print(f"   Best Value (val_loss): {study.best_value:.6f}")
    print(f"\n🎯 Best Hyperparameters:")
    print("   (Use these in train_opf.py or config.yaml for final training)")
    for key, value in study.best_params.items():
        print(f"   {key}: {value}")
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    # Save best hyperparameters to JSON file for later use
    
    results_dir = Path(__file__).parent.parent.parent / 'optuna_results'
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / f"optuna_{case_name}_{args.model_type}_{int(time.time())}.json"
    results = {
        'best_trial': study.best_trial.number,
        'best_value': study.best_value,
        'best_params': study.best_params,
        'n_trials': len(study.trials),
        'case_name': case_name,
        'model_type': args.model_type,
        'loss_type': args.loss_type,
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # ========================================================================
    # GENERATE VISUALIZATIONS (optional)
    # ========================================================================
    # Creates interactive HTML plots showing optimization progress
    
    try:
        import optuna.visualization as vis
        print("\n📊 Generating visualizations...")
        
        # Plot 1: Optimization history
        # Shows how validation loss improved over trials
        fig = vis.plot_optimization_history(study)
        fig.write_html(str(results_dir / f"optimization_history_{case_name}_{args.model_type}.html"))
        print(f"   ✓ optimization_history_{case_name}_{args.model_type}.html")
        
        # Plot 2: Parameter importance
        # Shows which hyperparameters matter most for performance
        try:
            fig = vis.plot_param_importances(study)
            fig.write_html(str(results_dir / f"param_importances_{case_name}_{args.model_type}.html"))
            print(f"   ✓ param_importances_{case_name}_{args.model_type}.html")
        except:
            pass  # Need at least 2 completed trials for importance
        
        print(f"\n   Open HTML files in browser to view interactive plots")
        print(f"   Location: {results_dir}")
    except ImportError:
        print("\n   (Install plotly for visualizations: pip install plotly)")


if __name__ == "__main__":
    main()
