"""
Full training script using Augmented Lagrangian method for ACOPF, it uses PyTorch Lightning for streamlined training.


"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path
import os

# Set CUDA_VISIBLE_DEVICES early to prevent PyTorch from detecting CUDA
# if we're going to run on CPU. This must be done BEFORE importing torch.
# We'll check actual GPU availability later and re-enable if needed.
# For now, we'll let the code detect and handle it properly.
# If you want to force CPU-only, uncomment the next line:
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

try:
    from lightning.pytorch.loggers import WandbLogger
except ImportError:
    WandbLogger = None

# Optional plotting imports
try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from lumina.dataset.opf.opf_dataset import OPFDataset
from lumina.model.opf.augmented_lagrangian import AugmentedLagrangianACOPF
from lumina.model.opf.losses import OPFLossManager
from lumina.model.opf.homo_model import get_gnnNets
from lumina.model.opf.hetero_model import HEAT, HGT, RGAT, OPFHeteroGNN
from lumina.utils.graph_utils import HomoOPFDataset, convert_opf_to_homo


def initialize_model(model, sample_data, device):
    """Lazy initialize model parameters by running a dummy forward pass"""
    print("Initializing model parameters...")
    model = model.to(device)
    sample_data = sample_data.to(device)

    model.eval()
    with torch.no_grad():
        try:
            if isinstance(sample_data, (dict, torch.nn.ParameterDict)) or hasattr(sample_data, 'x_dict'):
                # Ensure inputs are float32
                x_dict = {k: v.float() for k, v in sample_data.x_dict.items()}
                _ = model(x_dict, sample_data.edge_index_dict)
            else:
                if hasattr(sample_data, 'x'):
                    sample_data.x = sample_data.x.float()
                _ = model(sample_data)
            print("Model parameters initialized successfully!")
        except Exception as e:
            print(f"Warning: Model initialization failed: {e}")
            print("Model may still work during training...")

    return model


def get_case_name_mapping():
    """Map short case names to full pglib names."""
    return {
        'case14': 'pglib_opf_case14_ieee',
        'case30': 'pglib_opf_case30_ieee',
        'case57': 'pglib_opf_case57_ieee',
        'case118': 'pglib_opf_case118_ieee',
        'case500': 'pglib_opf_case500_goc',
        'case2000': 'pglib_opf_case2000_goc',
        'case4661': 'pglib_opf_case4661_sdet',
        'case6470': 'pglib_opf_case6470_rte',
        'case10000': 'pglib_opf_case10000_goc',
        'case13659': 'pglib_opf_case13659_pegase',
    }


def parse_case_name(case_input: str) -> str:
    """Parse case name input to full pglib case name."""
    case_mapping = get_case_name_mapping()

    if case_input.startswith('pglib_opf_'):
        return case_input

    if case_input in case_mapping:
        return case_mapping[case_input]

    if not case_input.startswith('case'):
        case_input = 'case' + case_input
        if case_input in case_mapping:
            return case_mapping[case_input]

    available_short = list(case_mapping.keys())
    available_full = list(case_mapping.values())
    raise ValueError(
        f"Invalid case name '{case_input}'. Available short names: {available_short}, or use full names: {available_full}")


class OPFLightningModule(pl.LightningModule):
    def __init__(self, config, case_name, group_id, model_type, loss_type='mse', minmax_scaling=True):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.case_name = case_name
        self.group_id = group_id
        self.model_type = model_type
        self.loss_type = loss_type
        self.minmax_scaling = minmax_scaling

        # Initialize dataset and model
        self._load_dataset()
        self._create_model()

        # Initialize loss manager with appropriate loss type
        self._initialize_loss_manager()

        self.training_step_outputs = []
        self.validation_step_outputs = []

    def _load_dataset(self):
        """Load OPF dataset using config settings."""
        self.dataset = OPFDataset(
            root=self.config['root'],
            case_name=self.case_name,
            group_id=self.group_id,
            local_raw_folder=self.config.get('local_raw_folder'),
            force_reload=False
        )
        print(f"✓ Dataset loaded: {len(self.dataset)} samples")

    def _create_model(self):
        """Create GNN model (Hetero or Homo)."""
        metadata = self.dataset.metadata()
        sample_data = self.dataset[0]

        per_node_output_size = sample_data['bus'].y.shape[-1]
        print(f"Per-node output size: {per_node_output_size}")

        if self.model_type in ['HeteroGNN', 'RGAT', 'HEAT', 'HGT']:
            input_channels = {}
            node_types = list(metadata['nodes'].keys())
            edge_types = list(metadata['edges'].keys())
            metadata_tuple = (node_types, edge_types)

            for node_type in node_types:
                if node_type in sample_data.x_dict:
                    input_channels[node_type] = sample_data[node_type].x.shape[1]

            if self.model_type in self.config['models']:
                model_config = self.config['models'][self.model_type]
            else:
                print(f"Warning: Config for {self.model_type} not found, using HeteroGNN config")
                model_config = self.config['models']['HeteroGNN']

            if self.model_type == 'HeteroGNN':
                ModelClass = OPFHeteroGNN
            elif self.model_type == 'RGAT':
                ModelClass = RGAT
            elif self.model_type == 'HEAT':
                ModelClass = HEAT
            elif self.model_type == 'HGT':
                ModelClass = HGT

            kwargs = {
                'metadata': metadata_tuple,
                'input_channels': input_channels,
                'hidden_channels': model_config['hidden_channels'],
                'out_channels': per_node_output_size,
                'num_layers': model_config['num_layers'],
                'backend': model_config.get('backend', 'sage')
            }

            if self.model_type in ['RGAT', 'HGT']:
                kwargs['num_heads'] = model_config.get('num_heads', 1)
            if self.model_type == 'HEAT':
                kwargs['attention_heads'] = model_config.get('attention_heads', 1)

            self.model = ModelClass(**kwargs)

            # Initialize on CPU first, Lightning will move it
            initialize_model(self.model, sample_data, torch.device('cpu'))
            print(f"✓ {self.model_type} Model created")

        else:
            homo_sample = sample_data.to_homogeneous(add_node_type=True, add_edge_type=True)#convert_opf_to_homo(sample_data)
            input_dim = homo_sample.x.shape[1]

            if hasattr(homo_sample, 'edge_attr') and homo_sample.edge_attr is not None:
                edge_dim = homo_sample.edge_attr.shape[1]
            else:
                edge_dim = 1

            if self.model_type in self.config['models']:
                model_config = self.config['models'][self.model_type]
            elif 'HomoGNN' in self.config['models']:
                model_config = self.config['models']['HomoGNN']
            else:
                model_config = {
                    'hidden_dim': 64,
                    'num_layers': 3,
                    'dropout': 0.1,
                    'readout': 'mean',
                    'edge_dim': edge_dim
                }

            model_config['model_name'] = self.model_type
            model_config['edge_dim'] = edge_dim

            self.model = get_gnnNets(
                input_dim=input_dim,
                output_dim=per_node_output_size,
                model_params=model_config
            )

            initialize_model(self.model, homo_sample, torch.device('cpu'))
            print(f"✓ {self.model_type} Model created")

    def _initialize_loss_manager(self):
        """Initialize loss manager based on loss_type configuration."""
        # Get grid data path for Lagrangian methods
        grid_data = None
        if self.loss_type in ['violated_lagrangian']:
            # Construct path to grid case file
            import os
            root_path = self.config.get('root', 'data')
            grid_data = os.path.join(root_path, self.case_name, 'raw', f'{self.case_name}.m')

            if not os.path.exists(grid_data):
                # Try alternate path
                grid_data = os.path.join(root_path, 'raw', self.case_name, f'{self.case_name}.m')

            if not os.path.exists(grid_data):
                print(f"Warning: Grid data file not found at {grid_data}")
                print("Attempting to use first .m file in raw directory...")
                raw_dir = os.path.join(root_path, self.case_name, 'raw')
                if os.path.exists(raw_dir):
                    m_files = [f for f in os.listdir(raw_dir) if f.endswith('.m')]
                    if m_files:
                        grid_data = os.path.join(raw_dir, m_files[0])
                        print(f"Using: {grid_data}")

        # Get Lagrangian configuration from config file
        lagrangian_config = self.config.get('lagrangian', {})

        # Create loss manager
        self.loss_manager = OPFLossManager(
            loss_type=self.loss_type,
            grid_data=grid_data,
            device=self.device,
            lagrangian_config=lagrangian_config
        )

        print(f"✓ Loss Manager initialized with loss_type='{self.loss_type}'")

        # For backward compatibility, set augmented_lagrangian attribute
        if self.loss_type == 'augmented_lagrangian':
            self.augmented_lagrangian = self.loss_manager.lagrangian

    def setup(self, stage=None):
        n_samples = len(self.dataset)
        n_train = int(n_samples * self.config['train_split'])
        n_val = int(n_samples * self.config['val_split'])

        self.train_dataset = torch.utils.data.Subset(self.dataset, range(n_train))
        self.val_dataset = torch.utils.data.Subset(self.dataset, range(n_train, n_train + n_val))
        self.test_dataset = torch.utils.data.Subset(self.dataset, range(n_train + n_val, n_samples))


    def on_fit_start(self):
        # Loss manager will initialize network parameters on first batch
        pass

    def train_dataloader(self):
        loader_config = self.config['loader']
        return DataLoader(self.train_dataset, batch_size=loader_config['batch_size'],
                          shuffle=loader_config['shuffle'],
                          num_workers=loader_config['num_workers'])

    def val_dataloader(self):
        loader_config = self.config['loader']
        return DataLoader(self.val_dataset, batch_size=loader_config['batch_size'],
                          shuffle=False, num_workers=loader_config['num_workers'])

    def test_dataloader(self):
        loader_config = self.config['loader']
        return DataLoader(self.test_dataset, batch_size=loader_config['batch_size'],
                          shuffle=False, num_workers=loader_config['num_workers'])

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), **self.config['optimizer']['AdamW'])

    def forward(self, batch):
        if self.model_type in ['HeteroGNN', 'RGAT', 'HEAT', 'HGT']:
            # Ensure inputs are float32
            x_dict = {k: v.float() for k, v in batch.x_dict.items()}
            return self.model(x_dict, batch.edge_index_dict, minmax_scaling=self.minmax_scaling)
        else:
            if isinstance(batch, torch.Tensor) or hasattr(batch, 'node_type'):
                homo_batch = batch
            else:
                homo_batch = batch.to_homogeneous(add_node_type=True, add_edge_type=True)
                homo_batch = homo_batch.to(self.device)

            # Ensure inputs are float32
            if hasattr(homo_batch, 'x'):
                homo_batch.x = homo_batch.x.float()
            if hasattr(homo_batch, 'edge_attr') and homo_batch.edge_attr is not None:
                homo_batch.edge_attr = homo_batch.edge_attr.float()

            homo_output = self.model(homo_batch)

            predictions = {}
            node_types = ['bus', 'generator', 'load', 'shunt']
            for i, node_type in enumerate(node_types):
                mask = (homo_batch.node_type == i)
                if mask.any():
                    predictions[node_type] = homo_output[mask]
            return predictions

    def training_step(self, batch, batch_idx):
        batch_size = getattr(batch, 'num_graphs', 1)
        if self.model_type in ['HeteroGNN', 'RGAT', 'HEAT', 'HGT']:
            predictions = self(batch)
        else:
            # in homogeneous models, retain pre-conversion batch dimensions for loss computation
            pre_conversion_batch = batch
            predictions = self(batch)
            batch = pre_conversion_batch

        # Use loss manager to compute loss
        loss, loss_info = self.loss_manager.compute_loss(predictions, batch, return_info=True)

        # Update Lagrange multipliers/penalty when triggers are met
        self.loss_manager.update_lagrangian(
            constraint_violation=loss_info.get('constraint_violation'),
            constraints=loss_info.get('constraints'),
            is_training=self.training
        )

        # Log metrics
        self.log('loss/total', loss, batch_size=batch_size)

        if 'objective' in loss_info:
            self.log('loss/task', loss_info['objective'], batch_size=batch_size)
        if 'lagrange_term' in loss_info:
            self.log('loss/lagrangian', loss_info['lagrange_term'], batch_size=batch_size)
        if 'penalty_term' in loss_info:
            self.log('loss/penalty', loss_info['penalty_term'], batch_size=batch_size)
        if 'raw_constraint_violation' in loss_info:
            self.log('feas/total_violation', loss_info['raw_constraint_violation'], batch_size=batch_size)
        if 'ema_constraint_violation' in loss_info:
            self.log('feas/total_violation_ema', loss_info['ema_constraint_violation'], batch_size=batch_size)
        if 'p_balance_rmse' in loss_info:
            self.log('feas/p_balance_rmse_pu', loss_info['p_balance_rmse'], batch_size=batch_size)
        if 'q_balance_rmse' in loss_info:
            self.log('feas/q_balance_rmse_pu', loss_info['q_balance_rmse'], batch_size=batch_size)
        if 'line_limit_rmse' in loss_info:
            self.log('feas/line_limit_rmse_pu', loss_info['line_limit_rmse'], batch_size=batch_size)
        if 'penalty_parameter' in loss_info:
            self.log('al/mu', loss_info['penalty_parameter'], batch_size=batch_size)
        if 'last_multiplier_norm' in loss_info:
            self.log('al/lagrangian_norm', loss_info['last_multiplier_norm'], batch_size=batch_size)

        return loss

    def on_train_epoch_end(self):

        # Step epoch counter for Lagrangian methods
        self.loss_manager.step_epoch()

    def validation_step(self, batch, batch_idx):
        batch_size = getattr(batch, 'num_graphs', 1)
        if self.model_type in ['HeteroGNN', 'RGAT', 'HEAT', 'HGT']:
            predictions = self(batch)
        else:
            # in homogeneous models, retain pre-conversion batch dimensions for loss computation
            pre_conversion_batch = batch
            predictions = self(batch)
            batch = pre_conversion_batch

        # Use loss manager to compute validation loss
        loss, loss_info = self.loss_manager.compute_loss(predictions, batch, return_info=True)

        # Log validation metrics
        self.log('val/loss/total', loss, batch_size=batch_size)
        if 'objective' in loss_info:
            self.log('val/loss/task', loss_info['objective'], batch_size=batch_size)
        if 'raw_constraint_violation' in loss_info:
            self.log('val/feas/total_violation', loss_info['raw_constraint_violation'], batch_size=batch_size)
        if 'p_balance_rmse' in loss_info:
            self.log('val/feas/p_balance_rmse_pu', loss_info['p_balance_rmse'], batch_size=batch_size)
        if 'q_balance_rmse' in loss_info:
            self.log('val/feas/q_balance_rmse_pu', loss_info['q_balance_rmse'], batch_size=batch_size)
        if 'line_limit_rmse' in loss_info:
            self.log('val/feas/line_limit_rmse_pu', loss_info['line_limit_rmse'], batch_size=batch_size)

        return loss


def main():
    parser = argparse.ArgumentParser(description='OPF Training with PyTorch Lightning - Supports Multiple Loss Types')
    parser.add_argument('--case', type=str, default='case14',
                        help='Case name (short form like case14, case2000 or full pglib name)')
    parser.add_argument('--group_id', type=int, default=0,
                        help='Group ID for dataset (default: 0)')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--model_type', type=str, default='HeteroGNN',
                        choices=['HeteroGNN', 'GCN', 'GAT', 'GIN', 'Transformer', 'RGAT', 'HEAT', 'HGT'],
                        help='Model type to train (default: HeteroGNN)')
    parser.add_argument(
        '--loss_type',
        type=str,
        default='mse',
        choices=[
            'mse',
            'rmse',
            'mae',
            'mape',
            'smooth_l1',
            'augmented_lagrangian',
            'violated_lagrangian'],
        help='Loss function type (default: mse)')
    parser.add_argument('--minmax_scaling', dest='minmax_scaling', action='store_true',
                        help='Apply min-max scaling to model outputs (default: enabled)')
    parser.add_argument('--no_minmax_scaling', dest='minmax_scaling', action='store_false',
                        help='Disable min-max scaling of model outputs')
    parser.set_defaults(minmax_scaling=True)

    parser.add_argument('--accelerator', type=str, default='auto', help='Accelerator type (default: auto)')
    parser.add_argument('--devices', type=int, default=1, help='Number of devices (default: 1)')
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes (default: 1)')
    parser.add_argument('--precision', type=str, default='32-true', help='Precision (default: 32-true)')
    parser.add_argument('--strategy', type=str, default='ddp_find_unused_parameters_true',
                        help='Strategy (default: ddp_find_unused_parameters_true)')
    parser.add_argument('--wandb', action='store_true', default=False,
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='lumina-opf',
                        help='Weights & Biases project name (default: lumina-opf)')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Weights & Biases entity/team name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Custom name for the Weights & Biases run')
    parser.add_argument('--wandb_mode', type=str, default=None,
                        choices=['online', 'offline', 'disabled'],
                        help='Weights & Biases mode override')

    args = parser.parse_args()

    if args.wandb_mode:
        os.environ['WANDB_MODE'] = args.wandb_mode

    wandb_logger = None

    config_path = args.config
    if not os.path.exists(config_path):
        parent_config = os.path.join(Path(__file__).parent.parent, 'config_files', 'single.yaml')
        if os.path.exists(parent_config):
            config_path = parent_config
        else:
            acopf_config = os.path.join(
                Path(__file__).parent.parent.parent,
                'acopf-benchmark',
                'config_files',
                'single.yaml')
            if os.path.exists(acopf_config):
                config_path = acopf_config

    print(f"Loading config from: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load additional model config
    # Assuming the structure is configs/model/{model_type}.yaml relative to the config file or project root
    config_dir = Path(config_path).parent
    # If config_path is configs/config.yaml, parent is configs/
    # We want configs/model/heterognn.yaml

    model_config_path = config_dir / 'model' / 'heterognn.yaml'
    if not model_config_path.exists():
        # Try project root relative
        model_config_path = Path(__file__).parent.parent / 'configs' / 'model' / 'heterognn.yaml'

    if model_config_path.exists():
        print(f"Loading additional model config from: {model_config_path}")
        with open(model_config_path, "r") as f:
            model_config = yaml.safe_load(f)
            if 'models' in model_config:
                if 'models' not in config:
                    config['models'] = {}
                config['models'].update(model_config['models'])

    # Ensure loader and split config exists
    if 'loader' not in config:
        config['loader'] = {
            'batch_size': 32,
            'shuffle': True,
            'num_workers': 4
        }

    if 'train_split' not in config:
        config['train_split'] = 0.8
    if 'val_split' not in config:
        config['val_split'] = 0.1

    sweep_overrides = {}
    if args.wandb:
        if WandbLogger is None:
            print("⚠️ Weights & Biases is not available. Install wandb or omit --wandb.")
        else:
            wandb_kwargs = {}
            if args.wandb_project:
                wandb_kwargs['project'] = args.wandb_project
            if args.wandb_entity:
                wandb_kwargs['entity'] = args.wandb_entity
            if args.wandb_run_name:
                wandb_kwargs['name'] = args.wandb_run_name

            wandb_kwargs.setdefault('log_model', False)

            wandb_logger = WandbLogger(**wandb_kwargs)
            sweep_overrides = dict(wandb_logger.experiment.config)

    def apply_nested(target_dict, dotted_key, value):
        if not isinstance(target_dict, dict):
            return
        if not isinstance(dotted_key, str):
            return

        keys = dotted_key.split('.')
        current = target_dict
        for key in keys[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

    if sweep_overrides:
        reserved_override_keys = {'wandb_version'}
        reserved_args = {'wandb', 'wandb_project', 'wandb_entity', 'wandb_run_name', 'wandb_mode'}
        
        # Map trainer.* keys to args.*
        trainer_to_args_map = {
            'trainer.devices': 'devices',
            'trainer.num_nodes': 'num_nodes',
            'trainer.precision': 'precision',
            'trainer.strategy': 'strategy',
            'trainer.max_epochs': None,  # Goes to config, not args
        }

        for key, value in sweep_overrides.items():
            if not isinstance(key, str):
                continue
            if key in reserved_override_keys or key.startswith('_'):
                continue
            if key in reserved_args:
                continue
            
            # Handle trainer.* keys that map to args
            if key in trainer_to_args_map:
                mapped_key = trainer_to_args_map[key]
                if mapped_key and hasattr(args, mapped_key):
                    setattr(args, mapped_key, value)
                    continue
            
            # Direct args attribute match
            if hasattr(args, key):
                setattr(args, key, value)
                continue
            
            # Apply to nested config
            apply_nested(config, key, value)

    case_name = parse_case_name(args.case)
    print(f"Using case: {case_name}")

    loss_type = args.loss_type

    print(f"🚀 ACOPF Training (Lightning) with loss_type = {loss_type}")
    print("=" * 60)

    # Initialize Lightning Module
    model = OPFLightningModule(
        config=config,
        case_name=case_name,
        group_id=args.group_id,
        model_type=args.model_type,
        loss_type=loss_type,
        minmax_scaling=args.minmax_scaling
    )

    # Initialize Trainer
    trainer_config = copy.deepcopy(config.get('trainer', {}))

    # Auto-detect GPU availability and fallback to CPU if needed
    # The issue: torch.cuda.is_available() can return True even when no GPU is accessible
    # Solution: Try to actually use CUDA, and if it fails, force CPU mode
    cuda_available = False
    gpu_name = None
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        try:
            # Try to actually access GPU to verify it's really available
            gpu_name = torch.cuda.get_device_name(0)
            cuda_available = True
        except (RuntimeError, AttributeError) as e:
            # CUDA libraries installed but no GPU actually available
            cuda_available = False
            gpu_name = None
            # Force CPU mode by hiding CUDA from PyTorch
            # Note: This won't affect already-imported torch, but will help with Lightning
            if 'CUDA_VISIBLE_DEVICES' not in os.environ:
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # Determine accelerator
    if args.accelerator == 'auto':
        if cuda_available and gpu_name:
            trainer_config['accelerator'] = 'gpu'
            print(f"✓ GPU detected: Using CUDA (device: {gpu_name})")
        else:
            trainer_config['accelerator'] = 'cpu'
            print("⚠️  No GPU detected: Falling back to CPU")
    else:
        trainer_config['accelerator'] = args.accelerator
        if args.accelerator in ['gpu', 'cuda'] and not (cuda_available and gpu_name):
            print("⚠️  GPU requested but not available: Falling back to CPU")
            trainer_config['accelerator'] = 'cpu'
    
    # Force CPU if no GPU available (override any GPU settings)
    if not (cuda_available and gpu_name):
        trainer_config['accelerator'] = 'cpu'
        # Monkey-patch torch.cuda.is_available to return False for Lightning
        # This prevents Lightning from trying to access CUDA
        original_is_available = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
    
    # Set devices - if CPU, use 'auto', if GPU and available, use specified devices
    if trainer_config['accelerator'] == 'cpu':
        trainer_config['devices'] = 'auto'  # CPU uses 'auto'
    else:
        trainer_config['devices'] = args.devices
    
    trainer_config['num_nodes'] = args.num_nodes
    trainer_config['precision'] = args.precision
    trainer_config['strategy'] = args.strategy

    # Handle sync_batchnorm logic (only for multi-GPU)
    if trainer_config['accelerator'] != 'cpu' and args.devices > 1:
        trainer_config['sync_batchnorm'] = True
    else:
        trainer_config['sync_batchnorm'] = False

    # Initialize WandB logger
    from lightning.pytorch.loggers import WandbLogger
    wandb_logger = WandbLogger(
        save_dir=config['logging_dir'],
        project='lumina-training',
        name=f'acopf-{args.model_type}-{loss_type}',
        log_model=False
    )
    # Setup callbacks
    if args.loss_type in ['augmented_lagrangian', 'violated_lagrangian']:
        # For Lagrangian methods, monitor constraint violation
        monitoring_metric = 'val/feas/total_violation'
        checkpoint_filename = f'best-{case_name}-{{epoch:02d}}-{{val/feas/total_violation:.4f}}'
    else:
        monitoring_metric = 'val/loss/total'
        checkpoint_filename = f'best-{case_name}-{{epoch:02d}}-{{val/loss/total:.4f}}'


    early_stop_callback = EarlyStopping(
        monitor=monitoring_metric,
        patience=20,
        verbose=True,
        mode='min'
    )

    callbacks = list(trainer_config.pop('callbacks', []))
    if trainer_config.get('enable_checkpointing', True):
        checkpoint_callback = ModelCheckpoint(
            monitor=monitoring_metric,
            filename=checkpoint_filename,
            save_top_k=1,
            mode='min',
        )
        callbacks.extend([checkpoint_callback, early_stop_callback])
    else:
        callbacks.extend([early_stop_callback])

    trainer_kwargs = {**trainer_config, 'callbacks': callbacks}

    if wandb_logger is not None:
        run_metadata = {
            'case_name': case_name,
            'config_path': str(config_path),
            'group_id': args.group_id,
            'loss_type': loss_type,
            'model_type': args.model_type,
            'cli_args': {k: v for k, v in vars(args).items()
                         if k not in {'wandb', 'wandb_project', 'wandb_entity', 'wandb_run_name', 'wandb_mode'}},
        }
        try:
            wandb_logger.experiment.config.update({'run_metadata': run_metadata}, allow_val_change=True)
        except AttributeError:
            pass

    if wandb_logger is not None:
        trainer_kwargs['logger'] = wandb_logger

    trainer = pl.Trainer(**trainer_kwargs)

    # Train
    trainer.fit(model)

    if wandb_logger is not None:
        try:
            wandb_logger.experiment.summary['best_model_path'] = checkpoint_callback.best_model_path
        except AttributeError:
            pass

    print("\n🎉 Training completed!")
    if trainer_config.get('enable_checkpointing', True):
        print(f"💾 Best model saved to: {checkpoint_callback.best_model_path}")

    if wandb_logger is not None:
        try:
            import wandb
            wandb.finish()
        except ImportError:
            pass


if __name__ == "__main__":
    main()
