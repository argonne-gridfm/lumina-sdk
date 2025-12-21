import argparse
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import yaml
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Optional plotting imports
try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# Optional W&B logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from lumina.dataset.opf.opf_dataset import OPFDataset
from lumina.model.opf.losses import OPFLossManager
from lumina.model.opf.homo_model import get_gnnNets
from lumina.model.opf.hetero_model import HEAT, HGT, RGAT, OPFHeteroGNN
from lumina.utils.graph_utils import HomoOPFDataset, convert_opf_to_homo
from lumina.utils.throughput import ThroughputTracker

def initialize_model(model, sample_data, device):
    if dist.get_rank() == 0:
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
            if dist.get_rank() == 0:
                print("Model parameters initialized successfully!")
        except Exception as e:
            if dist.get_rank() == 0:
                print(f"Warning: Model initialization failed: {e}")
                print("Model may still work during training...")

    return model


def get_case_name_mapping():
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


class OPFTrainer:
    def __init__(self, config, case_name, group_id, model_type, loss_type='mse', 
                 minmax_scaling=True, local_rank=0, global_rank = 0, world_size=1):
        self.config = config
        self.case_name = case_name
        self.group_id = group_id
        self.model_type = model_type
        self.loss_type = loss_type
        self.minmax_scaling = minmax_scaling
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{local_rank}')

        training_config = self.config['training']
        self.max_epochs = training_config["max_epochs"]
        self.patience = training_config["patience"]
        self.grad_clip_val = training_config.get("gradient_clip_val")
        self.grad_clip_algo = training_config.get("gradient_clip_algorithm", "norm")
        self.accumulate_grad_batches = max(1, int(training_config.get("accumulate_grad_batches", 1)))
        self.log_every_n_steps = training_config.get("log_every_n_steps", 0)
        self.val_check_interval = max(1, int(training_config.get("val_check_interval", 1)))

        self.checkpoint_dir = config['checkpoint_dir']
        
        # Initialize dataset and model
        self._load_dataset()
        self._create_dataloaders()
        self._create_model()
        self._initialize_loss_manager()
        
        # Setup optimizer
        optimizer_config = config['optimizer']
        if 'Adam' in optimizer_config:
            self.optimizer = optim.Adam(self.model.parameters(), **optimizer_config['Adam'])
        elif 'AdamW' in optimizer_config:
            self.optimizer = optim.AdamW(self.model.parameters(), **optimizer_config['AdamW'])
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.global_step = 0

        self.wandb_run = None
        self.wandb_enabled = False

        self.train_metric_names = [
            'loss/total',
            'loss/task',
            'loss/lagrangian',
            'loss/penalty',
            'feas/total_violation',
            'feas/total_violation_ema',
            'feas/p_balance_rmse_pu',
            'feas/q_balance_rmse_pu',
            'feas/line_limit_rmse_pu',
            'al/mu',
            'al/lagrangian_norm',
        ]
        self.val_metric_names = [
            'val/loss/total',
            'val/loss/task',
            'val/feas/total_violation',
            'val/feas/p_balance_rmse_pu',
            'val/feas/q_balance_rmse_pu',
            'val/feas/line_limit_rmse_pu',
        ]
        self.train_metric_map = [
            ('objective', 'loss/task'),
            ('lagrange_term', 'loss/lagrangian'),
            ('penalty_term', 'loss/penalty'),
            ('raw_constraint_violation', 'feas/total_violation'),
            ('ema_constraint_violation', 'feas/total_violation_ema'),
            ('p_balance_rmse', 'feas/p_balance_rmse_pu'),
            ('q_balance_rmse', 'feas/q_balance_rmse_pu'),
            ('line_limit_rmse', 'feas/line_limit_rmse_pu'),
            ('penalty_parameter', 'al/mu'),
            ('last_multiplier_norm', 'al/lagrangian_norm'),
        ]
        self.val_metric_map = [
            ('raw_constraint_violation', 'val/feas/total_violation'),
            ('p_balance_rmse', 'val/feas/p_balance_rmse_pu'),
            ('q_balance_rmse', 'val/feas/q_balance_rmse_pu'),
            ('line_limit_rmse', 'val/feas/line_limit_rmse_pu'),
        ]
        self.train_metric_groups = [
            ['loss/total', 'loss/task', 'loss/lagrangian', 'loss/penalty'],
            ['feas/total_violation', 'feas/total_violation_ema', 'feas/p_balance_rmse_pu',
             'feas/q_balance_rmse_pu', 'feas/line_limit_rmse_pu'],
            ['al/mu', 'al/lagrangian_norm'],
        ]
        self.val_metric_groups = [
            ['val/loss/total', 'val/loss/task'],
            ['val/feas/total_violation', 'val/feas/p_balance_rmse_pu',
             'val/feas/q_balance_rmse_pu', 'val/feas/line_limit_rmse_pu'],
        ]

        self._init_wandb()
        self.throughput_tracker = None
        if training_config.get("throughput_enabled", True):
            self.throughput_tracker = ThroughputTracker(
                config=self.config,
                world_size=self.world_size,
                global_rank=self.global_rank,
                get_global_step=lambda: self.global_step,
                wandb_enabled=self.wandb_enabled,
            )
            if not self.throughput_tracker.enabled:
                self.throughput_tracker = None
        
    def _load_dataset(self):
        dataset_kwargs = dict(
            root=self.config['root'],
            case_name=self.case_name,
            group_id=self.group_id,
            local_raw_folder=self.config.get('local_raw_folder'),
            force_reload=False,
        )
        if dist.is_available() and dist.is_initialized() and self.world_size > 1:
            if self.global_rank == 0:
                self.dataset = OPFDataset(**dataset_kwargs)
            dist.barrier()
            if self.global_rank != 0:
                self.dataset = OPFDataset(**dataset_kwargs)
        else:
            self.dataset = OPFDataset(**dataset_kwargs)
        if self.global_rank == 0:
            print(f"Dataset loaded: {len(self.dataset)} samples")
    
    def _create_dataloaders(self):
        n_samples = len(self.dataset)
        n_train = int(n_samples * self.config['train_split'])
        n_val = int(n_samples * self.config['val_split'])
        
        train_dataset = torch.utils.data.Subset(self.dataset, range(n_train))
        val_dataset = torch.utils.data.Subset(self.dataset, range(n_train, n_train + n_val))
        test_dataset = torch.utils.data.Subset(self.dataset, range(n_train + n_val, n_samples))
        
        if self.model_type not in ['HeteroGNN', 'RGAT', 'HEAT', 'HGT']:
            train_dataset = HomoOPFDataset(train_dataset)
            val_dataset = HomoOPFDataset(val_dataset)
            test_dataset = HomoOPFDataset(test_dataset)
        
        loader_config = self.config['loader']
        
        # Create distributed samplers
        self.train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.global_rank,
            shuffle=loader_config['shuffle']
        )
        
        self.val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=self.world_size,
            rank=self.global_rank,
            shuffle=False
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=loader_config['batch_size'],
            sampler=self.train_sampler,
            num_workers=loader_config['num_workers'],
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=loader_config['batch_size'],
            sampler=self.val_sampler,
            num_workers=loader_config['num_workers'],
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=loader_config['batch_size'],
            shuffle=False,
            num_workers=loader_config['num_workers'],
            pin_memory=True
        )
    
    def _create_model(self):
        metadata = self.dataset.metadata()
        sample_data = self.dataset[0]

        per_node_output_size = sample_data['bus'].y.shape[-1]
        if self.global_rank == 0:
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
                if self.global_rank == 0:
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
            initialize_model(self.model, sample_data, self.device)
            
            if self.global_rank == 0:
                print(f"{self.model_type} Model created")

        else:
            homo_sample = convert_opf_to_homo(sample_data)
            input_dim = homo_sample.x.shape[1]

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
                    'edge_dim': homo_sample.edge_attr.shape[1]
                }

            model_config['model_name'] = self.model_type
            if 'edge_dim' not in model_config:
                model_config['edge_dim'] = homo_sample.edge_attr.shape[1]

            self.model = get_gnnNets(
                input_dim=input_dim,
                output_dim=per_node_output_size,
                model_params=model_config
            )

            initialize_model(self.model, homo_sample, self.device)
            if self.global_rank == 0:
                print(f"{self.model_type} Model created")

        # Wrap model with DDP
        self.model = DDP(self.model, 
                         device_ids=[self.local_rank],
                         find_unused_parameters=True)
    
    def _initialize_loss_manager(self):
        grid_data = None
        self.loss_manager = OPFLossManager(
            loss_type=self.loss_type,
            grid_data=grid_data,
            device=self.device,
        )
        
        if self.global_rank == 0:
            print(f"Loss Manager initialized with loss_type='{self.loss_type}'")

    def _clip_gradients(self):
        if self.grad_clip_val is None or self.grad_clip_val <= 0:
            return

        parameters = [p for p in self.model.parameters() if p.requires_grad]
        if self.grad_clip_algo == "value":
            torch.nn.utils.clip_grad_value_(parameters, self.grad_clip_val)
        else:
            torch.nn.utils.clip_grad_norm_(parameters, self.grad_clip_val)

    def _init_wandb(self):
        if not WANDB_AVAILABLE or self.global_rank != 0:
            return
        logging_dir = self.config.get('logging_dir')
        run_name = f"acopf-ddp-{self.model_type}-{self.loss_type}"
        try:
            self.wandb_run = wandb.init(
                project='lumina-training',
                name=run_name,
                dir=logging_dir,
                config=self.config,
            )
            self.wandb_enabled = True
        except Exception as exc:
            print(f"Warning: W&B init failed: {exc}")
            self.wandb_run = None
            self.wandb_enabled = False

    def _should_log_step(self):
        if not self.wandb_enabled:
            return False
        if self.log_every_n_steps and self.log_every_n_steps > 0:
            return self.global_step % self.log_every_n_steps == 0
        return True

    def _log_wandb_step(self, loss_value, loss_info):
        if not self._should_log_step():
            return
        metrics = {'loss/total': self._as_float(loss_value)}
        for info_key, metric_name in self.train_metric_map:
            if info_key in loss_info:
                metric_value = self._as_float(loss_info[info_key])
                if metric_value is not None:
                    metrics[metric_name] = metric_value
        wandb.log(metrics, step=self.global_step)

    def _log_wandb_validation(self, metric_avgs):
        if not self.wandb_enabled or not metric_avgs:
            return
        metrics = dict(metric_avgs)
        wandb.log(metrics, step=self.global_step)

    def _as_float(self, value):
        if value is None:
            return None
        if torch.is_tensor(value):
            if value.numel() == 1:
                return value.detach().item()
            return value.detach().float().mean().item()
        if isinstance(value, np.ndarray):
            return float(value.mean())
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _init_metric_trackers(self, metric_names):
        metric_sums = {name: 0.0 for name in metric_names}
        metric_counts = {name: 0.0 for name in metric_names}
        return metric_sums, metric_counts

    def _add_metric(self, metric_sums, metric_counts, name, value, weight=1.0):
        numeric_value = self._as_float(value)
        if numeric_value is None:
            return
        metric_sums[name] += numeric_value * weight
        metric_counts[name] += weight

    def _reduce_metrics(self, metric_sums, metric_counts):
        if dist.is_available() and dist.is_initialized() and self.world_size > 1:
            for name in metric_sums:
                sum_tensor = torch.tensor(metric_sums[name], device=self.device)
                count_tensor = torch.tensor(metric_counts[name], device=self.device)
                dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
                metric_sums[name] = sum_tensor.item()
                metric_counts[name] = count_tensor.item()
        return metric_sums, metric_counts

    def _compute_metric_avgs(self, metric_sums, metric_counts):
        metric_avgs = {}
        for name, total in metric_sums.items():
            count = metric_counts.get(name, 0.0)
            if count > 0:
                metric_avgs[name] = total / count
        return metric_avgs

    def _print_metric_groups(self, title, metric_avgs, groups):
        if not metric_avgs:
            return
        print(title)
        for names in groups:
            parts = []
            for name in names:
                if name in metric_avgs:
                    parts.append(f"{name}: {metric_avgs[name]:.4f}")
            if parts:
                print("    " + ", ".join(parts))
    
    def forward(self, batch):
        if self.model_type in ['HeteroGNN', 'RGAT', 'HEAT', 'HGT']:
            x_dict = {k: v.float() for k, v in batch.x_dict.items()}
            return self.model.module(x_dict, batch.edge_index_dict, minmax_scaling=self.minmax_scaling)
        else:
            if isinstance(batch, torch.Tensor) or hasattr(batch, 'node_type'):
                homo_batch = batch
            else:
                homo_batch = convert_opf_to_homo(batch)
                homo_batch = homo_batch.to(self.device)

            if hasattr(homo_batch, 'x'):
                homo_batch.x = homo_batch.x.float()
            if hasattr(homo_batch, 'edge_attr') and homo_batch.edge_attr is not None:
                homo_batch.edge_attr = homo_batch.edge_attr.float()

            homo_output = self.model.module(homo_batch)

            predictions = {}
            node_types = ['bus', 'generator', 'load', 'shunt']
            for i, node_type in enumerate(node_types):
                mask = (homo_batch.node_type == i)
                if mask.any():
                    predictions[node_type] = homo_output[mask]
            return predictions
    
    def train_epoch(self, epoch):
        self.model.train()
        self.train_sampler.set_epoch(epoch)
        
        total_loss = 0.0
        total_task_loss = 0.0
        num_batches = 0
        total_steps = len(self.train_loader)
        metric_sums, metric_counts = self._init_metric_trackers(self.train_metric_names)
        step_start_time = None
        step_samples = 0
        accum_batches = 0
        tracker = self.throughput_tracker

        # Create progress bar only on rank 0 and if W&B is not enabled
        if self.global_rank == 0 and not self.wandb_enabled:
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        else:
            pbar = self.train_loader
        
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            # Check if we should start throughput measurement
            is_step_start = accum_batches == 0
            if is_step_start:
                if tracker:
                    tracker.maybe_start_measurement()
                    if tracker.measure_active():
                        tracker.accelerator_synchronize()
                        step_start_time = time.perf_counter()
                        step_samples = 0

            batch = batch.to(self.device)
            
            # Accumulate num of samples for throughput measurement
            if tracker and tracker.measure_active():
                step_samples += tracker.get_batch_samples(batch)

            # Forward pass and compute loss
            predictions = self.forward(batch)
            loss, loss_info = self.loss_manager.compute_loss(predictions, batch, return_info=True)

            loss_value = loss.item()
            self._add_metric(metric_sums, metric_counts, 'loss/total', loss_value)
            loss = loss / self.accumulate_grad_batches
            
            loss.backward()

            should_step = ((batch_idx + 1) % self.accumulate_grad_batches == 0) or ((batch_idx + 1) == total_steps)

            if should_step:
                if tracker and tracker.measure_active():
                    tracker.accelerator_synchronize()
                self._clip_gradients()
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.global_step += 1
                self._log_wandb_step(loss_value, loss_info)
                if tracker:
                    step_metrics = None
                    if tracker.measure_active():
                        tracker.accelerator_synchronize()
                        step_time = time.perf_counter() - step_start_time
                        total_samples = step_samples * self.world_size
                        samples_per_sec = total_samples / step_time if step_time > 0 else 0.0
                        step_metrics = {
                            'throughput/samples_per_sec': samples_per_sec,
                        }
                    tracker.on_step_end(step_metrics)
                accum_batches = 0
            else:
                accum_batches += 1
            
            total_loss += loss_value
            if 'objective' in loss_info:
                objective_value = self._as_float(loss_info['objective'])
                if objective_value is not None:
                    total_task_loss += objective_value
                    self._add_metric(metric_sums, metric_counts, 'loss/task', objective_value)
            for info_key, metric_name in self.train_metric_map:
                if info_key in loss_info:
                    self._add_metric(metric_sums, metric_counts, metric_name, loss_info[info_key])
            num_batches += 1
            
            if self.global_rank == 0 and not self.wandb_enabled:
                if self.log_every_n_steps and self.log_every_n_steps > 0:
                    if self.global_step % self.log_every_n_steps == 0:
                        pbar.set_postfix({'loss': loss_value})
                else:
                    pbar.set_postfix({'loss': loss_value})
        
        avg_loss = total_loss / num_batches
        avg_task_loss = total_task_loss / num_batches
        
        # Reduce losses across all processes
        loss_tensor = torch.tensor([avg_loss, avg_task_loss], device=self.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)

        metric_sums, metric_counts = self._reduce_metrics(metric_sums, metric_counts)
        metric_avgs = self._compute_metric_avgs(metric_sums, metric_counts)
        
        self.loss_manager.step_epoch()
        
        return loss_tensor[0].item(), loss_tensor[1].item(), metric_avgs
    
    def validate(self):
        self.model.eval()
        
        total_loss = 0.0
        total_task_loss = 0.0
        num_batches = 0
        metric_sums, metric_counts = self._init_metric_trackers(self.val_metric_names)
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                
                predictions = self.forward(batch)
                loss, loss_info = self.loss_manager.compute_loss(predictions, batch, return_info=True)
                
                loss_value = loss.item()
                total_loss += loss_value
                self._add_metric(metric_sums, metric_counts, 'val/loss/total', loss_value)
                if 'objective' in loss_info:
                    objective_value = self._as_float(loss_info['objective'])
                    if objective_value is not None:
                        total_task_loss += objective_value
                        self._add_metric(metric_sums, metric_counts, 'val/loss/task', objective_value)
                for info_key, metric_name in self.val_metric_map:
                    if info_key in loss_info:
                        self._add_metric(metric_sums, metric_counts, metric_name, loss_info[info_key])
                num_batches += 1
        
        # Average losses
        avg_loss = total_loss / num_batches
        avg_task_loss = total_task_loss / num_batches
        
        # Reduce losses across all processes
        loss_tensor = torch.tensor([avg_loss, avg_task_loss], device=self.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        
        metric_sums, metric_counts = self._reduce_metrics(metric_sums, metric_counts)
        metric_avgs = self._compute_metric_avgs(metric_sums, metric_counts)
        
        return loss_tensor[0].item(), loss_tensor[1].item(), metric_avgs
    
    def save_checkpoint(self, filepath):
        if self.global_rank == 0:
            checkpoint = {
                'epoch': self.current_epoch,
                'model_state_dict': self.model.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_val_loss': self.best_val_loss,
            }
            torch.save(checkpoint, filepath)
            print(f"Checkpoint saved to {filepath}")
    
    def train(self):
        checkpoint_dir = self.checkpoint_dir
        if self.global_rank == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
        if self.throughput_tracker:
            self.throughput_tracker.write_metadata()
        
        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            
            # Training
            train_loss, train_task_loss, train_metrics = self.train_epoch(epoch)
            
            # Validation
            should_validate = ((epoch + 1) % self.val_check_interval == 0)
            val_loss = val_task_loss = None
            val_metrics = None
            if should_validate:
                val_loss, val_task_loss, val_metrics = self.validate()
                self._log_wandb_validation(val_metrics)
            
            if self.global_rank == 0:
                # W&B step logging is handled during training; add epoch-level logging here if needed.
                print(f"\nEpoch {epoch}:")
                print(f"  Train Loss: {train_loss:.4f}, Train Task: {train_task_loss:.4f}")
                self._print_metric_groups("  Train Metrics:", train_metrics, self.train_metric_groups)
                if should_validate:
                    print(f"  Val Loss: {val_loss:.4f}, Val Task: {val_task_loss:.4f}")
                    if val_metrics:
                        self._print_metric_groups("  Val Metrics:", val_metrics, self.val_metric_groups)
                else:
                    print("  Validation skipped this epoch")
            
            # Early stopping and checkpointing (only on rank 0)
            if should_validate:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    
                    checkpoint_path = os.path.join(
                        checkpoint_dir,
                        f'best-{self.case_name}-epoch{epoch:02d}-val{val_loss:.4f}.pt'
                    )
                    self.save_checkpoint(checkpoint_path)
                else:
                    self.patience_counter += 1
                    if self.global_rank == 0:
                        print(f"  No improvement. Patience: {self.patience_counter}/{self.patience}")
                    
                    if self.patience_counter >= self.patience:
                        if self.global_rank == 0:
                            print(f"\nEarly stopping triggered after {epoch+1} epochs")
                        break
            
            dist.barrier()

        if self.throughput_tracker:
            self.throughput_tracker.finalize(partial=True)

        if self.wandb_enabled:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='OPF Training with PyTorch DDP')
    parser.add_argument('--case', type=str, default='case14',
                        help='Case name (short form like case14, case2000 or full pglib name)')
    parser.add_argument('--group_id', type=int, default=0,
                        help='Group ID for dataset (default: 0)')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--model_type', type=str, default='HeteroGNN',
                        choices=['HeteroGNN', 'RGAT', 'HGT'],
                        help='Model type to train (default: HeteroGNN)')
    parser.add_argument('--loss_type', type=str, default='mse',
                        choices=['mse', 'rmse', 'mae', 'mape', 'smooth_l1'],
                        help='Loss function type (default: mse)')
    parser.add_argument('--minmax_scaling', dest='minmax_scaling', action='store_true',
                        help='Apply min-max scaling to model outputs (default: enabled)')
    
    args = parser.parse_args()
    
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    global_rank = int(os.environ.get('RANK', 0))

    # Initialize process group
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=global_rank,
        device_id=local_rank
    )

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    
    if global_rank == 0:
        print(f"ACOPF Training (PyTorch DDP)")
        print("=" * 60)
        print(f"World Size: {world_size}")
    
    # Load config
    config_path = args.config
    if not os.path.exists(config_path):
        parent_config = os.path.join(Path(__file__).parent.parent, 'config_files', 'single.yaml')
        if os.path.exists(parent_config):
            config_path = parent_config
        else:
            acopf_config = os.path.join(
                Path(__file__).parent.parent.parent,
                'configs',
                'config.polaris.ddp.yaml')
            if os.path.exists(acopf_config):
                config_path = acopf_config
    
    if global_rank == 0:
        print(f"Loading config from: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Load model config
    config_dir = Path(config_path).parent
    model_config_path = config_dir / 'model' / 'heterognn.yaml'
    if not model_config_path.exists():
        model_config_path = Path(__file__).parent.parent / 'configs' / 'model' / 'heterognn.yaml'
    
    if model_config_path.exists():
        if global_rank == 0:
            print(f"Loading additional model config from: {model_config_path}")
        with open(model_config_path, "r") as f:
            model_config = yaml.safe_load(f)
            if 'models' in model_config:
                if 'models' not in config:
                    config['models'] = {}
                config['models'].update(model_config['models'])
    
    case_name = parse_case_name(args.case)
    
    # Initialize trainer
    trainer = OPFTrainer(
        config=config,
        case_name=case_name,
        group_id=args.group_id,
        model_type=args.model_type,
        loss_type=args.loss_type,
        minmax_scaling=args.minmax_scaling,
        local_rank=local_rank,
        global_rank=global_rank,
        world_size=world_size
    )

    # Train
    trainer.train()
    
    if global_rank == 0:
        print("\nTraining completed!")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
