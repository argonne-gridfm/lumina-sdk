"""
Multi-case training script using PyTorch Lightning for ACOPF models.

This variant extends the single-case pipeline to support training on multiple
grid cases within a single run. It assumes that all selected cases share the
same feature schema (node/edge types and attribute dimensions) but allows for
different graph sizes.

Supports standard ML losses (mse, rmse, mae, mape, smooth_l1) and the physics
informed augmented/violated Lagrangian variants. Each training step draws
from a single case to keep per-grid constraint states independent.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.optim as optim
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
import yaml

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lumina.dataset.opf.opf_dataset import OPFDataset
from lumina.model.opf.augmented_lagrangian import AugmentedLagrangianACOPF  # noqa: F401 imported for compatibility
from lumina.model.opf.losses import OPFLossManager
from lumina.model.opf.homo_model import get_gnnNets
from lumina.model.opf.hetero_model import HEAT, HGT, RGAT, OPFHeteroGNN
from lumina.utils.graph_utils import convert_opf_to_homo


HETERO_MODEL_TYPES = {'HeteroGNN', 'RGAT', 'HEAT', 'HGT'}


def initialize_model(model, sample_data, device):
    """Lazy initialize model parameters by running a dummy forward pass."""
    print("Initializing model parameters...")
    model = model.to(device)
    sample_data = sample_data.to(device)

    model.eval()
    with torch.no_grad():
        try:
            if isinstance(sample_data, (dict, torch.nn.ParameterDict)) or hasattr(sample_data, 'x_dict'):
                x_dict = {k: v.float() for k, v in sample_data.x_dict.items()}
                _ = model(x_dict, sample_data.edge_index_dict)
            else:
                if hasattr(sample_data, 'x'):
                    sample_data.x = sample_data.x.float()
                _ = model(sample_data)
            print("Model parameters initialized successfully!")
        except Exception as e:  # pragma: no cover - informative logging
            print(f"Warning: Model initialization failed: {e}")
            print("Model may still work during training...")

    model.train()
    return model


def get_case_name_mapping() -> Dict[str, str]:
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
        f"Invalid case name '{case_input}'. Available short names: {available_short}, or use full names: {available_full}"
    )


class MultiCaseOPFLightningModule(pl.LightningModule):
    def __init__(
        self,
        config: Dict,
        case_names: Sequence[str],
        group_id: int,
        model_type: str,
        loss_type: str = 'mse'
    ):
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        self.case_names = list(case_names)
        self.case_index_to_name = {idx: name for idx, name in enumerate(self.case_names)}
        self.case_keys = {idx: f"case_{idx}" for idx in range(len(self.case_names))}
        self.case_key_to_index = {key: idx for idx, key in self.case_keys.items()}
        self.group_id = group_id
        self.model_type = model_type
        self.loss_type = loss_type

        self._load_datasets()
        self._create_model()
        self._initialize_loss_manager()

        self.train_subsets: Dict[int, Subset] = {}
        self.val_subsets: Dict[int, Subset] = {}
        self.test_subsets: Dict[int, Subset] = {}
        self._train_loaders: List[DataLoader] = []
        self._val_loaders: List[DataLoader] = []
        self._test_loaders: List[DataLoader] = []
        self._train_loader_map: Dict[int, DataLoader] = {}
        self._val_loader_map: Dict[int, DataLoader] = {}
        self._test_loader_map: Dict[int, DataLoader] = {}
        self._train_case_indices: List[int] = []
        self._val_case_indices: List[int] = []
        self._test_case_indices: List[int] = []
        self.case_network_stats: Dict[int, Dict[str, torch.Tensor]] = {}

        self.training_step_outputs = []

    def _load_datasets(self) -> None:
        """Load OPF datasets for each requested case."""
        self.case_datasets: List[OPFDataset] = []
        reference_metadata = None
        reference_out_dim = None

        for idx, case_name in enumerate(self.case_names):
            dataset = OPFDataset(
                root=self.config['root'],
                case_name=case_name,
                group_id=self.group_id,
                local_raw_folder=self.config.get('local_raw_folder'),
                force_reload=False,
            )
            print(f"✓ Dataset loaded for {case_name}: {len(dataset)} samples")

            metadata = dataset.metadata()
            sample = dataset[0]

            out_dim = sample['bus'].y.shape[-1]
            if reference_metadata is None:
                reference_metadata = metadata
                reference_out_dim = out_dim
            else:
                if metadata != reference_metadata:
                    raise ValueError(
                        f"Dataset metadata mismatch between cases. {case_name} does not share the same schema."
                    )
                if out_dim != reference_out_dim:
                    raise ValueError(
                        f"Output dimension mismatch detected for case {case_name} "
                        f"(expected {reference_out_dim}, found {out_dim})."
                    )

            self.case_datasets.append(dataset)

        self.reference_metadata = reference_metadata
        self.reference_output_dim = reference_out_dim

    def _create_model(self) -> None:
        """Instantiate the GNN model (heterogeneous or homogeneous)."""
        sample_data = self.case_datasets[0][0]
        per_node_output_size = self.reference_output_dim
        print(f"Per-node output size: {per_node_output_size}")

        if self.model_type in HETERO_MODEL_TYPES:
            input_channels: Dict[str, int] = {}
            node_types = list(self.reference_metadata['nodes'].keys())
            edge_types = list(self.reference_metadata['edges'].keys())
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
            else:  # pragma: no cover - guarded by choices
                raise ValueError(f"Unsupported model type {self.model_type}")

            kwargs = {
                'metadata': metadata_tuple,
                'input_channels': input_channels,
                'hidden_channels': model_config['hidden_channels'],
                'out_channels': per_node_output_size,
                'num_layers': model_config['num_layers'],
                'backend': model_config.get('backend', 'sage'),
            }

            if self.model_type in {'RGAT', 'HGT'}:
                kwargs['num_heads'] = model_config.get('num_heads', 1)
            if self.model_type == 'HEAT':
                kwargs['attention_heads'] = model_config.get('attention_heads', 1)

            self.model = ModelClass(**kwargs)
            initialize_model(self.model, sample_data, torch.device('cpu'))
            print(f"✓ {self.model_type} Model created")
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
                    'edge_dim': homo_sample.edge_attr.shape[1],
                }

            model_config['model_name'] = self.model_type
            if 'edge_dim' not in model_config:
                model_config['edge_dim'] = homo_sample.edge_attr.shape[1]

            self.model = get_gnnNets(
                input_dim=input_dim,
                output_dim=per_node_output_size,
                model_params=model_config,
            )

            initialize_model(self.model, homo_sample, torch.device('cpu'))
            print(f"✓ {self.model_type} Model created")

    def _initialize_loss_manager(self) -> None:
        """Initialize loss managers for each case."""
        lagrangian_config = self.config.get('lagrangian', {})
        self.loss_managers: Dict[int, OPFLossManager] = {}

        for case_idx, case_name in enumerate(self.case_names):
            grid_data = None
            if self.loss_type in {'augmented_lagrangian', 'violated_lagrangian'}:
                grid_data = self._resolve_grid_data(case_name)

            manager = OPFLossManager(
                loss_type=self.loss_type,
                grid_data=grid_data,
                device=torch.device('cpu'),
                lagrangian_config=lagrangian_config,
            )

            self.loss_managers[case_idx] = manager

        print(f"✓ Loss Managers initialized for {len(self.loss_managers)} cases with loss_type='{self.loss_type}'")

    def _resolve_grid_data(self, case_name: str) -> str | None:
        """Locate MATPOWER grid data file for a specific case."""
        root_path = self.config.get('root', 'data')
        default_path = os.path.join(root_path, case_name, 'raw', f'{case_name}.m')
        if os.path.exists(default_path):
            return default_path

        alt_path = os.path.join(root_path, 'raw', case_name, f'{case_name}.m')
        if os.path.exists(alt_path):
            return alt_path

        raw_dir = os.path.join(root_path, case_name, 'raw')
        if os.path.exists(raw_dir):
            m_files = [f for f in os.listdir(raw_dir) if f.endswith('.m')]
            if m_files:
                fallback = os.path.join(raw_dir, m_files[0])
                print(f"Warning: Grid data file not found at default location for {case_name}. Using {fallback}")
                return fallback

        print(f"Warning: Grid data file not found for case {case_name}")
        return None

    def _resolve_case_batch(self, batch, stage: str, dataloader_idx: int | None):
        """Determine the originating case index for a batch."""
        if isinstance(batch, dict):
            if len(batch) != 1:
                raise ValueError("Expected batches from CombinedLoader to contain exactly one case per step.")
            case_key, case_batch = next(iter(batch.items()))

            if isinstance(case_key, str):
                if case_key in self.case_key_to_index:
                    return self.case_key_to_index[case_key], case_batch
                # Allow keys formatted as 'case_0' even if regenerated externally
                if case_key.startswith('case_'):
                    idx = int(case_key.split('_', 1)[1])
                    return idx, case_batch
                raise KeyError(f"Unknown case key '{case_key}' in batch.")

            if isinstance(case_key, int):
                return case_key, case_batch

            raise TypeError(f"Unsupported case key type: {type(case_key)}")

        if isinstance(batch, (list, tuple)):
            if stage == 'train':
                mapping = self._train_case_indices or [0]
            elif stage == 'val':
                mapping = self._val_case_indices or [0]
            else:
                mapping = self._test_case_indices or [0]

            if dataloader_idx is None:
                index = 0
            else:
                index = min(dataloader_idx, len(mapping) - 1)

            case_idx = mapping[index]
            inner_batch = batch[index]
            return case_idx, inner_batch

        if stage == 'train':
            mapping = self._train_case_indices or [0]
        elif stage == 'val':
            mapping = self._val_case_indices or [0]
        else:
            mapping = self._test_case_indices or [0]

        if dataloader_idx is None:
            return mapping[0], batch

        if dataloader_idx >= len(mapping):
            return mapping[-1], batch

        return mapping[dataloader_idx], batch

    def _prepare_case_network_parameters(self, case_idx: int) -> None:
        """Initialize network parameters for augmented Lagrangian loss."""
        if case_idx in self.case_network_stats:
            return

        sample = self.case_datasets[case_idx][0]
        n_bus = sample['bus'].x.shape[0]

        if hasattr(sample, 'Y'):
            Y_dense = torch.tensor(sample.Y.todense(), dtype=torch.float32)
            Y_real = Y_dense.real
            Y_imag = Y_dense.imag
        else:
            Y_real = torch.zeros(n_bus, n_bus, dtype=torch.float32)
            Y_imag = torch.zeros(n_bus, n_bus, dtype=torch.float32)
            Y_real.fill_diagonal_(0.1)
            Y_imag.fill_diagonal_(-0.5)
            for i in range(n_bus - 1):
                Y_real[i, i + 1] = Y_real[i + 1, i] = -0.05
                Y_imag[i, i + 1] = Y_imag[i + 1, i] = 0.2

        if ('bus', 'ac_line', 'bus') in sample.edge_index_dict:
            n_lines = sample[('bus', 'ac_line', 'bus')].edge_index.shape[1]
        else:
            n_lines = max(n_bus - 1, 1)

        line_limits = torch.ones(n_lines, dtype=torch.float32) * 100.0
        base_mva = getattr(sample, 'base_mva', 100.0)

        manager = self.loss_managers[case_idx]
        if manager.lagrangian is not None:
            manager.lagrangian.set_network_parameters(
                Y_real=Y_real.to(self.device),
                Y_imag=Y_imag.to(self.device),
                line_limits=line_limits.to(self.device),
                base_mva=base_mva,
            )

        self.case_network_stats[case_idx] = {
            'n_bus': n_bus,
            'n_lines': n_lines,
            'base_mva': base_mva,
        }

    def _create_dummy_batch_data(self, batch, case_idx: int, predictions=None):
        """Create placeholder batch data for constraint evaluation when needed."""
        stats = self.case_network_stats.get(case_idx, {})

        if self.model_type in HETERO_MODEL_TYPES:
            bus_store = batch['bus']
            if hasattr(bus_store, 'batch') and bus_store.batch is not None:
                batch_size = int(bus_store.batch.max().item()) + 1
            else:
                batch_size = 1
            device = bus_store.x.device

            if predictions is not None and 'generator' in predictions:
                pred_shape = predictions['generator'].shape
                if len(pred_shape) == 2:
                    n_generators = pred_shape[0] // batch_size if batch_size else pred_shape[0]
                else:
                    n_generators = pred_shape[1]
            else:
                n_generators = batch['generator'].x.shape[0] if 'generator' in batch else 5
        else:
            batch_size = getattr(batch, 'num_graphs', 1)
            device = batch.x.device if hasattr(batch, 'x') else self.device
            if hasattr(batch, 'node_type'):
                n_generators = (batch.node_type == 1).sum().item()
            else:
                n_generators = stats.get('n_generators', 5)

        n_bus = stats.get('n_bus')
        if n_bus is None:
            if self.model_type in HETERO_MODEL_TYPES and 'bus' in batch:
                n_bus = batch['bus'].x.shape[0]
            elif hasattr(batch, 'num_nodes'):
                n_bus = batch.num_nodes
            else:
                n_bus = 10

        class ConstraintBatch:
            def __init__(self, device, n_generators, n_bus, base_mva):
                self.baseMVA = base_mva
                self.device = device
                self.n_generators = n_generators
                self.n_bus = n_bus

            def get(self, key, default=None):
                if key == 'pd':
                    return torch.rand(batch_size, self.n_bus, device=self.device) * 50
                if key == 'qd':
                    return torch.rand(batch_size, self.n_bus, device=self.device) * 25
                if key == 'gen_bus_indices':
                    return torch.tensor(list(range(self.n_generators)), device=self.device)
                if key == 'load_bus_indices':
                    return torch.tensor(list(range(2, min(self.n_bus, 7))), device=self.device)
                if key == 'line_edge_index':
                    from_buses = list(range(max(self.n_bus - 1, 1)))
                    to_buses = list(range(1, self.n_bus)) if self.n_bus > 1 else [0]
                    return torch.tensor([from_buses, to_buses], device=self.device)
                return default

        base_mva = stats.get('base_mva', 100.0)
        return ConstraintBatch(device, n_generators, n_bus, base_mva)

    def setup(self, stage: str | None = None) -> None:
        """Create train/val/test splits for each case independently."""
        self.train_subsets = {}
        self.val_subsets = {}
        self.test_subsets = {}

        train_ratio = self.config['train_split']
        val_ratio = self.config['val_split']

        generator = torch.Generator().manual_seed(self.config.get('split_seed', 42))

        for case_idx, dataset in enumerate(self.case_datasets):
            n_samples = len(dataset)
            if n_samples == 0:
                raise ValueError(f"Dataset for case {self.case_names[case_idx]} is empty.")

            train_len = max(1, int(n_samples * train_ratio))
            val_len = int(n_samples * val_ratio)
            if train_len + val_len >= n_samples:
                val_len = max(0, n_samples - train_len - 1)
            test_len = n_samples - train_len - val_len

            lengths = [train_len, val_len, test_len]
            subsets = torch.utils.data.random_split(dataset, lengths, generator=generator)

            self.train_subsets[case_idx] = subsets[0]
            self.val_subsets[case_idx] = subsets[1] if val_len > 0 else None
            self.test_subsets[case_idx] = subsets[2] if test_len > 0 else None

    def train_dataloader(self):
        loader_config = self.config['loader']
        batch_size = loader_config['batch_size']
        loaders: List[DataLoader] = []
        case_order: List[Tuple[str, int]] = []

        self._train_loader_map = {}
        for case_idx, subset in self.train_subsets.items():
            if subset is None or len(subset) == 0:
                continue

            case_key = self.case_keys[case_idx]
            loader = DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=loader_config['shuffle'],
                num_workers=loader_config['num_workers'],
            )
            loaders.append(loader)
            case_order.append((case_key, case_idx))
            self._train_loader_map[case_idx] = loader

        self._train_loaders = loaders
        self._train_case_indices = [idx for _, idx in case_order] or [0]

        if not loaders:
            raise ValueError("No training data available for any case.")

        if len(loaders) == 1:
            return loaders[0]

        return loaders

    def val_dataloader(self):
        loader_config = self.config['loader']
        batch_size = loader_config['batch_size']
        loaders: List[DataLoader] = []
        case_order: List[Tuple[str, int]] = []

        self._val_loader_map = {}
        for case_idx, subset in self.val_subsets.items():
            if subset is None or len(subset) == 0:
                continue

            case_key = self.case_keys[case_idx]
            loader = DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=loader_config['num_workers'],
            )
            loaders.append(loader)
            case_order.append((case_key, case_idx))
            self._val_loader_map[case_idx] = loader

        self._val_case_indices = [idx for _, idx in case_order] or [0]

        if not loaders:
            return None

        if len(loaders) == 1:
            return loaders[0]

        return loaders

    def test_dataloader(self):
        loader_config = self.config['loader']
        batch_size = loader_config['batch_size']
        loaders: List[DataLoader] = []
        case_order: List[Tuple[str, int]] = []

        self._test_loader_map = {}
        for case_idx, subset in self.test_subsets.items():
            if subset is None or len(subset) == 0:
                continue

            case_key = self.case_keys[case_idx]
            loader = DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=loader_config['num_workers'],
            )
            loaders.append(loader)
            case_order.append((case_key, case_idx))
            self._test_loader_map[case_idx] = loader

        self._test_case_indices = [idx for _, idx in case_order] or [0]

        if not loaders:
            return None

        if len(loaders) == 1:
            return loaders[0]

        return loaders

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), **self.config['optimizer']['Adam'])

    def forward(self, batch):
        if self.model_type in HETERO_MODEL_TYPES:
            x_dict = {k: v.float() for k, v in batch.x_dict.items()}
            return self.model(x_dict, batch.edge_index_dict)

        if isinstance(batch, torch.Tensor) or hasattr(batch, 'node_type'):
            homo_batch = batch
        else:
            homo_batch = convert_opf_to_homo(batch)
            homo_batch = homo_batch.to(self.device)

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

    def training_step(self, batch, batch_idx, dataloader_idx: int | None = None):
        case_idx, case_batch = self._resolve_case_batch(batch, 'train', dataloader_idx)
        case_key = self.case_keys.get(case_idx, f"case_{case_idx}")
        manager = self.loss_managers[case_idx]
        is_primary_loader = dataloader_idx in (None, 0)

        predictions = self(case_batch)

        if self.loss_type == 'augmented_lagrangian':
            constraint_batch = self._create_dummy_batch_data(case_batch, case_idx, predictions)
            loss, loss_info = manager.compute_loss(
                predictions,
                case_batch,
                return_info=True,
                constraint_data=constraint_batch,
            )
        else:
            loss, loss_info = manager.compute_loss(predictions, case_batch, return_info=True)

        batch_size = int(getattr(case_batch, 'num_graphs', 1))

        if is_primary_loader:
            self.log(
                'train_loss',
                loss,
                prog_bar=True,
                batch_size=batch_size,
                add_dataloader_idx=False,
                on_step=True,
                on_epoch=True,
            )
        self.log(
            f'train_loss/{case_key}',
            loss,
            prog_bar=False,
            batch_size=batch_size,
            add_dataloader_idx=False,
            on_step=True,
            on_epoch=True,
        )

        mse_loss = loss_info.get('mse_loss')
        if mse_loss is not None:
            if is_primary_loader:
                self.log(
                    'train_mse_loss',
                    mse_loss,
                    prog_bar=True,
                    batch_size=batch_size,
                    add_dataloader_idx=False,
                    on_step=True,
                    on_epoch=True,
                )
            self.log(
                f'train_mse_loss/{case_key}',
                mse_loss,
                prog_bar=False,
                batch_size=batch_size,
                add_dataloader_idx=False,
                on_step=True,
                on_epoch=True,
            )

        if self.loss_type in {'augmented_lagrangian', 'violated_lagrangian'}:
            violation = loss_info.get('constraint_violation')
            if violation is not None:
                violation_value = (
                    violation.detach().to('cpu').item()
                    if isinstance(violation, torch.Tensor)
                    else float(violation)
                )
                if is_primary_loader:
                    self.log(
                        'train_constraint_violation',
                        violation,
                        prog_bar=True,
                        batch_size=batch_size,
                        add_dataloader_idx=False,
                        on_step=True,
                        on_epoch=True,
                    )
                self.log(
                    f'train_constraint_violation/{case_key}',
                    violation,
                    prog_bar=False,
                    batch_size=batch_size,
                    add_dataloader_idx=False,
                    on_step=True,
                    on_epoch=True,
                )
                self.training_step_outputs.append(
                    {'case_idx': case_idx, 'constraint_violation': violation_value}
                )
            else:
                self.training_step_outputs.append({'case_idx': case_idx})
        else:
            self.training_step_outputs.append({'case_idx': case_idx})

        return loss

    def on_fit_start(self):
        # Ensure loss managers use the active device
        for manager in self.loss_managers.values():
            manager.device = self.device

        if self.loss_type == 'augmented_lagrangian':
            for case_idx in self.loss_managers.keys():
                self._prepare_case_network_parameters(case_idx)

    def on_train_epoch_end(self):
        if self.loss_type in {'augmented_lagrangian', 'violated_lagrangian'}:
            for case_idx, manager in self.loss_managers.items():
                case_entries = [
                    entry for entry in self.training_step_outputs if entry.get('case_idx') == case_idx
                ]
                if not case_entries:
                    continue

                loader = self._train_loader_map.get(case_idx)
                if loader is None:
                    continue

                if self.loss_type == 'augmented_lagrangian':
                    violations = [
                        torch.tensor(float(entry.get('constraint_violation', 0.0)), device=self.device)
                        for entry in case_entries
                    ]
                    avg_violation = torch.stack(violations).mean().item() if violations else 0.0
                    manager.update_lagrangian(self.model, loader, constraint_violation=avg_violation)
                else:
                    manager.update_lagrangian(self.model, loader)

                manager.step_epoch()
        else:
            for manager in self.loss_managers.values():
                manager.step_epoch()

        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx, dataloader_idx: int | None = None):
        case_idx, case_batch = self._resolve_case_batch(batch, 'val', dataloader_idx)
        case_key = self.case_keys.get(case_idx, f"case_{case_idx}")
        manager = self.loss_managers[case_idx]
        is_primary_loader = dataloader_idx in (None, 0)

        predictions = self(case_batch)

        if self.loss_type == 'augmented_lagrangian':
            constraint_batch = self._create_dummy_batch_data(case_batch, case_idx, predictions)
            loss, loss_info = manager.compute_loss(
                predictions,
                case_batch,
                return_info=True,
                constraint_data=constraint_batch,
            )
        else:
            loss, loss_info = manager.compute_loss(predictions, case_batch, return_info=True)

        batch_size = int(getattr(case_batch, 'num_graphs', 1))

        if is_primary_loader:
            self.log(
                'val_loss',
                loss,
                prog_bar=True,
                batch_size=batch_size,
                add_dataloader_idx=False,
            )
        self.log(
            f'val_loss/{case_key}',
            loss,
            prog_bar=False,
            batch_size=batch_size,
            add_dataloader_idx=False,
        )

        mse_loss = loss_info.get('mse_loss')
        if mse_loss is not None:
            if is_primary_loader:
                self.log(
                    'val_mse_loss',
                    mse_loss,
                    prog_bar=True,
                    batch_size=batch_size,
                    add_dataloader_idx=False,
                )
            self.log(
                f'val_mse_loss/{case_key}',
                mse_loss,
                prog_bar=False,
                batch_size=batch_size,
                add_dataloader_idx=False,
            )

        if self.loss_type in {'augmented_lagrangian', 'violated_lagrangian'}:
            violation = loss_info.get('constraint_violation')
            if violation is not None:
                if is_primary_loader:
                    self.log(
                        'val_constraint_violation',
                        violation,
                        prog_bar=True,
                        batch_size=batch_size,
                        add_dataloader_idx=False,
                    )
                self.log(
                    f'val_constraint_violation/{case_key}',
                    violation,
                    prog_bar=False,
                    batch_size=batch_size,
                    add_dataloader_idx=False,
                )

        return loss


def parse_cases_arg(cases_arg: Iterable[str]) -> List[str]:
    """Expand case CLI arguments which may include JSON lists or comma-separated values."""
    expanded: List[str] = []
    for entry in cases_arg:
        entry = entry.strip()
        if not entry:
            continue
        if entry.startswith('['):
            expanded.extend(json.loads(entry))
        elif ',' in entry:
            expanded.extend(x.strip() for x in entry.split(',') if x.strip())
        else:
            expanded.append(entry)
    return expanded


def main():
    parser = argparse.ArgumentParser(
        description='Multi-case OPF Training with PyTorch Lightning (standard and physics-informed losses)'
    )
    parser.add_argument(
        '--cases',
        type=str,
        nargs='+',
        required=True,
        help='List of case names (short e.g. case14 or full pglib names).',
    )
    parser.add_argument('--group_id', type=int, default=0, help='Group ID for dataset (default: 0)')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument(
        '--model_type',
        type=str,
        default='HeteroGNN',
        choices=['HeteroGNN', 'GCN', 'GAT', 'GIN', 'Transformer', 'RGAT', 'HEAT', 'HGT'],
        help='Model type to train (default: HeteroGNN)',
    )
    parser.add_argument(
        '--loss_type',
        type=str,
        default='mse',
        choices=['mse', 'rmse', 'mae', 'mape', 'smooth_l1', 'augmented_lagrangian', 'violated_lagrangian'],
        help='Loss function type (default: mse). Physics-informed options require per-case network data.',
    )
    parser.add_argument('--accelerator', type=str, default='auto', help='Accelerator type (default: auto)')
    parser.add_argument('--devices', type=int, default=1, help='Number of devices (default: 1)')
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes (default: 1)')
    parser.add_argument('--precision', type=str, default='32-true', help='Precision (default: 32-true)')
    parser.add_argument('--strategy', type=str, default='auto', help='Distributed strategy (default: auto)')

    args = parser.parse_args()

    print(f"🚀 ACOPF Multi-case Training (Lightning) with loss_type = {args.loss_type}")
    print("=" * 70)

    # Load configuration
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
                'single.yaml',
            )
            if os.path.exists(acopf_config):
                config_path = acopf_config

    print(f"Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load additional model configuration if available
    config_dir = Path(config_path).parent
    model_config_path = config_dir / 'model' / 'heterognn.yaml'
    if not model_config_path.exists():
        model_config_path = Path(__file__).parent.parent / 'configs' / 'model' / 'heterognn.yaml'

    if model_config_path.exists():
        print(f"Loading additional model config from: {model_config_path}")
        with open(model_config_path, 'r') as f:
            model_config = yaml.safe_load(f)
            if 'models' in model_config:
                config.setdefault('models', {})
                config['models'].update(model_config['models'])

    # Ensure minimal loader/split defaults
    config.setdefault('loader', {'batch_size': 32, 'shuffle': True, 'num_workers': 4})
    config.setdefault('train_split', 0.8)
    config.setdefault('val_split', 0.1)

    # Parse and normalize case names
    raw_cases = parse_cases_arg(args.cases)
    if not raw_cases:
        raise ValueError("No valid cases provided. Use --cases case14 case57 ...")

    case_names = [parse_case_name(case) for case in raw_cases]
    print(f"Training on cases: {case_names}")

    module = MultiCaseOPFLightningModule(
        config=config,
        case_names=case_names,
        group_id=args.group_id,
        model_type=args.model_type,
        loss_type=args.loss_type,
    )

    trainer_config = config.get('trainer', {})
    trainer_config['accelerator'] = args.accelerator
    trainer_config['devices'] = args.devices
    trainer_config['num_nodes'] = args.num_nodes
    trainer_config['precision'] = args.precision
    trainer_config['strategy'] = args.strategy
    trainer_config['sync_batchnorm'] = args.devices > 1

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='multi-case-best-{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        mode='min',
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=True,
        mode='min',
    )

    trainer = pl.Trainer(
        **trainer_config,
        callbacks=[checkpoint_callback, early_stop_callback],
    )

    trainer.fit(module)

    print("\n🎉 Training completed!")
    print(f"💾 Best model saved to: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()


