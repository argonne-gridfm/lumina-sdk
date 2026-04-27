"""
Data handling utilities for training HomoGNN models on OPF heterogeneous data.

This module leverages PyTorch Geometric's native functionality:
1. `data.to_homogeneous()` - converts heterogeneous graphs to homogeneous
2. `to_hetero(model, metadata)` - wraps homogeneous models to work on hetero data

Reference: PyTorch Geometric documentation and examples.

Copyright (c) 2025 by Argonne National Laboratory.
All rights reserved.
"""

from typing import Dict, List, Optional, Tuple
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import to_hetero
from torch_geometric.transforms import ToUndirected


class OPFHomoWrapper:
    """Wrapper using PyG's ``to_homogeneous()`` for OPF hetero-to-homo conversion.

    Converts ``HeteroData`` objects into ``Data`` objects suitable for
    homogeneous GNN training, preserving node/edge type indicators and
    optionally storing padded edge attributes.

    Args:
        add_node_type (bool): Whether to add ``node_type`` tensor to the
            converted data.
        add_edge_type (bool): Whether to add ``edge_type`` tensor to the
            converted data.
        dummy_values (bool): Whether to fill missing attributes with zeros.
        attach_full_edge_attr (bool): Whether to compute and store
            ``edge_attr_full`` (all edge attributes padded to a uniform
            width).
    """

    def __init__(self,
                 add_node_type: bool = True,
                 add_edge_type: bool = True,
                 dummy_values: bool = True,
                 attach_full_edge_attr: bool = False):
        self.add_node_type = add_node_type
        self.add_edge_type = add_edge_type
        self.dummy_values = dummy_values
        self.attach_full_edge_attr = attach_full_edge_attr

    def convert(self, hetero_data: HeteroData) -> Data:
        """Convert heterogeneous OPF data to homogeneous format via PyG.

        Args:
            hetero_data (HeteroData): Input heterogeneous graph.

        Returns:
            Data: Homogeneous graph with ``node_type_names`` and
                ``edge_type_names`` attributes attached.
        """
        homo_data = hetero_data.to_homogeneous(
            add_node_type=self.add_node_type,
            add_edge_type=self.add_edge_type,
            dummy_values=self.dummy_values
        )
        homo_data.node_type_names = list(getattr(hetero_data, "node_types", []))
        homo_data.edge_type_names = [
            f"{src}::{rel}::{dst}" for (src, rel, dst) in getattr(hetero_data, "edge_types", [])
        ]
        if self.attach_full_edge_attr:
            self._attach_full_edge_attr(homo_data, hetero_data)

        return homo_data

    def _attach_full_edge_attr(self, homo_data: Data, hetero_data: HeteroData):
        edge_attr_list = []
        max_dim = 0
        edge_types = list(getattr(hetero_data, "edge_types", []))
        if not edge_types:
            return

        for edge_type in edge_types:
            edge_data = hetero_data[edge_type]
            edge_index = getattr(edge_data, "edge_index", None)
            if edge_index is None:
                continue
            num_edges = edge_index.size(1)
            if num_edges == 0:
                continue
            edge_attr = getattr(edge_data, "edge_attr", None)
            if torch.is_tensor(edge_attr):
                dim = int(edge_attr.size(1)) if edge_attr.dim() > 1 else 1
                max_dim = max(max_dim, dim)
            else:
                max_dim = max(max_dim, 0)

        if max_dim == 0:
            return

        for edge_type in edge_types:
            edge_data = hetero_data[edge_type]
            edge_index = getattr(edge_data, "edge_index", None)
            if edge_index is None:
                continue
            num_edges = edge_index.size(1)
            if num_edges == 0:
                continue
            edge_attr = getattr(edge_data, "edge_attr", None)
            if torch.is_tensor(edge_attr):
                if edge_attr.dim() == 1:
                    edge_attr = edge_attr.view(-1, 1)
                if edge_attr.size(1) < max_dim:
                    padding = torch.zeros(
                        edge_attr.size(0),
                        max_dim - edge_attr.size(1),
                        dtype=edge_attr.dtype,
                        device=edge_attr.device,
                    )
                    edge_attr = torch.cat([edge_attr, padding], dim=1)
                elif edge_attr.size(1) > max_dim:
                    edge_attr = edge_attr[:, :max_dim]
            else:
                edge_attr = torch.zeros(
                    num_edges,
                    max_dim,
                    dtype=torch.float32,
                    device=edge_index.device,
                )
            edge_attr_list.append(edge_attr)

        if edge_attr_list:
            homo_data.edge_attr_full = torch.cat(edge_attr_list, dim=0)


class OPFHeteroWrapper:
    """Wrapper that runs a homogeneous GNN on heterogeneous OPF inputs.

    Instead of using ``to_hetero()`` (which can cause ``torch.fx`` issues),
    this wrapper converts heterogeneous inputs to a simplified homogeneous
    format, runs the underlying model, and returns outputs keyed by node
    type.

    Args:
        model (torch.nn.Module): Homogeneous GNN model instance.
        metadata (tuple): Graph metadata ``(node_types, edge_types)``.
        aggr (str): Aggregation method for combining embeddings from
            different relations (currently unused; reserved for future use).
    """

    def __init__(self, model, metadata, aggr: str = 'sum'):
        self.homo_model = model
        self.metadata = metadata
        self.aggr = aggr

        # Instead of using to_hetero(), we'll work with homogeneous converted data
        # This is more stable and avoids torch.fx issues

    def __call__(self, x_dict, edge_index_dict, edge_attr_dict=None):
        """Forward pass: converts hetero inputs to homo, runs model, returns per-type dict.

        Args:
            x_dict (dict[str, torch.Tensor]): Per-node-type feature tensors.
            edge_index_dict (dict[tuple, torch.Tensor]): Per-edge-type
                connectivity tensors.
            edge_attr_dict (dict[tuple, torch.Tensor], optional): Per-edge-type
                attribute tensors.

        Returns:
            dict[str, torch.Tensor]: Model output keyed by the primary node type.
        """
        # Convert heterogeneous input to homogeneous format
        # This is a simplified version that works with OPF data structure

        # Get the main node type data (typically 'bus' nodes)
        if 'bus' in x_dict:
            x = x_dict['bus']
        else:
            # Fallback to first available node type
            first_node_type = list(x_dict.keys())[0]
            x = x_dict[first_node_type]

        # Get the main edge type data
        edge_index = None
        edge_attr = None

        for edge_type, ei in edge_index_dict.items():
            if ei.size(1) > 0:  # Use first non-empty edge type
                edge_index = ei
                if edge_attr_dict and edge_type in edge_attr_dict:
                    edge_attr = edge_attr_dict[edge_type]
                break

        # If no valid edges found, create a minimal edge structure
        if edge_index is None or edge_index.size(1) == 0:
            # Create self-loops for all nodes
            num_nodes = x.size(0)
            edge_index = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)], dim=0)
            edge_attr = torch.zeros(num_nodes, 1)  # Minimal edge attributes

        # Run the homogeneous model
        if hasattr(self.homo_model, 'return_node_embeddings'):
            output = self.homo_model(x, edge_index, edge_attr, return_node_embeddings=True)
        else:
            output = self.homo_model(x, edge_index, edge_attr)

        # Return output in dictionary format for compatibility
        # Return output for the main node type
        if 'bus' in x_dict:
            return {'bus': output}
        else:
            first_node_type = list(x_dict.keys())[0]
            return {first_node_type: output}

    def parameters(self):
        """Get model parameters."""
        return self.homo_model.parameters()

    def train(self):
        """Set model to training mode."""
        self.homo_model.train()
        return self

    def eval(self):
        """Set model to evaluation mode."""
        self.homo_model.eval()
        return self

    def to(self, device):
        """Move model to device."""
        self.homo_model = self.homo_model.to(device)
        return self


class HomoOPFDataset:
    """Dataset wrapper that converts OPF hetero graphs to homogeneous on-the-fly.

    Wraps an existing OPF dataset and applies ``OPFHomoWrapper.convert()``
    to each sample at access time. Non-finite targets are optionally
    sanitized (replaced with zeros) and flagged via a ``y_mask`` attribute.

    Args:
        opf_dataset: Original OPFDataset or ``Subset`` instance.
        add_node_type (bool): Whether to include node type information.
        add_edge_type (bool): Whether to include edge type information.
        sanitize_targets (bool): If ``True``, replace non-finite target values
            with zeros and attach a ``y_mask`` boolean tensor.
        log_bad_targets (bool): If ``True``, print a warning when non-finite
            targets are detected (rank-0 only).
        max_bad_target_logs (int): Maximum number of bad-target warnings to
            emit per dataset instance.
    """

    def __init__(
        self,
        opf_dataset,
        add_node_type: bool = True,
        add_edge_type: bool = True,
        sanitize_targets: bool = True,
        log_bad_targets: bool = True,
        max_bad_target_logs: int = 1,
    ):
        self.opf_dataset = opf_dataset
        self.converter = OPFHomoWrapper(
            add_node_type=add_node_type,
            add_edge_type=add_edge_type
        )
        self.sanitize_targets = sanitize_targets
        self.log_bad_targets = log_bad_targets
        self.max_bad_target_logs = max_bad_target_logs
        self._bad_target_logs = 0

    def __len__(self):
        return len(self.opf_dataset)

    def __getitem__(self, idx):
        hetero_data = self.opf_dataset[idx]
        homo_data = self.converter.convert(hetero_data)
        self._sanitize_targets(homo_data, idx)
        return homo_data

    def _sanitize_targets(self, homo_data, idx):
        y = getattr(homo_data, "y", None)
        if not torch.is_tensor(y):
            return

        finite_mask = torch.isfinite(y)
        if finite_mask.ndim > 1:
            row_mask = finite_mask.all(dim=-1)
        else:
            row_mask = finite_mask

        if bool(row_mask.all().item()):
            return

        if self.sanitize_targets:
            if y.ndim == 0:
                y = torch.zeros_like(y)
            else:
                y = y.clone()
                y[~row_mask] = 0
            homo_data.y = y

        homo_data.y_mask = row_mask.to(dtype=torch.bool)

        if self._should_log_bad_targets():
            bad_count = int((~row_mask).sum().item())
            total = int(row_mask.numel())
            action = "sanitized" if self.sanitize_targets else "left as-is"
            print(
                f"[HomoOPFDataset] Non-finite targets in sample {idx}: "
                f"{bad_count}/{total} rows {action}; stored y_mask."
            )
            self._bad_target_logs += 1

    def _should_log_bad_targets(self):
        if not self.log_bad_targets:
            return False
        if self._bad_target_logs >= self.max_bad_target_logs:
            return False
        try:
            rank = int(os.environ.get("RANK", "0"))
        except ValueError:
            rank = 0
        return rank == 0

    @property
    def num_node_features(self):
        # Get a sample to determine feature dimensions
        sample = self[0]
        return sample.x.size(1) if hasattr(sample, 'x') else 0

    @property
    def num_edge_features(self):
        # Get a sample to determine edge feature dimensions
        sample = self[0]
        return sample.edge_attr.size(1) if hasattr(sample, 'edge_attr') else 0

    @property
    def num_node_types(self):
        # Get a sample to determine number of node types
        sample = self[0]
        return int(sample.node_type.max().item()) + 1 if hasattr(sample, 'node_type') else 1

    @property
    def num_edge_types(self):
        # Get a sample to determine number of edge types
        sample = self[0]
        return int(sample.edge_type.max().item()) + 1 if hasattr(sample, 'edge_type') else 1


def prepare_opf_training_data(dataset,
                              train_ratio: float = 0.8,
                              val_ratio: float = 0.1,
                              test_ratio: float = 0.1,
                              use_homogeneous: bool = True):
    """Split an OPF dataset into train/val/test subsets.

    Creates random index-based subsets and optionally wraps each in
    ``HomoOPFDataset`` for homogeneous model training.

    Args:
        dataset: OPFDataset instance with ``__len__`` support.
        train_ratio (float): Fraction of data for training.
        val_ratio (float): Fraction of data for validation.
        test_ratio (float): Fraction of data for testing.
        use_homogeneous (bool): If ``True``, wrap each subset in
            ``HomoOPFDataset`` for on-the-fly hetero-to-homo conversion.

    Returns:
        tuple: ``(train_dataset, val_dataset, test_dataset)`` subsets.

    Raises:
        AssertionError: If ratios do not sum to 1.0.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    # Create random indices for splitting
    indices = torch.randperm(total_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Create subset datasets
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)
    test_subset = torch.utils.data.Subset(dataset, test_indices)

    if use_homogeneous:
        # Wrap with homogeneous conversion
        train_dataset = HomoOPFDataset(train_subset)
        val_dataset = HomoOPFDataset(val_subset)
        test_dataset = HomoOPFDataset(test_subset)
    else:
        train_dataset = train_subset
        val_dataset = val_subset
        test_dataset = test_subset

    return train_dataset, val_dataset, test_dataset


def get_opf_metadata(dataset):
    """Extract graph metadata from an OPF dataset.

    Retrieves the ``(node_types, edge_types)`` tuple needed by
    ``to_hetero()`` and other heterogeneous model utilities.

    Args:
        dataset: OPFDataset instance supporting indexing.

    Returns:
        tuple: ``(node_types, edge_types)`` where each element is a list
            of strings or tuples respectively.
    """
    # Get a sample to extract metadata
    sample = dataset[0]
    if hasattr(sample, 'metadata'):
        return sample.metadata()
    else:
        # Manually extract from the heterogeneous data structure
        node_types = list(sample.node_types)
        edge_types = list(sample.edge_types)
        return (node_types, edge_types)


class HeteroToHomoConverter:
    """Converts heterogeneous OPF graphs to homogeneous format with feature projection.

    Unlike ``OPFHomoWrapper`` (which delegates to PyG's ``to_homogeneous``),
    this class manually projects node and edge features to fixed-width
    unified feature spaces, builds global node/edge indices, and preserves
    type indicators.

    Conversion strategy:
        1. Project or pad/truncate all node features to ``node_dim``.
        2. Add integer ``node_type`` indicators.
        3. Convert all edge types to a single type with ``edge_type`` indicators.
        4. Preserve targets for appropriate node/edge types.

    Args:
        node_dim (int): Target dimension for unified node features.
        edge_dim (int): Target dimension for unified edge features.
        use_node_type_embedding (bool): Whether to add learnable node type
            embeddings (reserved for future use).
        use_edge_type_embedding (bool): Whether to add learnable edge type
            embeddings (reserved for future use).
    """

    def __init__(self,
                 node_dim: int = 64,
                 edge_dim: int = 32,
                 use_node_type_embedding: bool = True,
                 use_edge_type_embedding: bool = True):
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.use_node_type_embedding = use_node_type_embedding
        self.use_edge_type_embedding = use_edge_type_embedding

        # Node type mapping
        self.node_types = ['bus', 'generator', 'load', 'shunt']
        self.node_type_to_id = {nt: i for i, nt in enumerate(self.node_types)}

        # Edge type mapping
        self.edge_types = ['ac_line', 'transformer', 'generator_link', 'load_link', 'shunt_link']
        self.edge_type_to_id = {et: i for i, et in enumerate(self.edge_types)}

        self.num_node_types = len(self.node_types)
        self.num_edge_types = len(self.edge_types)

    def convert(self, hetero_data: HeteroData) -> Data:
        """Convert a heterogeneous graph to a homogeneous graph.

        Args:
            hetero_data (HeteroData): Input heterogeneous graph with
                per-type node/edge features and targets.

        Returns:
            Data: Homogeneous graph with unified ``x``, ``edge_index``,
                ``edge_attr``, ``node_type``, ``edge_type``, and optional
                ``y`` / ``edge_label`` attributes.
        """
        # Step 1: Build unified node features and mapping
        node_features, node_types, node_targets, node_mapping = self._unify_nodes(hetero_data)

        # Step 2: Build unified edge features and indices
        edge_index, edge_attr, edge_types, edge_targets = self._unify_edges(hetero_data, node_mapping)

        # Step 3: Create homogeneous data object
        homo_data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_type=node_types,
            edge_type=edge_types,
        )

        # Add targets if they exist
        if node_targets is not None:
            homo_data.y = node_targets
        if edge_targets is not None:
            homo_data.edge_label = edge_targets

        # Add graph-level properties
        if hasattr(hetero_data, 'baseMVA'):
            homo_data.baseMVA = hetero_data.baseMVA
        if hasattr(hetero_data, 'objective'):
            homo_data.objective = hetero_data.objective

        return homo_data

    def _unify_nodes(self, hetero_data: HeteroData) -> Tuple[torch.Tensor,
                                                             torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Unify all node types into a single node feature matrix.

        Returns:
            node_features: Unified node feature matrix [total_nodes, node_dim]
            node_types: Node type indicators [total_nodes]
            node_targets: Unified node targets [total_nodes, target_dim] or None
            node_mapping: Mapping from hetero node indices to homo node indices
        """
        all_node_features = []
        all_node_types = []
        all_node_targets = []
        node_mapping = {}
        current_node_idx = 0

        for node_type in self.node_types:
            if node_type not in hetero_data.node_types:
                continue

            node_data = hetero_data[node_type]
            num_nodes = node_data.x.size(0)

            # Get original features
            node_x = node_data.x  # [num_nodes, original_dim]

            # Project to target dimension or pad/truncate
            if node_x.size(1) < self.node_dim:
                # Pad with zeros
                padding = torch.zeros(num_nodes, self.node_dim - node_x.size(1),
                                      dtype=node_x.dtype, device=node_x.device)
                unified_features = torch.cat([node_x, padding], dim=1)
            elif node_x.size(1) > self.node_dim:
                # Use linear projection (this would need to be learned in practice)
                unified_features = node_x[:, :self.node_dim]
            else:
                unified_features = node_x

            all_node_features.append(unified_features)

            # Node type indicators
            node_type_id = self.node_type_to_id[node_type]
            node_type_tensor = torch.full((num_nodes,), node_type_id,
                                          dtype=torch.long, device=node_x.device)
            all_node_types.append(node_type_tensor)

            # Node targets (if they exist)
            if hasattr(node_data, 'y') and node_data.y is not None:
                all_node_targets.append(node_data.y)
            else:
                # Add dummy targets for consistency
                dummy_targets = torch.zeros(num_nodes, 2, dtype=torch.float32, device=node_x.device)
                all_node_targets.append(dummy_targets)

            # Build mapping
            node_indices = torch.arange(current_node_idx, current_node_idx + num_nodes,
                                        dtype=torch.long, device=node_x.device)
            node_mapping[node_type] = node_indices
            current_node_idx += num_nodes

        # Concatenate all features
        node_features = torch.cat(all_node_features, dim=0)
        node_types = torch.cat(all_node_types, dim=0)

        # Handle targets
        if all_node_targets and any(target.numel() > 0 for target in all_node_targets):
            # Find max target dimension
            max_target_dim = max(target.size(1) for target in all_node_targets if target.numel() > 0)

            # Pad all targets to the same dimension
            padded_targets = []
            for target in all_node_targets:
                if target.size(1) < max_target_dim:
                    padding = torch.zeros(target.size(0), max_target_dim - target.size(1),
                                          dtype=target.dtype, device=target.device)
                    padded_target = torch.cat([target, padding], dim=1)
                else:
                    padded_target = target[:, :max_target_dim]
                padded_targets.append(padded_target)

            node_targets = torch.cat(padded_targets, dim=0)
        else:
            node_targets = None

        return node_features, node_types, node_targets, node_mapping

    def _unify_edges(self,
                     hetero_data: HeteroData,
                     node_mapping: Dict[str,
                                        torch.Tensor]) -> Tuple[torch.Tensor,
                                                                torch.Tensor,
                                                                torch.Tensor,
                                                                Optional[torch.Tensor]]:
        """
        Unify all edge types into homogeneous edge representation.

        Returns:
            edge_index: Unified edge indices [2, total_edges]
            edge_attr: Unified edge features [total_edges, edge_dim]
            edge_types: Edge type indicators [total_edges]
            edge_targets: Unified edge targets [total_edges, target_dim] or None
        """
        all_edge_indices = []
        all_edge_attrs = []
        all_edge_types = []
        all_edge_targets = []

        # Process edge types that correspond to actual connections
        for edge_type_tuple in hetero_data.edge_types:
            src_type, edge_type, dst_type = edge_type_tuple

            if edge_type not in self.edge_types:
                continue

            edge_data = hetero_data[edge_type_tuple]
            if not hasattr(edge_data, 'edge_index'):
                continue

            edge_index = edge_data.edge_index  # [2, num_edges]
            num_edges = edge_index.size(1)

            if num_edges == 0:
                continue

            # Map to homogeneous node indices
            src_mapping = node_mapping.get(src_type, None)
            dst_mapping = node_mapping.get(dst_type, None)

            if src_mapping is None or dst_mapping is None:
                continue

            # Convert edge indices
            homo_edge_index = torch.zeros_like(edge_index)
            homo_edge_index[0] = src_mapping[edge_index[0]]
            homo_edge_index[1] = dst_mapping[edge_index[1]]
            all_edge_indices.append(homo_edge_index)

            # Process edge attributes
            if hasattr(edge_data, 'edge_attr') and edge_data.edge_attr is not None:
                edge_attr = edge_data.edge_attr

                # Project to target dimension or pad/truncate
                if edge_attr.size(1) < self.edge_dim:
                    # Pad with zeros
                    padding = torch.zeros(num_edges, self.edge_dim - edge_attr.size(1),
                                          dtype=edge_attr.dtype, device=edge_attr.device)
                    unified_edge_attr = torch.cat([edge_attr, padding], dim=1)
                elif edge_attr.size(1) > self.edge_dim:
                    # Truncate (in practice, use learned projection)
                    unified_edge_attr = edge_attr[:, :self.edge_dim]
                else:
                    unified_edge_attr = edge_attr
            else:
                # Create dummy edge features
                unified_edge_attr = torch.zeros(num_edges, self.edge_dim,
                                                dtype=torch.float32, device=edge_index.device)

            all_edge_attrs.append(unified_edge_attr)

            # Edge type indicators
            edge_type_id = self.edge_type_to_id[edge_type]
            edge_type_tensor = torch.full((num_edges,), edge_type_id,
                                          dtype=torch.long, device=edge_index.device)
            all_edge_types.append(edge_type_tensor)

            # Edge targets (if they exist)
            if hasattr(edge_data, 'edge_label') and edge_data.edge_label is not None:
                all_edge_targets.append(edge_data.edge_label)
            else:
                # Add dummy targets for consistency
                dummy_targets = torch.zeros(num_edges, 4, dtype=torch.float32, device=edge_index.device)
                all_edge_targets.append(dummy_targets)

        # Concatenate all edges
        if all_edge_indices:
            edge_index = torch.cat(all_edge_indices, dim=1)
            edge_attr = torch.cat(all_edge_attrs, dim=0)
            edge_types = torch.cat(all_edge_types, dim=0)

            # Handle edge targets
            if all_edge_targets and any(target.numel() > 0 for target in all_edge_targets):
                # Find max target dimension
                max_target_dim = max(target.size(1) for target in all_edge_targets if target.numel() > 0)

                # Pad all targets to the same dimension
                padded_targets = []
                for target in all_edge_targets:
                    if target.size(1) < max_target_dim:
                        padding = torch.zeros(target.size(0), max_target_dim - target.size(1),
                                              dtype=target.dtype, device=target.device)
                        padded_target = torch.cat([target, padding], dim=1)
                    else:
                        padded_target = target[:, :max_target_dim]
                    padded_targets.append(padded_target)

                edge_targets = torch.cat(padded_targets, dim=0)
            else:
                edge_targets = None
        else:
            # No edges found
            device = next(iter(node_mapping.values())).device
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.empty((0, self.edge_dim), dtype=torch.float32, device=device)
            edge_types = torch.empty((0,), dtype=torch.long, device=device)
            edge_targets = None

        return edge_index, edge_attr, edge_types, edge_targets


def convert_opf_to_homo(hetero_data: HeteroData,
                        node_dim: int = 64,
                        edge_dim: int = 32) -> Data:
    """Convert OPF heterogeneous data to homogeneous format.

    Convenience function that creates a ``HeteroToHomoConverter`` and calls
    its ``convert`` method.

    Args:
        hetero_data (HeteroData): Input heterogeneous OPF graph.
        node_dim (int): Target unified node feature dimension.
        edge_dim (int): Target unified edge feature dimension.

    Returns:
        Data: Homogeneous graph data object.
    """
    converter = HeteroToHomoConverter(node_dim=node_dim, edge_dim=edge_dim)
    return converter.convert(hetero_data)


def from_adj_to_edge_index_torch(adj):
    """Convert a dense adjacency matrix to sparse edge index and edge attributes.

    Non-zero entries of the adjacency matrix become edges; their values
    become edge attributes.

    Args:
        adj (torch.Tensor): Dense adjacency matrix of shape ``[N, N]``.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(edge_index, edge_attr)`` where
            ``edge_index`` has shape ``[2, E]`` (``torch.long``) and
            ``edge_attr`` has shape ``[E]``.
    """
    adj_sparse = adj.to_sparse()
    edge_index = adj_sparse.indices().to(dtype=torch.long)
    edge_attr = adj_sparse.values()
    return edge_index, edge_attr
