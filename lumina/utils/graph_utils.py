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

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import to_hetero
from torch_geometric.transforms import ToUndirected


class OPFHomoWrapper:
    """
    Simple wrapper that uses PyTorch Geometric's native to_homogeneous() method
    to convert OPF heterogeneous data for homogeneous GNN training.
    """

    def __init__(self,
                 add_node_type: bool = True,
                 add_edge_type: bool = True,
                 dummy_values: bool = True):
        """
        Args:
            add_node_type: Whether to add node type information to converted data
            add_edge_type: Whether to add edge type information to converted data
            dummy_values: Whether to fill missing attributes with dummy values
        """
        self.add_node_type = add_node_type
        self.add_edge_type = add_edge_type
        self.dummy_values = dummy_values

    def convert(self, hetero_data: HeteroData) -> Data:
        """
        Convert heterogeneous OPF data to homogeneous format using PyG's native method.

        Args:
            hetero_data: Input heterogeneous graph

        Returns:
            Homogeneous graph data object
        """
        homo_data = hetero_data.to_homogeneous(
            add_node_type=self.add_node_type,
            add_edge_type=self.add_edge_type,
            dummy_values=self.dummy_values
        )

        return homo_data


class OPFHeteroWrapper:
    """
    Wrapper that converts homogeneous GNN models to work with heterogeneous OPF data
    without using to_hetero() to avoid compatibility issues.
    """

    def __init__(self, model, metadata, aggr: str = 'sum'):
        """
        Args:
            model: Homogeneous GNN model to convert
            metadata: Graph metadata (node_types, edge_types)
            aggr: Aggregation method for combining embeddings from different relations
        """
        self.homo_model = model
        self.metadata = metadata
        self.aggr = aggr

        # Instead of using to_hetero(), we'll work with homogeneous converted data
        # This is more stable and avoids torch.fx issues

    def __call__(self, x_dict, edge_index_dict, edge_attr_dict=None):
        """
        Forward pass that converts hetero input to homo format, runs the model,
        and converts back to hetero format.
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
    """
    Dataset wrapper that converts OPF heterogeneous data to homogeneous on-the-fly.
    """

    def __init__(self, opf_dataset, add_node_type: bool = True, add_edge_type: bool = True):
        """
        Args:
            opf_dataset: Original OPFDataset instance
            add_node_type: Whether to include node type information
            add_edge_type: Whether to include edge type information
        """
        self.opf_dataset = opf_dataset
        self.converter = OPFHomoWrapper(
            add_node_type=add_node_type,
            add_edge_type=add_edge_type
        )

    def __len__(self):
        return len(self.opf_dataset)

    def __getitem__(self, idx):
        hetero_data = self.opf_dataset[idx]
        return self.converter.convert(hetero_data)

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
    """
    Prepare OPF dataset for training with proper train/val/test splits.

    Args:
        dataset: OPFDataset instance
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing
        use_homogeneous: Whether to convert to homogeneous format

    Returns:
        train_dataset, val_dataset, test_dataset
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
    """
    Extract metadata (node types, edge types) from OPF dataset for to_hetero conversion.

    Args:
        dataset: OPFDataset instance

    Returns:
        metadata: Tuple of (node_types, edge_types) for to_hetero()
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
    """
    Converts heterogeneous OPF graphs to homogeneous graphs.

    The conversion strategy:
    1. Create a unified node feature space by projecting all node types to the same dimension
    2. Add node type indicators to preserve type information
    3. Convert all edge types to a single edge type with edge type indicators
    4. Preserve target information for appropriate node/edge types
    """

    def __init__(self,
                 node_dim: int = 64,
                 edge_dim: int = 32,
                 use_node_type_embedding: bool = True,
                 use_edge_type_embedding: bool = True):
        """
        Args:
            node_dim: Target dimension for unified node features
            edge_dim: Target dimension for unified edge features
            use_node_type_embedding: Whether to add learnable node type embeddings
            use_edge_type_embedding: Whether to add learnable edge type embeddings
        """
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
        """
        Convert heterogeneous graph to homogeneous graph.

        Args:
            hetero_data: Input heterogeneous graph

        Returns:
            Homogeneous graph data object
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
    """
    Convenience function to convert OPF heterogeneous data to homogeneous format.

    Args:
        hetero_data: Input heterogeneous OPF data
        node_dim: Target node feature dimension
        edge_dim: Target edge feature dimension

    Returns:
        Homogeneous graph data
    """
    converter = HeteroToHomoConverter(node_dim=node_dim, edge_dim=edge_dim)
    return converter.convert(hetero_data)


def from_adj_to_edge_index_torch(adj):
    r""" Convert a dense adjacency matrix to a sparse edge index and edge attribute tensor.
    The edge attribute tensor is the non-zero values (weights) of the adjacency matrix.

    Args:
        adj (torch.Tensor): Dense adjacency matrix.

    Returns:
        edge_index (torch.Tensor): Sparse edge index tensor.
        edge_attr (torch.Tensor): Edge attribute tensor
    """
    adj_sparse = adj.to_sparse()
    edge_index = adj_sparse.indices().to(dtype=torch.long)
    edge_attr = adj_sparse.values()
    return edge_index, edge_attr
