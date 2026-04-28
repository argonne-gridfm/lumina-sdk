"""
Copyright (c) 2025, Argonne National Laboratory
All rights reserved.
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import (MLP, GATConv, GCNConv, GINConv, GraphConv,
                                HeteroConv, HGTConv, Linear, RGATConv,
                                SAGEConv)


class HeteroGNN(torch.nn.Module):
    """Generic Heterogeneous GNN with configurable message-passing backend.

    Supports SAGE, GCN, GIN, and GAT backends via ``HeteroConv``. Each edge
    type gets its own convolution instance. Produces per-node predictions for
    ``bus`` and ``generator`` node types.

    Input shape per node type: ``(N_type, input_channels[type])``.
    Output shape: ``{'bus': (N_bus, out_channels), 'generator': (N_gen, out_channels)}``.
    """

    def __init__(
            self,
            metadata,
            input_channels,
            hidden_channels=64,
            out_channels=2,
            num_layers=3,
            backend="sage",
            edge_attr_dim=None,
            **kwargs):
        """ Heterogeneous Graph Neural Network (HeteroGNN) model.

        Args:
            metadata (dict or tuple): Metadata containing node types and edge types.
                If dict: {'nodes': {node_type: dim, ...}, 'edges': {edge_type: dim, ...}}
                If tuple: (node_types, edge_types)
            input_channels (dict): Number of input features for each node type.
            hidden_channels (int): Hidden embedding size.
            out_channels (int): Size of each output sample. Defaults to 2.
            num_layers (int): Number of layers. Defaults to 3.
            backend (str): Graph convolutional layer backend. Defaults to "sage".
            edge_attr_dim (int, optional): Dimension of edge attributes for GAT. Defaults to None.
        """
        super().__init__()
        self.lin_dict = torch.nn.ModuleDict()

        # Handle both old tuple format and new dict format
        if isinstance(metadata, dict):
            node_types = list(metadata['nodes'].keys())
            edge_types = list(metadata['edges'].keys())
        else:
            # Legacy tuple format: (node_types, edge_types)
            node_types = metadata[0]
            edge_types = metadata[1]

        self.backend = backend
        self.edge_attr_support = backend == "gat"

        # Validate input_channels
        if not isinstance(input_channels, dict):
            raise ValueError("input_channels must be a dictionary")

        # Input layers for each node type - no more lazy initialization
        for node_type in node_types:
            if node_type not in input_channels:
                raise ValueError(f"input_channels must contain entry for node type '{node_type}'")
            self.lin_dict[node_type] = Linear(input_channels[node_type], hidden_channels)

        # Heterogeneous graph convolutional layers for edges
        self.convs = torch.nn.ModuleList()

        # Get edge attribute dimensions from metadata
        if isinstance(metadata, dict):
            edge_attr_dims = {edge_type: (dim if dim > 0 else None)
                              for edge_type, dim in metadata['edges'].items()}
        else:
            # Fallback to hardcoded dimensions for legacy format
            edge_attr_dims = {
                ('bus', 'ac_line', 'bus'): 9,
                ('bus', 'transformer', 'bus'): 11,
                # Link edges have no attributes
                ('generator', 'generator_link', 'bus'): None,
                ('bus', 'generator_link', 'generator'): None,
                ('load', 'load_link', 'bus'): None,
                ('bus', 'load_link', 'load'): None,
                ('shunt', 'shunt_link', 'bus'): None,
                ('bus', 'shunt_link', 'shunt'): None,
            }

        def get_conv_layer(edge_type):
            if backend == "sage":
                return SAGEConv((hidden_channels, hidden_channels), hidden_channels)
            elif backend == "gcn":
                return GraphConv((hidden_channels, hidden_channels), hidden_channels)
            elif backend == "gin":
                return GINConv(MLP([hidden_channels, hidden_channels]))
            elif backend == "gat":
                edge_dim = edge_attr_dims.get(edge_type, None)
                return GATConv((hidden_channels, hidden_channels),
                               hidden_channels,
                               add_self_loops=False,
                               edge_dim=edge_dim)
            else:
                raise ValueError(f"Unknown backend: {backend}")

        for _ in range(num_layers - 1):
            conv = HeteroConv({
                edge_type: get_conv_layer(edge_type)
                for edge_type in edge_types
            }, aggr='sum')
            self.convs.append(conv)

        # Output layers for target node types, ACOPF variables
        self.out_dict = torch.nn.ModuleDict({
            "bus": Linear(hidden_channels, out_channels),
            "generator": Linear(hidden_channels, out_channels),
        })

        self.reset_parameters()

        # # set param init
        # for lin in self.lin_dict.values():
        #     torch.nn.init.xavier_uniform_(lin.weight)
        # # for conv in self.convs:
        # #     for rel_conv in conv.convs.values():
        # #         torch.nn.init.xavier_normal_(rel_conv.lin_rel.weight)
        # #         torch.nn.init.xavier_normal_(rel_conv.lin_root.weight)
        # for out in self.out_dict.values():
        #     torch.nn.init.xavier_uniform_(out.weight)

    def reset_parameters(self):
        """Reset parameters of the model."""
        for lin in self.lin_dict.values():
            lin.reset_parameters()
        for conv in self.convs:
            for rel_conv in conv.convs.values():
                if hasattr(rel_conv, 'reset_parameters'):
                    rel_conv.reset_parameters()
            conv.reset_parameters()
        for out in self.out_dict.values():
            out.reset_parameters()

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None, minmax_scaling=False, **kwargs):
        """Forward pass of the HeteroGNN model.

        Args:
            x_dict (dict): Node features for each node type.
            edge_index_dict (dict): Edge indices for each edge type.
            edge_attr_dict (dict, optional): Edge attributes for each edge type. Defaults to None
            minmax_scaling (bool): Whether to apply min-max scaling to outputs. Defaults to False.

        Returns:
            dict: Output predictions for each target node type.
        """

        if minmax_scaling:
            _vmin = x_dict['bus'][:, 1].clone()  # Original voltage min
            _vmax = x_dict['bus'][:, 2].clone()  # Original voltage max
            _pmin = x_dict['generator'][:, 2].clone()  # Original active power min
            _pmax = x_dict['generator'][:, 3].clone()  # Original active power max
            _qmin = x_dict['generator'][:, 5].clone()  # Original reactive power min
            _qmax = x_dict['generator'][:, 6].clone()  # Original reactive power max

        # Transform input features
        x_dict = {
            node_type: F.relu(self.lin_dict[node_type](x))
            for node_type, x in x_dict.items()
        }
        # x_dict = {key: F.dropout(x, p=0.1, training=self.training) for key, x in x_dict.items()}

        # Message passing
        for conv in self.convs:
            if self.edge_attr_support and edge_attr_dict is not None:
                x_dict = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
                x_dict = {key: F.relu(x) for key, x in x_dict.items()}
                x_dict = {key: F.dropout(x, p=0.1, training=self.training) for key, x in x_dict.items()}
            else:
                x_dict = conv(x_dict, edge_index_dict)
                x_dict = {key: F.relu(x) for key, x in x_dict.items()}
                x_dict = {key: F.dropout(x, p=0.1, training=self.training) for key, x in x_dict.items()}
            # NOTE: no activation function applied here <== why?
            # x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        # Final predictions
        # bus_out: va, vm
        bus_out = self.out_dict["bus"](x_dict["bus"])
        # bus_out = F.dropout(bus_out, p=0.1, training=self.training)
        # gen_out: pg, qg
        gen_out = self.out_dict["generator"](x_dict["generator"])
        # gen_out = F.dropout(gen_out, p=0.1, training=self.training)

        if minmax_scaling:
            # Create new tensors instead of modifying in-place
            bus_out_final = bus_out.clone()
            gen_out_final = gen_out.clone()

            # Apply scaling without in-place operations
            bus_out_final[:, 1] = F.sigmoid(bus_out[:, 1]) * (_vmax - _vmin) + _vmin

            gen_out_sigmoid = F.sigmoid(gen_out)
            gen_out_final[:, 0] = gen_out_sigmoid[:, 0] * (_pmax - _pmin) + _pmin
            gen_out_final[:, 1] = gen_out_sigmoid[:, 1] * (_qmax - _qmin) + _qmin

            return {"bus": bus_out_final, "generator": gen_out_final}
        else:
            return {"bus": bus_out, "generator": gen_out}


class RGAT(torch.nn.Module):
    """Relational Graph Attention Network using ``RGATConv``.

    Applies relation-aware graph attention with shared hidden channels
    across node types. Produces predictions for ``bus`` and ``generator``
    node types.

    Input shape per node type: ``(N_type, input_channels[type])``.
    Output shape: ``{'bus': (N_bus, out_channels), 'generator': (N_gen, out_channels)}``.
    """

    def __init__(self,
                 metadata,
                 input_channels,
                 hidden_channels=64,
                 out_channels=2,
                 num_layers=3,
                 num_heads=1,
                 backend="sage",
                 edge_attr_dim=None,
                 **kwargs):
        r""" Relational Graph Attention Network (RGAT) model.

        Args:
            metadata (dict or tuple): Metadata containing node types and edge types.
                If dict: {'nodes': {node_type: dim, ...}, 'edges': {edge_type: dim, ...}}
                If tuple: (node_types, edge_types)
            input_channels (dict): Number of input features for each node type.
            hidden_channels (int): Hidden embedding size.
            out_channels (int): Size of each output sample. Defaults to 2.
            num_layers (int): Number of layers. Defaults to 3.
            num_heads (int): Number of multi-head-attention heads. Defaults to 1.
            backend (str): Graph convolutional layer backend. Defaults to "sage".
            edge_attr_dim (int, optional): Dimension of edge attributes for GAT. Defaults to None.
        """
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()

        # Handle both old tuple format and new dict format
        if isinstance(metadata, dict):
            node_types = list(metadata['nodes'].keys())
        else:
            # Legacy tuple format: (node_types, edge_types)
            node_types = metadata[0]

        # Validate input_channels
        if not isinstance(input_channels, dict):
            raise ValueError("input_channels must be a dictionary")

        # Input layers for each node type - no more lazy initialization
        for node_type in node_types:
            if node_type not in input_channels:
                raise ValueError(f"input_channels must contain entry for node type '{node_type}'")
            self.lin_dict[node_type] = Linear(input_channels[node_type], hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = RGATConv(metadata, hidden_channels, num_relations=3)
            self.convs.append(conv)

        # Output layers for target node types
        self.out_dict = torch.nn.ModuleDict({
            "bus": Linear(hidden_channels, out_channels),
            "generator": Linear(hidden_channels, out_channels),
        })

        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters of the model."""
        for lin in self.lin_dict.values():
            lin.reset_parameters()
        for conv in self.convs:
            if hasattr(conv, 'reset_parameters'):
                conv.reset_parameters()
        for out in self.out_dict.values():
            out.reset_parameters()

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None, minmax_scaling=False, **kwargs):
        """Forward pass of the RGAT model.

        Args:
            x_dict (dict): Node features for each node type.
            edge_index_dict (dict): Edge indices for each edge type.
            edge_attr_dict (dict, optional): Edge attributes for each edge type. Defaults to None
            minmax_scaling (bool): Whether to apply min-max scaling to outputs. Defaults to False.

        Returns:
            dict: Output predictions for each target node type.
        """

        if minmax_scaling:
            _vmin = x_dict['bus'][:, 1].clone()  # Original voltage min
            _vmax = x_dict['bus'][:, 2].clone()  # Original voltage max
            _pmin = x_dict['generator'][:, 2].clone()  # Original active power min
            _pmax = x_dict['generator'][:, 3].clone()  # Original active power max
            _qmin = x_dict['generator'][:, 5].clone()  # Original reactive power min
            _qmax = x_dict['generator'][:, 6].clone()  # Original reactive power max

        # Transform input features
        x_dict = {
            node_type: F.relu(self.lin_dict[node_type](x))
            for node_type, x in x_dict.items()
        }

        # Message passing
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        # Final predictions
        bus_out = self.out_dict["bus"](x_dict["bus"])
        gen_out = self.out_dict["generator"](x_dict["generator"])

        if minmax_scaling:
            # Create new tensors instead of modifying in-place
            bus_out_final = bus_out.clone()
            gen_out_final = gen_out.clone()

            # Apply scaling without in-place operations
            bus_out_final[:, 1] = F.sigmoid(bus_out[:, 1]) * (_vmax - _vmin) + _vmin

            gen_out_sigmoid = F.sigmoid(gen_out)
            gen_out_final[:, 0] = gen_out_sigmoid[:, 0] * (_pmax - _pmin) + _pmin
            gen_out_final[:, 1] = gen_out_sigmoid[:, 1] * (_qmax - _qmin) + _qmin

            return {"bus": bus_out_final, "generator": gen_out_final}
        else:
            return {"bus": bus_out, "generator": gen_out}


class HEAT(torch.nn.Module):
    """Heterogeneous Edge-Attributed Transformer using ``HEATConv`` via ``HeteroConv``.

    Wraps per-edge-type ``HEATConv`` layers inside ``HeteroConv`` for
    attention-based message passing with edge attributes.

    Input shape per node type: ``(N_type, input_channels[type])``.
    Output shape: ``{'bus': (N_bus, out_channels), 'generator': (N_gen, out_channels)}``.
    """

    def __init__(
            self,
            metadata,
            input_channels,
            hidden_channels=64,
            out_channels=2,
            num_layers=3,
            attention_heads=1,
            backend="sage",
            edge_attr_dim=None,
            **kwargs):
        r""" Heterogeneous Edge-Attributed Transformer (HEAT) model.

        Args:
            metadata (dict or tuple): Metadata containing node types and edge types.
                If dict: {'nodes': {node_type: dim, ...}, 'edges': {edge_type: dim, ...}}
                If tuple: (node_types, edge_types)
            input_channels (dict): Number of input features for each node type.
            hidden_channels (int): Hidden embedding size.
            out_channels (int): Size of each output sample. Defaults to 2.
            num_layers (int): Number of layers. Defaults to 3.
            attention_heads (int): Number of attention heads. Defaults to 1.
            backend (str): Graph convolutional layer backend. Defaults to "sage".
            edge_attr_dim (int, optional): Dimension of edge attributes for GAT. Defaults to None.
        """
        super().__init__()

        from torch_geometric.nn import HEATConv  # Import HEATConv

        self.lin_dict = torch.nn.ModuleDict()

        # Handle both old tuple format and new dict format
        if isinstance(metadata, dict):
            node_types = list(metadata['nodes'].keys())
            edge_types = list(metadata['edges'].keys())
        else:
            # Legacy tuple format: (node_types, edge_types)
            node_types = metadata[0]
            edge_types = metadata[1]

        # Validate input_channels
        if not isinstance(input_channels, dict):
            raise ValueError("input_channels must be a dictionary")

        # Input layers for each node type - no more lazy initialization
        for node_type in node_types:
            if node_type not in input_channels:
                raise ValueError(f"input_channels must contain entry for node type '{node_type}'")
            self.lin_dict[node_type] = Linear(input_channels[node_type], hidden_channels)

        # NOTE: HEATConv layers for handling heterogeneous edge attributes
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: HEATConv(
                    in_channels=(-1, -1),
                    out_channels=hidden_channels,
                    heads=attention_heads,
                    edge_dim=-1,  # Will use the edge attributes dimension from data
                    add_self_loops=False
                )
                for edge_type in edge_types
            }, aggr='sum')
            self.convs.append(conv)

        # Output layers for target node types
        self.out_dict = torch.nn.ModuleDict({
            "bus": Linear(hidden_channels, out_channels),
            "generator": Linear(hidden_channels, out_channels),
        })

        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters of the model."""
        for lin in self.lin_dict.values():
            lin.reset_parameters()
        for conv in self.convs:
            for rel_conv in conv.convs.values():
                if hasattr(rel_conv, 'reset_parameters'):
                    rel_conv.reset_parameters()
            conv.reset_parameters()
        for out in self.out_dict.values():
            out.reset_parameters()

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None, minmax_scaling=False, **kwargs):
        """Forward pass of the HEAT model.

        Args:
            x_dict (dict): Node features for each node type.
            edge_index_dict (dict): Edge indices for each edge type.
            edge_attr_dict (dict, optional): Edge attributes for each edge type. Defaults to None
            minmax_scaling (bool): Whether to apply min-max scaling to outputs. Defaults to False.

        Returns:
            dict: Output predictions for each target node type.
        """

        if minmax_scaling:
            _vmin = x_dict['bus'][:, 1].clone()  # Original voltage min
            _vmax = x_dict['bus'][:, 2].clone()  # Original voltage max
            _pmin = x_dict['generator'][:, 2].clone()  # Original active power min
            _pmax = x_dict['generator'][:, 3].clone()  # Original active power max
            _qmin = x_dict['generator'][:, 5].clone()  # Original reactive power min
            _qmax = x_dict['generator'][:, 6].clone()  # Original reactive power max

        # Transform input features
        x_dict = {
            node_type: F.relu(self.lin_dict[node_type](x))
            for node_type, x in x_dict.items()
        }

        # Message passing with edge attributes
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        # Final predictions
        bus_out = self.out_dict["bus"](x_dict["bus"])
        gen_out = self.out_dict["generator"](x_dict["generator"])

        if minmax_scaling:
            # Create new tensors instead of modifying in-place
            bus_out_final = bus_out.clone()
            gen_out_final = gen_out.clone()

            # Apply scaling without in-place operations
            bus_out_final[:, 1] = F.sigmoid(bus_out[:, 1]) * (_vmax - _vmin) + _vmin

            gen_out_sigmoid = F.sigmoid(gen_out)
            gen_out_final[:, 0] = gen_out_sigmoid[:, 0] * (_pmax - _pmin) + _pmin
            gen_out_final[:, 1] = gen_out_sigmoid[:, 1] * (_qmax - _qmin) + _qmin

            return {"bus": bus_out_final, "generator": gen_out_final}
        else:
            return {"bus": bus_out, "generator": gen_out}


class HGT(torch.nn.Module):
    """Heterogeneous Graph Transformer using ``HGTConv``.

    Applies type-specific multi-head attention following the HGT
    architecture. Each node and edge type receives dedicated attention
    weight matrices.

    Input shape per node type: ``(N_type, input_channels[type])``.
    Output shape: ``{'bus': (N_bus, out_channels), 'generator': (N_gen, out_channels)}``.
    """

    def __init__(self,
                 metadata,
                 input_channels,
                 hidden_channels=64,
                 out_channels=2,
                 num_layers=3,
                 num_heads=1,
                 backend="sage",
                 edge_attr_dim=None,
                 **kwargs):
        r""" Heterogeneous Graph Transformer (HGT) model.

        Args:
            metadata (dict or tuple): Metadata containing node types and edge types.
                If dict: {'nodes': {node_type: dim, ...}, 'edges': {edge_type: dim, ...}}
                If tuple: (node_types, edge_types)
            input_channels (dict): Number of input features for each node type.
            hidden_channels (int): Hidden embedding size.
            out_channels (int): Size of each output sample. Defaults to 2.
            num_layers (int): Number of layers. Defaults to 3.
            num_heads (int): Number of multi-head-attention heads. Defaults to 1.
            backend (str): Graph convolutional layer backend. Defaults to "sage".
            edge_attr_dim (int, optional): Dimension of edge attributes for GAT. Defaults to None.
        """
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()

        # Handle both old tuple format and new dict format
        if isinstance(metadata, dict):
            self.node_types = list(metadata['nodes'].keys())
            self.edge_types = list(metadata['edges'].keys())
        else:
            # Legacy tuple format: (node_types, edge_types)
            self.node_types = metadata[0]
            self.edge_types = metadata[1]

        # Validate input_channels
        if not isinstance(input_channels, dict):
            raise ValueError("input_channels must be a dictionary")

        # Input layers for each node type - no more lazy initialization
        for node_type in self.node_types:
            if node_type not in input_channels:
                raise ValueError(f"input_channels must contain entry for node type '{node_type}'")
            self.lin_dict[node_type] = Linear(input_channels[node_type], hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata, num_heads)
            self.convs.append(conv)

        # Output layers for target node types
        self.out_dict = torch.nn.ModuleDict({
            "bus": Linear(hidden_channels, out_channels),
            "generator": Linear(hidden_channels, out_channels),
        })

        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters of the model."""
        for lin in self.lin_dict.values():
            lin.reset_parameters()
        for conv in self.convs:
            if hasattr(conv, 'reset_parameters'):
                conv.reset_parameters()
        for out in self.out_dict.values():
            out.reset_parameters()

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None, minmax_scaling=False, **kwargs):
        """Forward pass of the HGT model.

        Args:
            x_dict (dict): Node features for each node type.
            edge_index_dict (dict): Edge indices for each edge type.
            edge_attr_dict (dict, optional): Edge attributes for each edge type. Defaults to None
            minmax_scaling (bool): Whether to apply min-max scaling to outputs. Defaults to False.

        Returns:
            dict: Output predictions for each target node type.
        """

        if minmax_scaling:
            _vmin = x_dict['bus'][:, 1].clone()  # Original voltage min
            _vmax = x_dict['bus'][:, 2].clone()  # Original voltage max
            _pmin = x_dict['generator'][:, 2].clone()  # Original active power min
            _pmax = x_dict['generator'][:, 3].clone()  # Original active power max
            _qmin = x_dict['generator'][:, 5].clone()  # Original reactive power min
            _qmax = x_dict['generator'][:, 6].clone()  # Original reactive power max

        # Transform input features
        x_dict = {
            node_type: torch.relu(self.lin_dict[node_type](x_dict[node_type]))
            for node_type in self.node_types
        }

        # Message passing
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        # Final predictions
        bus_out = self.out_dict["bus"](x_dict["bus"])
        gen_out = self.out_dict["generator"](x_dict["generator"])

        if minmax_scaling:
            # Create new tensors instead of modifying in-place
            bus_out_final = bus_out.clone()
            gen_out_final = gen_out.clone()

            # Apply scaling without in-place operations
            bus_out_final[:, 1] = F.sigmoid(bus_out[:, 1]) * (_vmax - _vmin) + _vmin

            gen_out_sigmoid = F.sigmoid(gen_out)
            gen_out_final[:, 0] = gen_out_sigmoid[:, 0] * (_pmax - _pmin) + _pmin
            gen_out_final[:, 1] = gen_out_sigmoid[:, 1] * (_qmax - _qmin) + _qmin

            return {"bus": bus_out_final, "generator": gen_out_final}
        else:
            return {"bus": bus_out, "generator": gen_out}


class HEAT_v2(torch.nn.Module):
    """Improved HEAT model with per-edge-type linear projections and concat heads.

    Adds learnable edge attribute projections for every edge type and
    uses ``HEATConv`` with ``concat=True`` for richer multi-head
    representations.

    Input shape per node type: ``(N_type, input_channels[type])``.
    Output shape: ``{'bus': (N_bus, out_channels), 'generator': (N_gen, out_channels)}``.
    """

    def __init__(
            self,
            metadata,
            input_channels,
            hidden_channels=64,
            out_channels=2,
            num_layers=3,
            attention_heads=1,
            edge_type_emb_dim=16,
            edge_attr_emb_dim=16,
            backend="sage",
            edge_attr_dim=None,
            **kwargs):
        r""" Improved Heterogeneous Edge-Attributed Transformer (HEAT) model.

        Args:
            metadata (dict or tuple): Metadata containing node types and edge types.
                If dict: {'nodes': {node_type: dim, ...}, 'edges': {edge_type: dim, ...}}
                If tuple: (node_types, edge_types)
            input_channels (dict): Number of input features for each node type.
            hidden_channels (int): Hidden embedding size.
            out_channels (int): Size of each output sample. Defaults to 2.
            num_layers (int): Number of layers. Defaults to 3.
            attention_heads (int): Number of attention heads. Defaults to 1.
            edge_type_emb_dim (int): Dimension of edge type embeddings. Defaults to 16.
            edge_attr_emb_dim (int): Dimension of edge attribute embeddings. Defaults to 16.
            backend (str): Graph convolutional layer backend. Defaults to "sage".
            edge_attr_dim (int, optional): Dimension of edge attributes for GAT. Defaults to None.
        """
        super().__init__()

        from torch_geometric.nn import HEATConv

        self.lin_dict = torch.nn.ModuleDict()
        self.edge_lin_dict = torch.nn.ModuleDict()

        # Handle both old tuple format and new dict format
        if isinstance(metadata, dict):
            node_types = list(metadata['nodes'].keys())
            edge_types = list(metadata['edges'].keys())
        else:
            # Legacy tuple format: (node_types, edge_types)
            node_types = metadata[0]
            edge_types = metadata[1]

        # Validate input_channels
        if not isinstance(input_channels, dict):
            raise ValueError("input_channels must be a dictionary")

        # Input layers for nodes - no more lazy initialization
        for node_type in node_types:
            if node_type not in input_channels:
                raise ValueError(f"input_channels must contain entry for node type '{node_type}'")
            self.lin_dict[node_type] = Linear(input_channels[node_type], hidden_channels)

        # Input layers for edges
        for edge_type in edge_types:
            # Convert edge_type tuple to string for ModuleDict key
            edge_type_str = str(edge_type)
            self.edge_lin_dict[edge_type_str] = Linear(-1, hidden_channels)

        # HEATConv layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            # conv = HEATConv(
            #     in_channels=(-1, -1),
            #     out_channels=hidden_channels,
            #     heads=attention_heads,
            #     edge_dim=hidden_channels,
            #     add_self_loops=False
            # )
            conv = HEATConv(
                in_channels=-1,
                out_channels=hidden_channels,
                num_node_types=len(node_types),
                num_edge_types=len(edge_types),
                edge_type_emb_dim=edge_type_emb_dim,
                edge_dim=9,
                edge_attr_emb_dim=edge_attr_emb_dim,
                heads=attention_heads,
                concat=True,
            )
            self.convs.append(conv)

        # Output layers
        self.out_dict = torch.nn.ModuleDict({
            "bus": Linear(hidden_channels, out_channels),
            "generator": Linear(hidden_channels, out_channels),
        })

        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters of the model."""
        for lin in self.lin_dict.values():
            lin.reset_parameters()
        for edge_lin in self.edge_lin_dict.values():
            edge_lin.reset_parameters()
        for conv in self.convs:
            if hasattr(conv, 'reset_parameters'):
                conv.reset_parameters()
        for out in self.out_dict.values():
            out.reset_parameters()

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None, minmax_scaling=False, **kwargs):
        """Forward pass of the HEAT_v2 model.

        Args:
            x_dict (dict): Node features for each node type.
            edge_index_dict (dict): Edge indices for each edge type.
            edge_attr_dict (dict, optional): Edge attributes for each edge type. Defaults to None
            minmax_scaling (bool): Whether to apply min-max scaling to outputs. Defaults to False.

        Returns:
            dict: Output predictions for each target node type.
        """

        if minmax_scaling:
            _vmin = x_dict['bus'][:, 1].clone()  # Original voltage min
            _vmax = x_dict['bus'][:, 2].clone()  # Original voltage max
            _pmin = x_dict['generator'][:, 2].clone()  # Original active power min
            _pmax = x_dict['generator'][:, 3].clone()  # Original active power max
            _qmin = x_dict['generator'][:, 5].clone()  # Original reactive power min
            _qmax = x_dict['generator'][:, 6].clone()  # Original reactive power max

        # Transform node features
        x_dict = {
            node_type: torch.relu(self.lin_dict[node_type](x))
            for node_type, x in x_dict.items()
        }

        # Transform edge features if provided
        if edge_attr_dict is not None:
            edge_attr_dict = {
                edge_type: torch.relu(self.edge_lin_dict[str(edge_type)](edge_attr))
                for edge_type, edge_attr in edge_attr_dict.items()
            }

        # Message passing
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict,
                          node_types, edge_types, edge_attr_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        # Final predictions
        bus_out = self.out_dict["bus"](x_dict["bus"])
        gen_out = self.out_dict["generator"](x_dict["generator"])

        if minmax_scaling:
            # Create new tensors instead of modifying in-place
            bus_out_final = bus_out.clone()
            gen_out_final = gen_out.clone()

            # Apply scaling without in-place operations
            bus_out_final[:, 1] = F.sigmoid(bus_out[:, 1]) * (_vmax - _vmin) + _vmin

            gen_out_sigmoid = F.sigmoid(gen_out)
            gen_out_final[:, 0] = gen_out_sigmoid[:, 0] * (_pmax - _pmin) + _pmin
            gen_out_final[:, 1] = gen_out_sigmoid[:, 1] * (_qmax - _qmin) + _qmin

            return {"bus": bus_out_final, "generator": gen_out_final}
        else:
            return {"bus": bus_out, "generator": gen_out}


class HGNN_Base(torch.nn.Module):
    """Abstract base class for heterogeneous GNN models.

    Provides shared input projection layers (``lin_dict``) and output
    heads (``out_dict``) for ``bus`` and ``generator`` node types.
    Subclasses must implement ``forward``.
    """

    def __init__(self,
                 metadata,
                 input_channels,
                 hidden_channels=64,
                 out_channels=2,
                 num_layers=3,
                 backend="sage",
                 edge_attr_dim=None,
                 **kwargs):
        """Initialize the HGNN base with input projections and output heads.

        Args:
            metadata (dict or tuple): Metadata containing node types and edge types.
                If dict: {'nodes': {node_type: dim, ...}, 'edges': {edge_type: dim, ...}}
                If tuple: (node_types, edge_types)
            input_channels (dict): Number of input features for each node type.
            hidden_channels (int): Hidden embedding size.
            out_channels (int): Size of each output sample. Defaults to 2.
            num_layers (int): Number of layers. Defaults to 3.
            backend (str): Graph convolutional layer backend. Defaults to "sage".
            edge_attr_dim (int, optional): Dimension of edge attributes for GAT. Defaults to None.
        """
        super().__init__()
        self.lin_dict = torch.nn.ModuleDict()

        # Handle both old tuple format and new dict format
        if isinstance(metadata, dict):
            node_types = list(metadata['nodes'].keys())
        else:
            # Legacy tuple format: (node_types, edge_types)
            node_types = metadata[0]

        # Validate input_channels
        if not isinstance(input_channels, dict):
            raise ValueError("input_channels must be a dictionary")

        # Input layers for each node type - no more lazy initialization
        for node_type in node_types:
            if node_type not in input_channels:
                raise ValueError(f"input_channels must contain entry for node type '{node_type}'")
            self.lin_dict[node_type] = Linear(input_channels[node_type], hidden_channels)

        self.out_dict = torch.nn.ModuleDict({
            "bus": Linear(hidden_channels, out_channels),
            "generator": Linear(hidden_channels, out_channels),
        })

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None, minmax_scaling=False, **kwargs):
        """Forward pass (must be implemented by subclasses).

        Args:
            x_dict (dict): Node features for each node type.
            edge_index_dict (dict): Edge indices for each edge type.
            edge_attr_dict (dict, optional): Edge attributes for each edge
                type. Defaults to None.
            minmax_scaling (bool): Whether to apply min-max scaling to
                outputs. Defaults to False.

        Raises:
            NotImplementedError: Always raised in the base class.
        """
        raise NotImplementedError("Forward method not implemented in base class")


class GridEcoder(torch.nn.Module):
    """Placeholder encoder for power grid graph representations.

    Not yet implemented. Reserved for future grid-specific encoding logic.
    """

    def __init__(self):
        pass


class ACOPFModel(torch.nn.Module):
    """Placeholder ACOPF model for power systems.

    Not yet implemented. Reserved for a future end-to-end ACOPF model that
    combines graph encoding with constraint-aware decoding.
    """

    def __init__(self, ):
        pass


class ModelFactory:
    """Factory for instantiating heterogeneous GNN model architectures.

    Currently supports ``'heterognn'``, ``'hgt'``, ``'heat'``, and ``'rgat'``.
    HGT, HEAT, and RGAT fall back to HeteroGNN with a warning until their
    dedicated configurations are finalized.
    """

    @staticmethod
    def create_model(model_name, metadata, input_channels, config):
        """Create a model instance by name and configuration.

        Args:
            model_name (str): Model architecture name (case-insensitive).
                One of ``'heterognn'``, ``'hgt'``, ``'heat'``, ``'rgat'``.
            metadata (dict or tuple): Graph metadata describing node and
                edge types.
            input_channels (dict): Mapping of node type to input feature
                dimension.
            config (dict): Full training configuration dict; model
                hyperparameters are read from ``config['models'][name]``.

        Returns:
            torch.nn.Module: An instantiated GNN model.

        Raises:
            ValueError: If ``model_name`` is not recognized.
        """
        model_name = model_name.lower()

        if model_name == 'heterognn':
            model_config = config['models']['HeteroGNN']
            return HeteroGNN(
                metadata=metadata,
                input_channels=input_channels,
                hidden_channels=model_config['hidden_channels'],
                num_layers=model_config['num_layers'],
                backend=model_config.get('backend', 'gcn'),
                dropout=model_config.get('dropout', 0.0),
                out_channels=2  # VM, VA for bus; PG, QG for generator
            )
        elif model_name == 'hgt':
            model_config = config['models'].get('HGT', {})
            # Placeholder for HGT - to be implemented
            print("Warning: HGT model not yet implemented, falling back to HeteroGNN")
            return ModelFactory.create_model('heterognn', metadata, input_channels, config)
        elif model_name == 'heat':
            model_config = config['models'].get('HEAT', {})
            # Placeholder for HEAT - to be implemented
            print("Warning: HEAT model not yet implemented, falling back to HeteroGNN")
            return ModelFactory.create_model('heterognn', metadata, input_channels, config)
        elif model_name == 'rgat':
            model_config = config['models'].get('RGAT', {})
            # Placeholder for RGAT - to be implemented
            print("Warning: RGAT model not yet implemented, falling back to HeteroGNN")
            return ModelFactory.create_model('heterognn', metadata, input_channels, config)
        else:
            raise ValueError(f"Unknown model: {model_name}. Available models: heterognn, hgt, heat, rgat")

    @staticmethod
    def get_available_models():
        """Return the list of supported model architecture names.

        Returns:
            list[str]: Available model names.
        """
        return ['heterognn', 'hgt', 'heat', 'rgat']
