""" Heterogeneous Graph Neural Network models for ACOPF problem.
Copyright (c) 2025, Argonne National Laboratory
All rights reserved.
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import (MLP, GATConv, GCNConv, GINConv, GraphConv,
                                HeteroConv, HGTConv, Linear, RGATConv,
                                SAGEConv, HEATConv)


class OPFHeteroGNN(torch.nn.Module):
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
        """ Heterogeneous Graph Neural Network (HeteroGNN) model for OPF.

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
        Implemented as HeteroConv with GATConv (Relational GAT).

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

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            # Use HeteroConv with GATConv to simulate RGAT (Relational GAT)
            # Each edge type gets its own GATConv parameters
            conv = HeteroConv({
                edge_type: GATConv(
                    hidden_channels,
                    hidden_channels // num_heads,
                    heads=num_heads,
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
            if edge_attr_dict is not None:
                x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            else:
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


class HEAT(torch.nn.Module):
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
        """Heterogeneous Edge-Attributed Transformer (HEAT) model.

        Notes:
          - Externally, this model keeps the "hetero" forward signature:
              forward(x_dict, edge_index_dict, edge_attr_dict=None, ...)
          - Internally, torch_geometric.nn.HEATConv operates on a *homogeneous* view.
            We therefore build a temporary HeteroData and call to_homogeneous().
          - If edge_attr_dict is not provided by the caller (as in train_opf.py today),
            this model will still run by creating zero edge attributes for all relations.
        """
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        self.edge_lin_dict = torch.nn.ModuleDict()

        # Handle both old tuple format and new dict format
        if isinstance(metadata, dict):
            self.node_types = list(metadata['nodes'].keys())
            self.edge_types = list(metadata['edges'].keys())
            edge_attr_dims = dict(metadata['edges'])
        else:
            # Legacy tuple format: (node_types, edge_types)
            self.node_types = metadata[0]
            self.edge_types = metadata[1]
            edge_attr_dims = {}

        # Validate input_channels
        if not isinstance(input_channels, dict):
            raise ValueError("input_channels must be a dictionary")

        # Node input projections
        for node_type in self.node_types:
            if node_type not in input_channels:
                raise ValueError(f"input_channels must contain entry for node type '{node_type}'")
            self.lin_dict[node_type] = Linear(input_channels[node_type], hidden_channels)

        for edge_type in self.edge_types:
            dim = edge_attr_dims.get(edge_type, 0)
            if dim and dim > 0:
                self.edge_lin_dict[str(edge_type)] = Linear(dim, hidden_channels)

        self._heat_edge_dim = hidden_channels

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HEATConv(
                in_channels=-1,
                out_channels=hidden_channels,
                num_node_types=len(self.node_types),
                num_edge_types=len(self.edge_types),
                edge_type_emb_dim=edge_type_emb_dim,
                edge_dim=self._heat_edge_dim,
                edge_attr_emb_dim=edge_attr_emb_dim,
                heads=attention_heads,
                concat=False,
            )
            self.convs.append(conv)

        # Output heads (targets)
        self.out_dict = torch.nn.ModuleDict({
            "bus": Linear(hidden_channels, out_channels),
            "generator": Linear(hidden_channels, out_channels),
        })

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lin_dict.values():
            lin.reset_parameters()
        for lin in self.edge_lin_dict.values():
            lin.reset_parameters()
        for conv in self.convs:
            if hasattr(conv, 'reset_parameters'):
                conv.reset_parameters()
        for out in self.out_dict.values():
            out.reset_parameters()

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None, minmax_scaling=False, **kwargs):
        if minmax_scaling:
            _vmin = x_dict['bus'][:, 1].clone()
            _vmax = x_dict['bus'][:, 2].clone()
            _pmin = x_dict['generator'][:, 2].clone()
            _pmax = x_dict['generator'][:, 3].clone()
            _qmin = x_dict['generator'][:, 5].clone()
            _qmax = x_dict['generator'][:, 6].clone()

        # Project node features to hidden_channels
        x_dict = {
            node_type: F.relu(self.lin_dict[node_type](x))
            for node_type, x in x_dict.items()
        }

        # Project edge attributes (if provided) to hidden_channels
        projected_edge_attr_dict = {}
        if edge_attr_dict is not None:
            for edge_type, edge_attr in edge_attr_dict.items():
                key = str(edge_type)
                if key in self.edge_lin_dict:
                    projected_edge_attr_dict[edge_type] = F.relu(self.edge_lin_dict[key](edge_attr))

        # HEATConv interface in PyG is not compatible, we have to construct a temporary homogeneous dataset object
        hdata = HeteroData()

        for node_type in self.node_types:
            if node_type not in x_dict:
                raise ValueError(f"Missing node type '{node_type}' in x_dict")
            hdata[node_type].x = x_dict[node_type]

        # Ensure every relation exists, and every relation has edge_attr aligned to edge_index
        for edge_type in self.edge_types:
            if edge_type in edge_index_dict:
                ei = edge_index_dict[edge_type]
            else:
                device = hdata[self.node_types[0]].x.device
                ei = torch.empty((2, 0), dtype=torch.long, device=device)

            hdata[edge_type].edge_index = ei
            num_edges = ei.size(1)
            device = ei.device

            if edge_type in projected_edge_attr_dict:
                ea = projected_edge_attr_dict[edge_type]
                if ea.size(0) != num_edges:
                    raise ValueError(
                        f"edge_attr rows ({ea.size(0)}) must match num_edges ({num_edges}) for edge_type={edge_type}"
                    )
                hdata[edge_type].edge_attr = ea
            else:
                # padding for missing edge attrs
                hdata[edge_type].edge_attr = torch.zeros(
                    (num_edges, self._heat_edge_dim),
                    dtype=hdata[self.node_types[0]].x.dtype,
                    device=device,
                )

        homo = hdata.to_homogeneous(node_attrs=['x'], edge_attrs=['edge_attr'])

        x = homo.x
        edge_index = homo.edge_index
        node_type = homo.node_type
        edge_type = homo.edge_type
        edge_attr = homo.edge_attr

        # Message passing (homogeneous)
        for conv in self.convs:
            x = conv(x, edge_index, node_type, edge_type, edge_attr)
            x = F.relu(x)

        node_type_to_id = {nt: i for i, nt in enumerate(self.node_types)}
        bus_x = x[node_type == node_type_to_id["bus"]]
        gen_x = x[node_type == node_type_to_id["generator"]]

        bus_out = self.out_dict["bus"](bus_x)
        gen_out = self.out_dict["generator"](gen_x)

        if minmax_scaling:
            bus_out_final = bus_out.clone()
            gen_out_final = gen_out.clone()

            bus_out_final[:, 1] = torch.sigmoid(bus_out[:, 1]) * (_vmax - _vmin) + _vmin

            gen_out_sigmoid = torch.sigmoid(gen_out)
            gen_out_final[:, 0] = gen_out_sigmoid[:, 0] * (_pmax - _pmin) + _pmin
            gen_out_final[:, 1] = gen_out_sigmoid[:, 1] * (_qmax - _qmin) + _qmin
            return {"bus": bus_out_final, "generator": gen_out_final}

        return {"bus": bus_out, "generator": gen_out}


class HEAT_v2(torch.nn.Module):
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
            self.node_types = list(metadata['nodes'].keys())
            self.edge_types = list(metadata['edges'].keys())
        else:
            # Legacy tuple format: (node_types, edge_types)
            self.node_types = metadata[0]
            self.edge_types = metadata[1]

        # Validate input_channels
        if not isinstance(input_channels, dict):
            raise ValueError("input_channels must be a dictionary")

        # Input layers for nodes - no more lazy initialization
        for node_type in self.node_types:
            if node_type not in input_channels:
                raise ValueError(f"input_channels must contain entry for node type '{node_type}'")
            self.lin_dict[node_type] = Linear(input_channels[node_type], hidden_channels)

        # Input layers for edges
        for edge_type in self.edge_types:
            # Convert edge_type tuple to string for ModuleDict key
            edge_type_str = str(edge_type)
            self.edge_lin_dict[edge_type_str] = Linear(-1, hidden_channels)

        # HEATConv layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HEATConv(
                in_channels=-1,
                out_channels=hidden_channels,
                num_node_types=len(self.node_types),
                num_edge_types=len(self.edge_types),
                edge_type_emb_dim=edge_type_emb_dim,
                edge_dim=hidden_channels,  # Adjusted to match projected edge attributes
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
                          self.node_types, self.edge_types, edge_attr_dict)
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

    def __init__(self,
                 metadata,
                 input_channels,
                 hidden_channels=64,
                 out_channels=2,
                 num_layers=3,
                 num_heads=1,
                 dropout=0.0,
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
        self.dropout = float(dropout)

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

        metadata_tuple = (self.node_types, self.edge_types) if isinstance(metadata, dict) else metadata
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, metadata_tuple, num_heads)
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
        if self.dropout > 0.0:
            x_dict = {
                key: F.dropout(x, p=self.dropout, training=self.training)
                for key, x in x_dict.items()
            }

        # Message passing
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            if self.dropout > 0.0:
                x_dict = {
                    key: F.dropout(x, p=self.dropout, training=self.training)
                    for key, x in x_dict.items()
                }

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
