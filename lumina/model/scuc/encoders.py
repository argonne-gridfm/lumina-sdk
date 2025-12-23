"""
SCUC encoder modules.

These encoders mirror the legacy implementations that lived in `hgnn.py`.
They produce per-node embeddings without projecting to task-specific heads so
that downstream heads (e.g. SCUC temporal models) can consume them.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, HGTConv, Linear, SAGEConv


class HGTEncoder(torch.nn.Module):
    """Heterogeneous graph transformer encoder for SCUC graphs."""

    def __init__(
        self,
        metadata,
        input_channels,
        hidden_channels: int = 64,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.metadata = metadata

        self.lin_dict = torch.nn.ModuleDict(
            {
                ntype: Linear(input_channels[ntype], hidden_channels)
                for ntype in metadata[0]
            }
        )

        self.convs = torch.nn.ModuleList(
            [
                HGTConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    metadata=metadata,
                    heads=heads,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = torch.nn.Dropout(dropout)
        self.norms = torch.nn.ModuleDict(
            {ntype: torch.nn.LayerNorm(hidden_channels) for ntype in metadata[0]}
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        x_dict = {nt: F.relu_(self.lin_dict[nt](x)) for nt, x in x_dict.items()}
        for conv in self.convs:
            out = conv(x_dict, edge_index_dict)
            x_dict = {
                nt: self.norms[nt](x_dict[nt] + self.dropout(F.relu(v)))
                for nt, v in out.items()
            }
        return x_dict


class HGNNEncoder(torch.nn.Module):
    """Simple heterogeneous GNN encoder returning per-node embeddings."""

    def __init__(
        self,
        metadata,
        input_channels,
        hidden_channels: int = 64,
        num_layers: int = 3,
        backend: str = "sage",
    ):
        super().__init__()
        self.lin_dict = torch.nn.ModuleDict(
            {
                ntype: Linear(input_channels[ntype], hidden_channels)
                for ntype in metadata[0]
            }
        )

        if backend not in {"sage", "gcn"}:
            raise ValueError(f"Unsupported backend {backend}")

        def make_conv():
            return SAGEConv((-1, -1), hidden_channels)

        self.convs = torch.nn.ModuleList(
            [
                HeteroConv(
                    {etype: make_conv() for etype in metadata[1]},
                    aggr="sum",
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        x_dict = {ntype: self.lin_dict[ntype](x).relu_() for ntype, x in x_dict.items()}
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(v) for key, v in x_dict.items()}
        return x_dict


__all__ = ["HGNNEncoder", "HGTEncoder"]

