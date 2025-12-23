"""
SCUC heads for temporal generation and commitment modelling.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TimePositionalEncoding(nn.Module):
    """Simple learnable positional encoding over the discrete time horizon."""

    def __init__(self, n_time: int, d_time: int):
        super().__init__()
        self.embedding = nn.Embedding(n_time, d_time)

    def forward(self, n_nodes: int, device: torch.device):
        time_idx = torch.arange(self.embedding.num_embeddings, device=device)
        pos_emb = self.embedding(time_idx)
        return pos_emb.unsqueeze(0).expand(n_nodes, -1, -1)


class SCUCTransformerHead(nn.Module):
    """Transformer-based temporal head that predicts Pg and commitment sequences."""

    def __init__(
        self,
        d_emb: int,
        d_time: int = 16,
        n_time: int = 36,
        hidden_dim: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
    ):
        super().__init__()
        self.n_time = n_time
        self.d_emb = d_emb
        self.d_time = d_time

        self.time_pos_enc = TimePositionalEncoding(n_time, d_time)

        d_in = d_emb + 1 + d_time
        self.input_proj = nn.Linear(d_in, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.pg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.unit_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_all, lp_all, gen_mask, return_all: bool = False):
        n_nodes = h_all.size(0)

        h_exp = h_all.unsqueeze(1).repeat(1, self.n_time, 1)
        lp_exp = lp_all.unsqueeze(-1)
        pos_emb = self.time_pos_enc(n_nodes, h_all.device)

        x_in = torch.cat([h_exp, lp_exp, pos_emb], dim=-1)
        x_in = self.input_proj(x_in)

        x_out = self.transformer(x_in)

        gen_indices = gen_mask.nonzero(as_tuple=False).view(-1)
        x_gen = x_out[gen_indices]

        pg = self.pg_head(x_gen).squeeze(-1)
        unit = self.unit_head(x_gen).squeeze(-1)
        if return_all:
            return pg, unit, x_out
        return pg, unit


class SCUCLSTMHead(nn.Module):
    """LSTM-based SCUC head mirroring the legacy implementation."""

    def __init__(
        self,
        d_emb: int,
        d_time: int = 16,
        n_time: int = 36,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.n_time = n_time
        self.d_emb = d_emb
        self.d_time = d_time
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        self.time_embedding = nn.Embedding(n_time, d_time)

        input_dim = d_emb + 1 + d_time
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.pg_head = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.unit_head = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_all, lp_all, gen_mask):
        n_all = h_all.size(0)
        t_steps = self.n_time
        device = h_all.device

        h_gen = h_all[gen_mask]
        lp_gen = lp_all[gen_mask]
        n_gen = h_gen.size(0)

        time_idx = torch.arange(t_steps, device=device)
        time_emb = self.time_embedding(time_idx)

        h_exp = h_gen.unsqueeze(1).repeat(1, t_steps, 1)
        lp_exp = lp_gen.unsqueeze(-1)
        time_exp = time_emb.unsqueeze(0).repeat(n_gen, 1, 1)

        x = torch.cat([h_exp, lp_exp, time_exp], dim=-1)
        x = self.input_proj(x)

        lstm_out, _ = self.lstm(x)

        pg = self.pg_head(lstm_out).squeeze(-1)
        unit = self.unit_head(lstm_out).squeeze(-1)

        return pg, unit


__all__ = ["SCUCTransformerHead", "SCUCLSTMHead", "TimePositionalEncoding"]

