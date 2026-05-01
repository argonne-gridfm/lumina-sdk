"""Tests for lumina.dataset.opf.transforms."""

import torch
from torch_geometric.data import HeteroData

from lumina.dataset.opf.transforms import to_float32


def _make_sample():
    data = HeteroData()
    data['bus'].x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    data['bus'].y = torch.tensor([[0.5], [0.6]], dtype=torch.float64)
    data['generator'].x = torch.tensor([[7.0]], dtype=torch.float64)
    data['bus', 'ac_line', 'bus'].edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    data['bus', 'ac_line', 'bus'].edge_attr = torch.tensor([[0.1, 0.2]], dtype=torch.float64)
    return data


def test_to_float32_casts_float64_node_features():
    data = to_float32(_make_sample())
    assert data['bus'].x.dtype == torch.float32
    assert data['bus'].y.dtype == torch.float32
    assert data['generator'].x.dtype == torch.float32


def test_to_float32_casts_float64_edge_attrs():
    data = to_float32(_make_sample())
    assert data['bus', 'ac_line', 'bus'].edge_attr.dtype == torch.float32


def test_to_float32_preserves_long_edge_index():
    data = to_float32(_make_sample())
    assert data['bus', 'ac_line', 'bus'].edge_index.dtype == torch.long


def test_to_float32_preserves_values():
    original = _make_sample()
    expected = original['bus'].x.detach().clone().float()
    data = to_float32(original)
    assert torch.equal(data['bus'].x, expected)


def test_to_float32_leaves_float32_unchanged():
    data = HeteroData()
    data['bus'].x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    out = to_float32(data)
    assert out['bus'].x.dtype == torch.float32


def test_to_float32_returns_same_object():
    data = _make_sample()
    out = to_float32(data)
    assert out is data
