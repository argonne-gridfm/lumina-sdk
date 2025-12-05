import torch
from torch_geometric.data import HeteroData

from lumina.model.opf.losses import OPFLossManager


def _build_dummy_batch(device):
    data = HeteroData()

    # Two-bus system
    data['bus'].x = torch.tensor([[1.0], [1.0]], device=device)
    data['load'].x = torch.tensor([[10.0, 5.0]], device=device)  # pd, qd
    data['generator'].x = torch.tensor([[0.0, 0.0]], device=device)

    # Bus 0 has load 0, bus 1 hosts generator 0
    data['bus', 'load_link', 'load'].edge_index = torch.tensor([[0], [0]], device=device)
    data['bus', 'generator_link', 'generator'].edge_index = torch.tensor([[1], [0]], device=device)

    # Single line between bus 0 and 1 with rate_a as limit
    data['bus', 'ac_line', 'bus'].edge_index = torch.tensor([[0], [1]], device=device)
    data['bus', 'ac_line', 'bus'].edge_attr = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 50.0]], device=device
    )

    data.baseMVA = 100.0
    return data


def test_constraint_batch_uses_real_data():
    device = torch.device("cpu")
    manager = OPFLossManager(loss_type='augmented_lagrangian')
    batch = _build_dummy_batch(device)
    predictions = {
        'bus': torch.zeros((2, 2), device=device),
        'generator': torch.zeros((1, 2), device=device),
    }

    constraint_batch = manager._create_constraint_batch(batch, predictions)

    assert torch.allclose(constraint_batch.get('pd'), torch.tensor([10.0, 0.0], device=device))
    assert torch.allclose(constraint_batch.get('qd'), torch.tensor([5.0, 0.0], device=device))
    assert torch.equal(constraint_batch.get('gen_bus_indices'), torch.tensor([1], device=device))
    assert torch.equal(constraint_batch.get('load_bus_indices'), torch.tensor([0], device=device))
    assert constraint_batch.get('line_edge_index').shape == (2, 1)
    assert torch.equal(constraint_batch.get('line_limits'), torch.tensor([50.0], device=device))


def test_network_parameters_initialized_from_batch():
    device = torch.device("cpu")
    manager = OPFLossManager(loss_type='augmented_lagrangian')
    batch = _build_dummy_batch(device)
    manager._ensure_network_parameters(batch, device)

    assert manager.lagrangian.Y_real is not None
    assert manager.lagrangian.Y_real.shape == (2, 2)
    assert manager.lagrangian.Y_imag.shape == (2, 2)
    assert manager.lagrangian.line_limits is not None
    assert manager.lagrangian.line_limits.numel() == 1
