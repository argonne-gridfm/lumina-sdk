import pytest
import torch
from torch_geometric.data import Batch, HeteroData

from lumina.model.opf.hetero_model import HGT
from lumina.model.opf.losses import ACOPFLossFunction, OPFLossManager, compute_target_stats

N_BUS, N_GEN = 6, 3
INPUT_CHANNELS = {"bus": 7, "generator": 11}
METADATA = (
    ["bus", "generator"],
    [("bus", "ac_line", "bus"), ("generator", "generator_link", "bus"), ("bus", "bus_link", "generator")],
)


def make_sample(seed):
    g = torch.Generator().manual_seed(seed)
    data = HeteroData()
    data["bus"].x = torch.rand(N_BUS, 7, generator=g) * torch.tensor([200.0, 1, 1, 1, 1, 1, 1])
    data["generator"].x = torch.rand(N_GEN, 11, generator=g)
    data["bus"].y = torch.rand(N_BUS, 2, generator=g)
    data["generator"].y = torch.rand(N_GEN, 2, generator=g)
    data["generator"].y[0] = torch.tensor([0.5, 0.5])
    data["bus", "ac_line", "bus"].edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    data["generator", "generator_link", "bus"].edge_index = torch.tensor([[0, 1, 2], [0, 2, 4]])
    data["bus", "bus_link", "generator"].edge_index = torch.tensor([[0, 2, 4], [0, 1, 2]])
    return data


def make_model(**kwargs):
    return HGT(METADATA, INPUT_CHANNELS, hidden_channels=8, num_layers=2, num_heads=2, **kwargs)


def test_standardize_input_off_keeps_state_dict_unchanged():
    model = make_model()
    assert model.standardizer is None
    assert not any(key.startswith("standardizer") for key in model.state_dict())


def test_fit_input_stats_fills_buffers_and_standardizes():
    samples = [make_sample(i) for i in range(5)]
    model = make_model(standardize_input=True)
    model.fit_input_stats(samples)

    x = torch.cat([s["bus"].x for s in samples])
    z = model.standardizer["bus"](x)
    assert torch.allclose(z.mean(0), torch.zeros(7), atol=1e-5)
    assert torch.allclose(z.std(0, unbiased=False), torch.ones(7), atol=1e-4)

    state = model.state_dict()
    assert torch.equal(state["standardizer.bus.mean"], model.standardizer["bus"].mean)

    reloaded = make_model(standardize_input=True)
    reloaded.load_state_dict(state)
    assert torch.equal(reloaded.standardizer["generator"].std, model.standardizer["generator"].std)

    batch = Batch.from_data_list(samples)
    out = model(batch.x_dict, batch.edge_index_dict, minmax_scaling=True)
    assert out["bus"].shape == (5 * N_BUS, 2)
    assert out["generator"].shape == (5 * N_GEN, 2)


def test_constant_feature_keeps_unit_std():
    samples = [make_sample(i) for i in range(3)]
    for s in samples:
        s["bus"].x[:, 3] = 1.0
    model = make_model(standardize_input=True)
    model.fit_input_stats(samples)
    assert model.standardizer["bus"].std[3] == 1.0


def test_compute_target_stats_shapes_and_movers():
    samples = [make_sample(i) for i in range(8)]
    stats = compute_target_stats(samples)

    assert stats["bus"]["scale"].shape == (2,)
    assert stats["generator"]["node_std"].shape == (N_GEN, 2)
    y = torch.stack([s["generator"].y for s in samples])
    assert torch.allclose(stats["generator"]["node_std"], y.std(0, unbiased=False), atol=1e-5)
    assert torch.allclose(stats["generator"]["scale"], y.reshape(-1, 2).std(0, unbiased=False), atol=1e-5)
    assert not stats["generator"]["movers"][0].any()
    assert stats["generator"]["movers"][1:].all()


def test_normalize_targets_rescales_mse():
    samples = [make_sample(i) for i in range(4)]
    stats = compute_target_stats(samples)
    batch = Batch.from_data_list(samples)
    predictions = {"bus": torch.zeros(4 * N_BUS, 2), "generator": torch.zeros(4 * N_GEN, 2)}

    plain = OPFLossManager("mse").compute_loss(predictions, batch, return_info=False)
    scaled, info = OPFLossManager("mse", target_stats=stats, normalize_targets=True).compute_loss(predictions, batch)

    expected = sum(
        ((batch[nt].y / stats[nt]["scale"]) ** 2).mean() for nt in ("bus", "generator")
    )
    assert torch.allclose(scaled, expected)
    assert not torch.allclose(plain, scaled)
    assert "deviation_loss" not in info


def test_deviation_term_vanishes_for_per_node_offsets():
    samples = [make_sample(i) for i in range(4)]
    stats = compute_target_stats(samples)
    batch = Batch.from_data_list(samples)
    offset = {"bus": torch.randn(N_BUS, 2), "generator": torch.randn(N_GEN, 2)}
    predictions = {nt: batch[nt].y + offset[nt].repeat(4, 1) for nt in offset}

    loss_fn = ACOPFLossFunction("mse", target_stats=stats, deviation_weight=0.5)
    info = loss_fn(predictions, {nt: batch[nt].y for nt in offset}, num_graphs=4)
    assert info["deviation_loss"].abs() < 1e-6
    assert torch.allclose(info["total_loss"], info["bus_loss"] + info["generator_loss"])


def test_deviation_term_skipped_for_small_batches():
    samples = [make_sample(i) for i in range(3)]
    stats = compute_target_stats(samples)
    batch = Batch.from_data_list(samples)
    predictions = {nt: torch.zeros_like(batch[nt].y) for nt in ("bus", "generator")}

    manager = OPFLossManager("mse", target_stats=stats, deviation_weight=0.5)
    _, info = manager.compute_loss(predictions, batch)
    assert "deviation_loss" not in info

    manager = OPFLossManager("mse", target_stats=stats, deviation_weight=0.5, deviation_min_graphs=2)
    _, info = manager.compute_loss(predictions, batch)
    assert info["deviation_loss"] > 0


def test_stats_required_when_terms_enabled():
    with pytest.raises(ValueError):
        ACOPFLossFunction("mse", normalize_targets=True)
