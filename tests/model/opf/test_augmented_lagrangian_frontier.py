import torch
from torch_geometric.data import HeteroData

from lumina.model.opf.augmented_lagrangian import AugmentedLagrangianACOPF
from lumina.model.opf.losses import OPFLossManager


def _two_bus_batch():
    batch = HeteroData()
    batch["bus"].x = torch.zeros((2, 2))
    batch["bus"].y = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    batch["generator"].x = torch.zeros((1, 2))
    batch["generator"].y = torch.tensor([[0.5, 0.0]])
    batch["load"].x = torch.tensor([[0.5, 0.1]])
    batch["bus", "generator_link", "generator"].edge_index = torch.tensor([[0], [0]])
    batch["bus", "load_link", "load"].edge_index = torch.tensor([[1], [0]])
    batch["bus", "ac_line", "bus"].edge_index = torch.tensor([[0], [1]])
    batch["bus", "ac_line", "bus"].edge_attr = torch.tensor(
        [[-3.14, 3.14, 0.01, 0.01, 0.01, 0.1, 2.0, 2.0, 2.0]]
    )
    return batch


def test_augmented_lagrangian_is_finite_and_tracks_constraints():
    lagrangian = AugmentedLagrangianACOPF(
        mu_0=0.1,
        warmup_epochs=0,
        normalize_by_size=True,
        verbose=False,
    )
    objective = torch.tensor(2.0, requires_grad=True)
    constraints = torch.tensor([1.0, -0.5], requires_grad=True)

    loss, info = lagrangian.compute_augmented_lagrangian(objective, constraints)

    assert torch.isfinite(loss)
    assert info["penalty_term"].item() > 0
    assert info["raw_constraint_violation"].item() > 0
    loss.backward()
    assert objective.grad is not None
    assert constraints.grad is not None


def test_loss_manager_restores_lagrangian_training_state():
    source = OPFLossManager(
        loss_type="augmented_lagrangian",
        lagrangian_config={"mu_0": 0.2, "warmup_epochs": 0, "verbose": False},
    )
    source.lagrangian.lambda_k = torch.tensor([0.25, -0.5])
    source.lagrangian.mu_k = 0.75
    state = source.training_state_dict()

    restored = OPFLossManager(
        loss_type="augmented_lagrangian",
        lagrangian_config={"mu_0": 0.2, "warmup_epochs": 0, "verbose": False},
    )
    restored.load_training_state_dict(state)
    restored._restore_pending_lagrangian_state()

    assert restored.lagrangian.mu_k == 0.75
    assert torch.equal(restored.lagrangian.lambda_k, torch.tensor([0.25, -0.5]))


def test_loss_manager_runs_heterogeneous_acopf_forward_and_backward():
    predictions = {
        "bus": torch.tensor([[0.0, 1.0], [0.0, 1.0]], requires_grad=True),
        "generator": torch.tensor([[0.5, 0.0]], requires_grad=True),
    }
    manager = OPFLossManager(
        loss_type="augmented_lagrangian",
        lagrangian_config={"mu_0": 0.1, "normalize_by_size": True},
    )

    loss, info = manager.compute_loss(predictions, _two_bus_batch())
    loss.backward()

    assert torch.isfinite(loss)
    assert info["n_constraints"] == 5
    assert torch.isfinite(predictions["bus"].grad).all()
    assert torch.isfinite(predictions["generator"].grad).all()
