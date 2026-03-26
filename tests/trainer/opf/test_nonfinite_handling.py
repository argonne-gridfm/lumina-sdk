import sys
import types

import torch
import pytest
from torch_geometric.data import HeteroData


def _install_pandapower_stub():
    if "pandapower" in sys.modules:
        return
    pp_module = types.ModuleType("pandapower")
    pp_module.converter = types.SimpleNamespace(from_mpc=lambda *args, **kwargs: None, pypower=types.SimpleNamespace(to_ppc=lambda *args, **kwargs: {}))
    sys.modules["pandapower"] = pp_module

    pypower_module = types.ModuleType("pandapower.pypower")
    sys.modules["pandapower.pypower"] = pypower_module

    constants = {
        "idx_brch": ["BR_B", "BR_R", "BR_X", "F_BUS", "RATE_A", "RATE_B", "RATE_C", "SHIFT", "T_BUS", "TAP"],
        "idx_bus": ["BASE_KV", "BS", "BUS_AREA", "BUS_I", "BUS_TYPE", "GS", "PD", "QD", "VA", "VM", "VMAX", "VMIN", "ZONE"],
        "idx_gen": ["GEN_BUS", "GEN_STATUS", "MBASE", "PG", "PMAX", "PMIN", "QG", "QMAX", "QMIN", "VG"],
    }
    for module_suffix, names in constants.items():
        module_name = f"pandapower.pypower.{module_suffix}"
        module = types.ModuleType(module_name)
        for idx, name in enumerate(names):
            setattr(module, name, idx)
        sys.modules[module_name] = module


_install_pandapower_stub()

from lumina.trainer.opf import trainer as trainer_module


def _build_trainer_stub(*, fail_on_nonfinite=False, grad_clip_val=None, grad_clip_algo="norm"):
    stub = trainer_module.BaseOPFTrainer.__new__(trainer_module.BaseOPFTrainer)
    stub.global_rank = 0
    stub.global_step = 7
    stub.fail_on_nonfinite = fail_on_nonfinite
    stub.nonfinite_loss_skips = 0
    stub.nonfinite_grad_skips = 0
    stub.grad_clip_val = grad_clip_val
    stub.grad_clip_algo = grad_clip_algo
    stub.model = torch.nn.Linear(2, 1)
    stub.optimizer = torch.optim.SGD(stub.model.parameters(), lr=0.1)
    return stub


def test_nonfinite_loss_skip_path_zeros_gradients_and_increments_counter():
    trainer = _build_trainer_stub(fail_on_nonfinite=False)

    loss = trainer.model.weight.sum() * torch.tensor(float("nan"))
    skipped = trainer_module.BaseOPFTrainer._handle_nonfinite_loss(trainer, loss, batch_idx=3, case_name="caseA")

    assert skipped is True
    assert trainer.nonfinite_loss_skips == 1
    assert all(param.grad is None for param in trainer.model.parameters())


def test_nonfinite_loss_fail_fast_when_enabled():
    trainer = _build_trainer_stub(fail_on_nonfinite=True)

    loss = trainer.model.weight.sum() * torch.tensor(float("inf"))
    with pytest.raises(RuntimeError, match="Non-finite loss"):
        trainer_module.BaseOPFTrainer._handle_nonfinite_loss(trainer, loss, batch_idx=0)


def test_nonfinite_gradient_skips_step_when_failfast_disabled():
    trainer = _build_trainer_stub(fail_on_nonfinite=False, grad_clip_val=None)

    for parameter in trainer.model.parameters():
        parameter.grad = torch.full_like(parameter, float("nan"))

    finite = trainer_module.BaseOPFTrainer._ensure_finite_gradients(trainer, batch_idx=2, case_name="caseB")

    assert finite is False
    assert trainer.nonfinite_grad_skips == 1
    assert all(param.grad is None for param in trainer.model.parameters())


def test_nonfinite_gradient_fail_fast_when_enabled():
    trainer = _build_trainer_stub(fail_on_nonfinite=True, grad_clip_val=None)

    for parameter in trainer.model.parameters():
        parameter.grad = torch.full_like(parameter, float("inf"))

    with pytest.raises(RuntimeError, match="Non-finite gradients"):
        trainer_module.BaseOPFTrainer._ensure_finite_gradients(trainer, batch_idx=5)


def test_nonfinite_gradient_detected_by_clip_grad_norm():
    trainer = _build_trainer_stub(fail_on_nonfinite=False, grad_clip_val=1.0, grad_clip_algo="norm")

    for parameter in trainer.model.parameters():
        parameter.grad = torch.full_like(parameter, float("nan"))

    finite = trainer_module.BaseOPFTrainer._ensure_finite_gradients(trainer, batch_idx=6)

    assert finite is False
    assert trainer.nonfinite_grad_skips == 1


def test_build_model_passes_hgt_dropout_from_model_config(monkeypatch):
    trainer = trainer_module.BaseOPFTrainer.__new__(trainer_module.BaseOPFTrainer)
    trainer.global_rank = 1
    trainer.device = torch.device("cpu")
    trainer.local_rank = 0
    trainer.model_type = "HGT"
    trainer.config = {
        "models": {
            "HGT": {
                "hidden_channels": 8,
                "num_layers": 2,
                "num_heads": 2,
                "dropout": 0.35,
            }
        }
    }

    sample = HeteroData()
    sample["bus"].x = torch.randn(3, 4)
    sample["generator"].x = torch.randn(2, 5)

    metadata = {
        "nodes": {"bus": 4, "generator": 5},
        "edges": {("bus", "generator_link", "generator"): 0},
    }

    monkeypatch.setattr(trainer_module, "initialize_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(trainer_module, "DDP", lambda model, **kwargs: types.SimpleNamespace(module=model))

    model = trainer_module.BaseOPFTrainer._build_model(trainer, sample, metadata, per_node_output_size=2)

    assert isinstance(model.module, trainer_module.HGT)
    assert trainer.model_kwargs["dropout"] == 0.35
    assert model.module.dropout == pytest.approx(0.35)
