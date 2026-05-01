"""
Tests for the Modeler class in example/opf/evaluate_opf_constraint.py

These tests include:
- Lightweight unit tests for Modeler static methods that do not require network.
- Integration-style tests that download the model config and SafeTensors from HuggingFace.
  Integration tests are skipped by default; enable with RUN_LUMINA_INTEGRATION=1.

Additional tests:
- Pickling test for a Modeler instance that does not contain an attached model (ensures instance state is picklable).
- build_constraint_evaluator test using a minimal dummy batch object to verify an ACOPFConstraintEvaluator is constructed.

To run only the lightweight tests (no network):
    pytest tests/test_modeler.py::test_convert_checkpoint_key_to_model_key -q

To run integration tests (network + HF downloads):
    RUN_LUMINA_INTEGRATION=1 HF_TOKEN=<your-token> pytest -q
"""

import json
import os
import pickle

import pytest
import torch
from types import SimpleNamespace

from lumina.evaluator.opf.utils import Modeler
from lumina.evaluator.opf.evaluator import ACOPFConstraintEvaluator
from lumina.trainer.opf.utils import build_hetero_model_spec, resolve_hetero_model_type

# The integration tests below will perform network downloads; disabled by default.
RUN_INTEGRATION = os.getenv("RUN_LUMINA_INTEGRATION", "") == "1"


# --------------------
# Lightweight unit tests (no network)
# --------------------
def test_convert_checkpoint_key_to_model_key():
    """Test static conversion of checkpoint keys into model keys."""
    inp = "<bus___ac_line___weight>"
    expected = "('bus', 'ac_line', 'weight')"
    out = Modeler.convert_checkpoint_key_to_model_key(inp)
    assert out == expected


def test_derive_voltage_limits_with_and_without_columns():
    """Test voltage limit derivation when bus_x contains vmin/vmax columns and when it doesn't."""
    device = torch.device("cpu")

    # case: bus_x has vmin/vmax in columns 1 and 2
    bus_x = torch.tensor([
        [0.0, 0.97, 1.03],
        [0.0, 0.96, 1.04],
    ], dtype=torch.float32)
    vlims = Modeler.derive_voltage_limits(bus_x, device)
    assert torch.allclose(vlims["vmin"], torch.tensor([0.97, 0.96], device=device))
    assert torch.allclose(vlims["vmax"], torch.tensor([1.03, 1.04], device=device))

    # case: bus_x is None -> default values of 0.95 and 1.05 with length 0
    vlims_none = Modeler.derive_voltage_limits(None, device)
    assert "vmin" in vlims_none and "vmax" in vlims_none
    assert vlims_none["vmin"].numel() == 0
    assert vlims_none["vmax"].numel() == 0


def test_derive_generation_limits_with_and_without_columns():
    """Test generation limits derivation with explicit columns and with missing columns."""
    device = torch.device("cpu")

    # Provide gen_x with enough columns (7 columns expected by heuristic)
    gen_x = torch.zeros((2, 7), dtype=torch.float32)
    # set pmin (col 2), pmax (col 3), qmin (col 5), qmax (col 6)
    gen_x[0, 2] = -1.0
    gen_x[0, 3] = 2.5
    gen_x[0, 5] = -0.8
    gen_x[0, 6] = 0.9

    gen_x[1, 2] = 0.0
    gen_x[1, 3] = 1.5
    gen_x[1, 5] = -0.5
    gen_x[1, 6] = 0.5

    glims = Modeler.derive_generation_limits(gen_x, device)
    assert isinstance(glims, dict)
    assert torch.allclose(glims["pmin"], torch.tensor([-1.0, 0.0], device=device))
    assert torch.allclose(glims["pmax"], torch.tensor([2.5, 1.5], device=device))
    assert torch.allclose(glims["qmin"], torch.tensor([-0.8, -0.5], device=device))
    assert torch.allclose(glims["qmax"], torch.tensor([0.9, 0.5], device=device))

    # Case: gen_x is None or empty -> returns None
    assert Modeler.derive_generation_limits(None, device) is None
    assert Modeler.derive_generation_limits(torch.empty((0,)), device) is None


def test_to_float32_converts_batch_fields():
    """
    Test Modeler.to_float32 converts integer/float tensors to float32 for node and edge features.

    Create a minimal dummy batch object that exposes node_types, edge_types and supports indexed access.
    """
    class DummyNode(SimpleNamespace):
        pass

    class DummyBatch:
        def __init__(self):
            # two node types: 'bus' and 'generator'
            self.node_types = ['bus', 'generator']
            self.edge_types = []
            # bus with integer x and y tensors
            self.bus = DummyNode(
                x=torch.tensor([[1, 2], [3, 4]], dtype=torch.int64),
                y=torch.tensor([1, 0], dtype=torch.int64)
            )
            # generator with float32 already
            self.generator = DummyNode(
                x=torch.tensor([[0.1, 0.2]], dtype=torch.float32),
                y=None
            )

        def __getitem__(self, key):
            return getattr(self, key)

    batch = DummyBatch()
    batch_out = Modeler.to_float32(batch)

    assert batch_out.bus.x.dtype == torch.float32
    assert batch_out.bus.y.dtype == torch.float32
    assert batch_out.generator.x.dtype == torch.float32


def test_modeler_pickling_without_model_preserves_state():
    """
    Ensure Modeler instances without an attached model (model==None) are picklable and restore attributes.

    This verifies that the instance-level attributes (device, slack_bus_indices, etc.)
    can be serialized/deserialized via pickle when no unpicklable attributes (like a built PyTorch model)
    are present.
    """
    device = torch.device("cpu")
    m = Modeler(device=device, slack_bus_indices="0,2")
    # Ensure no model is attached
    assert m.model is None
    data = pickle.dumps(m)
    m2 = pickle.loads(data)
    assert isinstance(m2, Modeler)
    assert m2.device == m.device
    assert m2.slack_bus_indices == m.slack_bus_indices


def test_build_constraint_evaluator_returns_evaluator():
    """
    Test constructing an ACOPFConstraintEvaluator from a minimal dummy batch object.

    The dummy batch provides just enough data (bus.x) for _derive_voltage_limits and
    the line-params derivation will return None due to missing edge types.
    """
    device = torch.device("cpu")
    m = Modeler(device=device, slack_bus_indices="0")
    # Create a minimal batch with bus.x and minimal structure expected by Modeler
    class DummyNode(SimpleNamespace):
        pass

    class DummyBatch:
        def __init__(self):
            self.node_types = ["bus"]
            self.edge_types = []
            # bus node features: 2 buses, with columns (placeholder, vmin, vmax)
            self.bus = DummyNode(x=torch.tensor([[0.0, 0.96, 1.04], [0.0, 0.97, 1.03]], dtype=torch.float32))
        def __getitem__(self, key):
            return getattr(self, key)

    batch = DummyBatch()
    evaluator = m.build_constraint_evaluator(batch, device=device, cache_key=None)
    assert isinstance(evaluator, ACOPFConstraintEvaluator)
    # Ensure the evaluator has expected attributes set (voltage_limits present)
    assert hasattr(evaluator, "voltage_limits")
    assert "vmin" in evaluator.voltage_limits and "vmax" in evaluator.voltage_limits


def test_resolve_hetero_model_type_prefers_model_class_path():
    """Model class path should take precedence over model type when both are present."""
    resolved = resolve_hetero_model_type(
        model_type="HeteroGNN",
        model_class_path="lumina.model.opf.hetero_model.HGT",
        default="HeteroGNN",
    )
    assert resolved == "HGT"


def test_build_hetero_model_spec_uses_arch_specific_config():
    """HGT kwargs should come from HGT config, not HeteroGNN fallback, when available."""
    _, model_kwargs, _, used_fallback = build_hetero_model_spec(
        model_type="HGT",
        metadata=(["bus", "generator"], [("bus", "ac_line", "bus")]),
        input_channels={"bus": 7, "generator": 11},
        models_config={
            "HeteroGNN": {"hidden_channels": 16, "num_layers": 1, "backend": "gcn"},
            "HGT": {"hidden_channels": 128, "num_layers": 4, "num_heads": 8, "backend": "sage"},
        },
        out_channels=2,
    )
    assert used_fallback is False
    assert model_kwargs["hidden_channels"] == 128
    assert model_kwargs["num_layers"] == 4
    assert model_kwargs["num_heads"] == 8
    assert model_kwargs["backend"] == "sage"


# --------------------
# Integration-style tests (network; skipped unless enabled)
# --------------------
@pytest.fixture(scope="module")
def hf_token():
    """Optional HF token for authenticated downloads."""
    return os.getenv("HF_TOKEN", None)


@pytest.fixture(scope="module")
def config_data(hf_token):
    """Download config.json; skipped if RUN_INTEGRATION not set."""
    if not RUN_INTEGRATION:
        pytest.skip("Integration tests disabled. Set RUN_LUMINA_INTEGRATION=1 to enable.")
    from huggingface_hub import hf_hub_download
    config_path = hf_hub_download(repo_id="argonne/LUMINA-1B", filename="config.json", token=hf_token)
    with open(config_path, "r") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def state_dict(hf_token):
    """Download model.safetensors; skipped if RUN_INTEGRATION not set."""
    if not RUN_INTEGRATION:
        pytest.skip("Integration tests disabled. Set RUN_LUMINA_INTEGRATION=1 to enable.")
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    safetensors_path = hf_hub_download(repo_id="argonne/LUMINA-1B", filename="model.safetensors", token=hf_token)
    return load_file(safetensors_path)


@pytest.fixture(scope="module")
def device():
    """Select device for integration tests."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def modeler(config_data, state_dict, device, hf_token):
    """Create and load a Modeler instance using downloaded config and weights."""
    if not RUN_INTEGRATION:
        pytest.skip("Integration tests disabled. Set RUN_LUMINA_INTEGRATION=1 to enable.")
    m = Modeler(device=device, slack_bus_indices="0,1")
    m.load_model(config_data=config_data, state_dict=state_dict)
    return m


@pytest.fixture(scope="module")
def loader(config_data):
    """Create a DataLoader for one-batch retrieval (integration)."""
    if not RUN_INTEGRATION:
        pytest.skip("Integration tests disabled. Set RUN_LUMINA_INTEGRATION=1 to enable.")
    from lumina.dataset.opf.opf_dataset import OPFDataset
    from lumina.loader.opf.opf_loader import DataLoader
    case_name = config_data.get("case_name", "pglib_opf_case14_ieee")
    dataset = OPFDataset(root="./opf_data", case_name=case_name)
    return DataLoader(dataset, batch_size=1, shuffle=False)


def test_modeler_load_populates_model(modeler):
    """Integration: load_model must populate model and config_data."""
    assert modeler.model is not None
    assert isinstance(modeler.config_data, dict)


def test_run_predictions_returns_one_batch(modeler, loader):
    """Integration: run one batch and validate structure of the result."""
    preds = modeler.run_predictions(loader, max_batches=1)
    assert isinstance(preds, list)
    assert len(preds) == 1
    predictions_cpu, batch_cpu = preds[0]
    assert isinstance(predictions_cpu, dict)
    assert batch_cpu is not None


def test_evaluate_from_predictions_returns_stats(modeler, loader):
    """Integration: evaluate predictions and verify returned stats format."""
    pred_pairs = modeler.run_predictions(loader, max_batches=1)
    stats = modeler.evaluate_from_predictions(pred_pairs, normalize=True, cache_key="pglib_opf_case14_ieee")
    assert isinstance(stats, dict)
    if len(stats) > 0:
        for name, entry in stats.items():
            assert "mean" in entry and "var" in entry and "weight" in entry
            assert isinstance(entry["mean"], float)
            assert isinstance(entry["var"], float)
            assert isinstance(entry["weight"], float)
