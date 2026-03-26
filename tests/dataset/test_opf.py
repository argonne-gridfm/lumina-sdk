from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from lumina.dataset.opf.opf_dataset import build_heterodata_from_grid, process_json_file


_DATASET_DIR = Path(__file__).resolve().parent
_EXAMPLE_JSON_PATH = _DATASET_DIR / "example_0.json"
_PROCESS_JSON_SNAPSHOT_PATH = _DATASET_DIR / "example_0.process_json_file.pt"


def _load_process_json_snapshot() -> HeteroData:
    snapshot = torch.load(
        _PROCESS_JSON_SNAPSHOT_PATH,
        map_location="cpu",
        weights_only=False,
    )
    assert isinstance(snapshot, HeteroData)
    return snapshot


def _assert_value_equal(actual: Any, expected: Any, *, path: str) -> None:
    if isinstance(expected, Tensor):
        assert isinstance(actual, Tensor), (
            f"{path}: expected Tensor, got {type(actual).__name__}"
        )
        assert actual.dtype == expected.dtype, (
            f"{path}: dtype mismatch {actual.dtype} != {expected.dtype}"
        )
        assert tuple(actual.shape) == tuple(expected.shape), (
            f"{path}: shape mismatch {tuple(actual.shape)} != {tuple(expected.shape)}"
        )
        torch.testing.assert_close(
            actual,
            expected,
            rtol=0,
            atol=0,
            equal_nan=True,
            msg=lambda msg: f"{path}: {msg}",
        )
        return

    assert type(actual) is type(expected), (
        f"{path}: type mismatch {type(actual).__name__} != {type(expected).__name__}"
    )
    assert actual == expected, f"{path}: value mismatch {actual!r} != {expected!r}"


def _assert_store_equal(actual_store: Any, expected_store: Any, *, path: str) -> None:
    actual_keys = sorted(actual_store.keys())
    expected_keys = sorted(expected_store.keys())
    assert actual_keys == expected_keys, (
        f"{path}: key mismatch {actual_keys} != {expected_keys}"
    )
    for key in expected_keys:
        _assert_value_equal(actual_store[key], expected_store[key], path=f"{path}.{key}")


def assert_heterodata_exact_match(actual: HeteroData, expected: HeteroData) -> None:
    assert isinstance(actual, HeteroData)
    assert isinstance(expected, HeteroData)
    assert actual.node_types == expected.node_types
    assert actual.edge_types == expected.edge_types
    assert len(actual.stores) == len(expected.stores)

    _assert_store_equal(actual._global_store, expected._global_store, path="global")

    for node_type in expected.node_types:
        _assert_store_equal(actual[node_type], expected[node_type], path=f"node[{node_type}]")

    for edge_type in expected.edge_types:
        edge_label = "__".join(edge_type)
        _assert_store_equal(actual[edge_type], expected[edge_type], path=f"edge[{edge_label}]")


def _drop_targets(hdata: HeteroData) -> HeteroData:
    out = copy.deepcopy(hdata)
    if hasattr(out["bus"], "y"):
        del out["bus"].y
    if hasattr(out["generator"], "y"):
        del out["generator"].y
    if hasattr(out["bus", "ac_line", "bus"], "edge_label"):
        del out["bus", "ac_line", "bus"].edge_label
    if hasattr(out["bus", "transformer", "bus"], "edge_label"):
        del out["bus", "transformer", "bus"].edge_label
    return out


def test_process_json_file_example_0_matches_snapshot() -> None:
    actual = process_json_file(str(_EXAMPLE_JSON_PATH))
    expected = _load_process_json_snapshot()

    assert_heterodata_exact_match(actual, expected)


def test_build_heterodata_from_grid_with_solution_matches_process_json_snapshot() -> None:
    with open(_EXAMPLE_JSON_PATH) as f:
        obj = json.load(f)

    actual = build_heterodata_from_grid(obj["grid"], obj["metadata"], obj["solution"])
    expected = _load_process_json_snapshot()

    assert_heterodata_exact_match(actual, expected)


def test_build_heterodata_from_grid_without_solution_matches_snapshot_without_targets() -> None:
    with open(_EXAMPLE_JSON_PATH) as f:
        obj = json.load(f)

    actual = build_heterodata_from_grid(obj["grid"], obj["metadata"])
    expected = _drop_targets(_load_process_json_snapshot())

    assert_heterodata_exact_match(actual, expected)
