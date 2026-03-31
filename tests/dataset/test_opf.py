from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from lumina.dataset.opf.opf_dataset import build_heterodata_from_grid, process_json_file
from lumina.dataset.opf.opf_on_disk_dataset import OPFOnDiskDataset


_DATASET_DIR = Path(__file__).resolve().parent
_EXAMPLE_JSON_PATH = _DATASET_DIR / "pglib_opf_case2000_goc_example_0.json"
_PROCESS_JSON_SNAPSHOT_PATH = _DATASET_DIR / "pglib_opf_case2000_goc_example_0.process_json_file.pt"
_ON_DISK_SNAPSHOT_PATH = _DATASET_DIR / "pglib_opf_case118_ieee_group_0_example_0.opf_on_disk_dataset.pt"
_CASE118_TAR_PATH = _DATASET_DIR / "pglib_opf_case118_ieee_0.tar.gz"


def _load_snapshot(path: Path) -> HeteroData:
    snapshot = torch.load(
        path,
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


def _build_semantic_opf_payload() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    grid = {
        "context": [[[100.0]]],
        "nodes": {
            "bus": [
                [138.0, 1.0, 0.9, 1.1],
                [230.0, 3.0, 0.95, 1.05],
            ],
            "generator": [
                [100.0, 0.4, 0.1, 0.9, 0.05, -0.2, 0.3, 1.0, 1.5, 2.5, 3.5],
            ],
            "load": [
                [0.7, 0.2],
            ],
            "shunt": [
                [0.01, 0.02],
            ],
        },
        "edges": {
            "ac_line": {
                "senders": [0],
                "receivers": [1],
                "features": [[-0.5, 0.5, 0.01, 0.02, 0.03, 0.04, 100.0, 101.0, 102.0]],
            },
            "transformer": {
                "senders": [1],
                "receivers": [0],
                "features": [[-0.4, 0.4, 0.11, 0.12, 110.0, 111.0, 112.0, 1.0, 0.0, 0.13, 0.14]],
            },
            "generator_link": {
                "senders": [0],
                "receivers": [1],
            },
            "load_link": {
                "senders": [0],
                "receivers": [0],
            },
            "shunt_link": {
                "senders": [0],
                "receivers": [1],
            },
        },
    }
    metadata = {
        "objective": 321.0,
    }
    solution = {
        "nodes": {
            "bus": [
                [0.01, 1.02],
                [-0.02, 0.99],
            ],
            "generator": [
                [0.8, 0.1],
            ],
        },
        "edges": {
            "ac_line": {
                "features": [[0.7, 0.08, -0.7, -0.08]],
            },
            "transformer": {
                "features": [[0.5, 0.06, -0.5, -0.06]],
            },
        },
    }
    return grid, metadata, solution


def test_process_json_file_example_0_matches_snapshot() -> None:
    actual = process_json_file(str(_EXAMPLE_JSON_PATH))
    expected = _load_snapshot(_PROCESS_JSON_SNAPSHOT_PATH)

    assert_heterodata_exact_match(actual, expected)


def test_build_heterodata_from_grid_with_solution_matches_process_json_snapshot() -> None:
    with open(_EXAMPLE_JSON_PATH) as f:
        obj = json.load(f)

    actual = build_heterodata_from_grid(obj["grid"], obj["metadata"], obj["solution"])
    expected = _load_snapshot(_PROCESS_JSON_SNAPSHOT_PATH)

    assert_heterodata_exact_match(actual, expected)


def test_build_heterodata_from_grid_without_solution_matches_snapshot_without_targets() -> None:
    with open(_EXAMPLE_JSON_PATH) as f:
        obj = json.load(f)

    actual = build_heterodata_from_grid(obj["grid"], obj["metadata"])
    expected = _drop_targets(_load_snapshot(_PROCESS_JSON_SNAPSHOT_PATH))

    assert_heterodata_exact_match(actual, expected)


def test_build_heterodata_from_grid_semantic_fields_and_links() -> None:
    grid, metadata, solution = _build_semantic_opf_payload()

    data = build_heterodata_from_grid(grid, metadata, solution)

    assert data.baseMVA == 100.0
    assert data.objective.item() == 321.0

    assert data["bus"].x.shape == (2, 7)
    torch.testing.assert_close(
        data["bus"].x,
        torch.tensor([
            [138.0, 0.9, 1.1, 1.0, 0.0, 0.0, 0.0],
            [230.0, 0.95, 1.05, 0.0, 0.0, 1.0, 0.0],
        ], dtype=data["bus"].x.dtype),
        rtol=0,
        atol=0,
    )

    torch.testing.assert_close(
        data["bus", "ac_line", "bus"].edge_index,
        torch.tensor([[0], [1]], dtype=data["bus", "ac_line", "bus"].edge_index.dtype),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        data["bus", "transformer", "bus"].edge_index,
        torch.tensor([[1], [0]], dtype=data["bus", "transformer", "bus"].edge_index.dtype),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        data["generator", "generator_link", "bus"].edge_index,
        torch.tensor([[0], [1]], dtype=data["generator", "generator_link", "bus"].edge_index.dtype),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        data["bus", "generator_link", "generator"].edge_index,
        torch.tensor([[1], [0]], dtype=data["bus", "generator_link", "generator"].edge_index.dtype),
        rtol=0,
        atol=0,
    )


def test_build_heterodata_from_grid_without_solution_omits_targets_semantically() -> None:
    grid, metadata, _ = _build_semantic_opf_payload()

    data = build_heterodata_from_grid(grid, metadata)

    assert not hasattr(data["bus"], "y")
    assert not hasattr(data["generator"], "y")
    assert not hasattr(data["bus", "ac_line", "bus"], "edge_label")
    assert not hasattr(data["bus", "transformer", "bus"], "edge_label")


def test_build_heterodata_from_grid_has_expected_schema_and_dtype_categories() -> None:
    grid, metadata, solution = _build_semantic_opf_payload()

    data = build_heterodata_from_grid(grid, metadata, solution)

    assert data["generator"].x.shape == (1, 11)
    assert data["load"].x.shape == (1, 2)
    assert data["shunt"].x.shape == (1, 2)
    assert data["bus"].y.shape == (2, 2)
    assert data["generator"].y.shape == (1, 2)
    assert data["bus", "ac_line", "bus"].edge_attr.shape == (1, 9)
    assert data["bus", "ac_line", "bus"].edge_label.shape == (1, 4)
    assert data["bus", "transformer", "bus"].edge_attr.shape == (1, 11)
    assert data["bus", "transformer", "bus"].edge_label.shape == (1, 4)

    assert data["bus"].x.dtype.is_floating_point
    assert data["generator"].x.dtype.is_floating_point
    assert data["load"].x.dtype.is_floating_point
    assert data["shunt"].x.dtype.is_floating_point
    assert data["bus"].y.dtype.is_floating_point
    assert data["generator"].y.dtype.is_floating_point
    assert data["bus", "ac_line", "bus"].edge_attr.dtype.is_floating_point
    assert data["bus", "ac_line", "bus"].edge_label.dtype.is_floating_point
    assert not data["bus", "ac_line", "bus"].edge_index.dtype.is_floating_point
    assert not data["generator", "generator_link", "bus"].edge_index.dtype.is_floating_point


def test_process_json_file_returns_none_for_malformed_json(tmp_path) -> None:
    bad_json_path = tmp_path / "bad.json"
    bad_json_path.write_text("{ not valid json")

    data = process_json_file(str(bad_json_path))

    assert data is None


def test_process_json_file_raises_for_missing_required_solution_key(tmp_path) -> None:
    grid, metadata, _ = _build_semantic_opf_payload()
    missing_solution_path = tmp_path / "missing_solution.json"
    missing_solution_path.write_text(json.dumps({
        "grid": grid,
        "metadata": metadata,
    }))

    with pytest.raises(KeyError, match="solution"):
        process_json_file(str(missing_solution_path))


def test_opf_on_disk_dataset_processes_json_file_like_process_json_file(tmp_path) -> None:
    raw_dir = tmp_path / "OPFData" / "raw" / "dataset_release_1"
    raw_dir.mkdir(parents=True)
    shutil.copy(_CASE118_TAR_PATH, raw_dir / "pglib_opf_case118_ieee_0.tar.gz")

    actual = OPFOnDiskDataset(
        root=str(tmp_path),
        case_name="pglib_opf_case118_ieee",
        group_id=0,
        log=False,
        write_batch_size=1,
    ).get(0)
    expected = _load_snapshot(_ON_DISK_SNAPSHOT_PATH)

    assert_heterodata_exact_match(actual, expected)
