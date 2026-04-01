import copy
import json
import os
import os.path as osp
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple

import h5py
import numpy as np
import pytest
import torch
from torch import Tensor
from torch_geometric.data import HeteroData

from lumina.dataset.opf.contingency import (
    ParsedContingency,
    add_slack_generators,
    apply_contingency,
    parse_contingency,
)
from lumina.dataset.opf.opf_dataset import (
    OPFDataset,
    build_heterodata_from_grid,
    process_hdf5_scenario,
    process_json_file,
)
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


def _build_semantic_opf_payload() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
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


def create_mock_scenario_dict():
    """Helper to create a nested dict matching the OPF JSON schema."""
    return {
        "grid": {
            "context": 100.0,
            "nodes": {
                "bus": [[1.0, 1, 0.9, 1.1]],
                "generator": [[100.0, 50.0, 0.0, 100.0, 20.0, -10.0, 30.0, 1.0, 0.01, 1.0, 0.0]],
                "load": [[20.0, 10.0]],
                "shunt": [[0.0, 0.1]],
            },
            "edges": {
                "ac_line": {
                    "senders": [0],
                    "receivers": [0],
                    "features": [[-0.5, 0.5, 0.0, 0.0, 0.01, 0.05, 100.0, 110.0, 120.0]],
                },
                "generator_link": {"senders": [0], "receivers": [0]},
                "load_link": {"senders": [0], "receivers": [0]},
                "shunt_link": {"senders": [0], "receivers": [0]},
                "transformer": {"senders": [], "receivers": [], "features": []},
            },
        },
        "solution": {
            "nodes": {
                "bus": [[0.0, 1.0]],
                "generator": [[50.0, 20.0]],
            },
            "edges": {
                "ac_line": {
                    "features": [[10.0, 5.0, 10.0, 5.0]],
                },
                "transformer": {
                    "features": [],
                },
            },
        },
        "metadata": {
            "objective": 1234.5,
        },
    }


def save_as_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f)


def save_as_hdf5(data, path):
    with h5py.File(path, "w") as f:
        scenario = f.create_group("scenario_1")
        grid = scenario.create_group("grid")

        # Grid context
        ctx = grid.create_group("context")
        ctx.create_dataset("baseMVA", data=np.array([data["grid"]["context"]]))

        # Nodes
        nodes = grid.create_group("nodes")

        gen_json = np.array(data["grid"]["nodes"]["generator"])
        gen_hdf5 = np.zeros((gen_json.shape[0], 10))
        # JSON: mbase(0), pg(1), pmin(2), pmax(3), qg(4), qmin(5), qmax(6), vg(7), cost_c2(8), cost_c1(9), cost_c0(10)
        # H5: pmax(0), pmin(1), qmax(2), qmin(3), cost_c2(4), cost_c1(5), cost_c0(6), vg(7), mbase(8), gen_status(9)
        gen_hdf5[:, 8] = gen_json[:, 0]
        gen_hdf5[:, 0] = gen_json[:, 3]
        gen_hdf5[:, 1] = gen_json[:, 2]
        gen_hdf5[:, 2] = gen_json[:, 6]
        gen_hdf5[:, 3] = gen_json[:, 5]
        gen_hdf5[:, 4] = gen_json[:, 8]
        gen_hdf5[:, 5] = gen_json[:, 9]
        gen_hdf5[:, 6] = gen_json[:, 10]
        gen_hdf5[:, 7] = gen_json[:, 7]
        gen_hdf5[:, 9] = 1.0

        nodes.create_dataset("generator", data=gen_hdf5.T)
        nodes.create_dataset("load", data=np.array(data["grid"]["nodes"]["load"]).T)
        shunt_json = np.array(data["grid"]["nodes"]["shunt"])
        shunt_hdf5 = np.zeros_like(shunt_json)
        shunt_hdf5[:, 0] = shunt_json[:, 1]
        shunt_hdf5[:, 1] = shunt_json[:, 0]
        nodes.create_dataset("shunt", data=shunt_hdf5.T)

        # Bus features in HDF5: [vmin, vmax, zone, area, bus_type]
        # JSON bus: [base_kv, bus_type, vmin, vmax]
        bus_json = np.array(data["grid"]["nodes"]["bus"])
        bus_hdf5 = np.zeros((bus_json.shape[0], 5))
        bus_hdf5[:, 0] = bus_json[:, 2]
        bus_hdf5[:, 1] = bus_json[:, 3]
        bus_hdf5[:, 4] = bus_json[:, 1]
        nodes.create_dataset("bus", data=bus_hdf5.T)

        # Edges
        edges = grid.create_group("edges")
        ac = edges.create_group("ac_line")
        ac.create_dataset("senders", data=np.array(data["grid"]["edges"]["ac_line"]["senders"]))
        ac.create_dataset("receivers", data=np.array(data["grid"]["edges"]["ac_line"]["receivers"]))

        ac_json = np.array(data["grid"]["edges"]["ac_line"]["features"])
        # JSON ACLine: angmin(0), angmax(1), b_fr(2), b_to(3), br_r(4), br_x(5), rate_a(6), rate_b(7), rate_c(8)
        # H5 ACLine: angmin(0), angmax(1), br_r(2), br_x(3), b_fr(4), b_to(5), rate_a(6), rate_b(7), rate_c(8), br_status(9)
        ac_hdf5 = np.zeros((ac_json.shape[0], 10))
        ac_hdf5[:, 0] = ac_json[:, 0]
        ac_hdf5[:, 1] = ac_json[:, 1]
        ac_hdf5[:, 2] = ac_json[:, 4]
        ac_hdf5[:, 3] = ac_json[:, 5]
        ac_hdf5[:, 4] = ac_json[:, 2]
        ac_hdf5[:, 5] = ac_json[:, 3]
        ac_hdf5[:, 6:9] = ac_json[:, 6:9]
        ac_hdf5[:, 9] = 1.0
        ac.create_dataset("features", data=ac_hdf5.T)

        tr = edges.create_group("transformer")
        tr.create_dataset("senders", data=np.array(data["grid"]["edges"]["transformer"]["senders"]))
        tr.create_dataset("receivers", data=np.array(data["grid"]["edges"]["transformer"]["receivers"]))
        tr_json = np.array(data["grid"]["edges"]["transformer"]["features"])
        if tr_json.size > 0:
            # JSON Transformer: angmin(0), angmax(1), br_r(2), br_x(3), rate_a(4), rate_b(5), rate_c(6), tap(7), shift(8), b_fr(9), b_to(10)
            # H5 Transformer: angmin(0), angmax(1), br_r(2), br_x(3), b_fr(4), b_to(5), rate_a(6), rate_b(7), rate_c(8), br_status(9), tap(10), shift(11)
            tr_hdf5 = np.zeros((tr_json.shape[0], 12))
            tr_hdf5[:, 0] = tr_json[:, 0]
            tr_hdf5[:, 1] = tr_json[:, 1]
            tr_hdf5[:, 2] = tr_json[:, 2]
            tr_hdf5[:, 3] = tr_json[:, 3]
            tr_hdf5[:, 4] = tr_json[:, 9]
            tr_hdf5[:, 5] = tr_json[:, 10]
            tr_hdf5[:, 6] = tr_json[:, 4]
            tr_hdf5[:, 7] = tr_json[:, 5]
            tr_hdf5[:, 8] = tr_json[:, 6]
            tr_hdf5[:, 9] = 1.0
            tr_hdf5[:, 10] = tr_json[:, 7]
            tr_hdf5[:, 11] = tr_json[:, 8]
            tr.create_dataset("features", data=tr_hdf5.T)
        else:
            tr.create_dataset("features", data=np.zeros((12, 0)))

        # Virtual links
        gl = edges.create_group("generator_link")
        gl.create_dataset("senders", data=np.array(data["grid"]["edges"]["generator_link"]["senders"]))
        gl.create_dataset("receivers", data=np.array(data["grid"]["edges"]["generator_link"]["receivers"]))

        ll = edges.create_group("load_link")
        ll.create_dataset("senders", data=np.array(data["grid"]["edges"]["load_link"]["senders"]))
        ll.create_dataset("receivers", data=np.array(data["grid"]["edges"]["load_link"]["receivers"]))

        sl = edges.create_group("shunt_link")
        sl.create_dataset("senders", data=np.array(data["grid"]["edges"]["shunt_link"]["senders"]))
        sl.create_dataset("receivers", data=np.array(data["grid"]["edges"]["shunt_link"]["receivers"]))

        # Solution
        sol = scenario.create_group("solution")
        sol_nodes = sol.create_group("nodes")
        sol_nodes.create_dataset("bus", data=np.array(data["solution"]["nodes"]["bus"]).T)
        sol_nodes.create_dataset("generator", data=np.array(data["solution"]["nodes"]["generator"]).T)

        sol_edges = sol.create_group("edges")
        sol_ac = sol_edges.create_group("ac_line")

        sol_ac_json = np.array(data["solution"]["edges"]["ac_line"]["features"])
        # JSON EdgeSolution: pt(0), qt(1), pf(2), qf(3)
        # H5 EdgeSolution: pf(0), qf(1), pt(2), qt(3)
        sol_ac_hdf5 = np.zeros_like(sol_ac_json)
        sol_ac_hdf5[:, 0] = sol_ac_json[:, 2]
        sol_ac_hdf5[:, 1] = sol_ac_json[:, 3]
        sol_ac_hdf5[:, 2] = sol_ac_json[:, 0]
        sol_ac_hdf5[:, 3] = sol_ac_json[:, 1]
        sol_ac.create_dataset("features", data=sol_ac_hdf5.T)

        sol_tr = sol_edges.create_group("transformer")
        sol_tr.create_dataset("features", data=np.zeros((4, 0)))

        # Metadata
        meta = scenario.create_group("metadata")
        meta.attrs["objective"] = data["metadata"]["objective"]


def _write_branch_mapping(directory, mapping):
    """Write a branch_mapping.json file into the given directory."""
    mapping_path = osp.join(directory, "branch_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(mapping, f)


def create_fake_contingency_h5(path, scenario_id="scenario_1"):
    """Create a fake contingency HDF5 file with edges and contingency definitions.

    Grid: 4 buses, 3 generators, 2 loads, 3 ac_line edges, 1 transformer edge.
    Contingency 1: N-1 branch LINE_1_2_1 (trips ac_line edge 0→1)
    Contingency 2: N-1 generator id=2 (trips generator index 1)

    Also writes a branch_mapping.json alongside the HDF5 file.
    """
    n_buses = 4
    n_gens = 3
    n_loads = 2

    with h5py.File(path, "w") as f:
        scenario = f.create_group(scenario_id)

        base_sol = scenario.create_group("base_solution")
        opf_sol = base_sol.create_group("opf")
        sol_nodes = opf_sol.create_group("nodes")
        sol_nodes.create_dataset("bus", data=np.random.rand(2, n_buses))
        sol_nodes.create_dataset("generator", data=np.random.rand(2, n_gens))
        sol_nodes.create_dataset("load", data=np.random.rand(2, n_loads))
        opf_sol.attrs["objective"] = 1111.1

        # contingency definitions
        cg = scenario.create_group("contingencies")
        cg.create_dataset("ids", data=np.array(["1", "2"], dtype="S32"))
        cg.create_dataset("types", data=np.array([0, 1], dtype=np.int8))
        cg.create_dataset("names", data=np.array(["LINE_1_2_1", "GEN_2"], dtype="S64"))

        # post-contingency solutions
        pc = scenario.create_group("post_contingency")

        # Contingency 1: branch LINE_1_2_1 tripped -> ac_line edge 0->1 has zero flow
        cont1 = pc.create_group("contingency_000001")
        cont1.attrs["opf_converged"] = 1
        cont1_opf = cont1.create_group("opf")
        cont1_opf.attrs["objective"] = 1234.5
        c1_nodes = cont1_opf.create_group("nodes")
        c1_nodes.create_dataset("bus", data=np.random.rand(2, n_buses))
        c1_nodes.create_dataset("generator", data=np.random.rand(2, n_gens))
        c1_nodes.create_dataset("load", data=np.random.rand(2, n_loads))
        c1_edges = cont1_opf.create_group("edges")
        c1_ac = c1_edges.create_group("ac_line")
        ac_sol_data = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [10.0, 5.0, -9.8, -4.9],
            [8.0, 3.0, -7.8, -2.9],
        ]).T
        c1_ac.create_dataset("features", data=ac_sol_data)
        c1_tr = c1_edges.create_group("transformer")
        tr_sol_data = np.array([[5.0, 2.0, -4.9, -1.9]]).T
        c1_tr.create_dataset("features", data=tr_sol_data)

        # Contingency 2: generator 2 tripped
        cont2 = pc.create_group("contingency_000002")
        cont2.attrs["opf_converged"] = 1
        cont2_opf = cont2.create_group("opf")
        cont2_opf.attrs["objective"] = 6789.0
        c2_nodes = cont2_opf.create_group("nodes")
        c2_nodes.create_dataset("bus", data=np.random.rand(2, n_buses))
        c2_nodes.create_dataset("generator", data=np.random.rand(2, n_gens))
        c2_nodes.create_dataset("load", data=np.random.rand(2, n_loads))
        c2_edges = cont2_opf.create_group("edges")
        c2_ac = c2_edges.create_group("ac_line")
        c2_ac.create_dataset("features", data=np.random.rand(4, 3))
        c2_tr = c2_edges.create_group("transformer")
        c2_tr.create_dataset("features", data=np.random.rand(4, 1))

        # grid group
        grid = scenario.create_group("grid")
        nodes = grid.create_group("nodes")

        bus_data = np.zeros((5, n_buses))
        bus_data[0, :] = 0.9
        bus_data[1, :] = 1.1
        bus_data[2, :] = 1
        bus_data[3, :] = 1
        bus_data[4, :] = [3, 1, 2, 1]
        nodes.create_dataset("bus", data=bus_data)

        gen_data = np.random.rand(10, n_gens)
        gen_data[9, :] = 1.0
        nodes.create_dataset("generator", data=gen_data)

        load_data = np.random.rand(4, n_loads)
        nodes.create_dataset("load", data=load_data)

        edges = grid.create_group("edges")

        ac_line = edges.create_group("ac_line")
        ac_line.create_dataset("senders", data=np.array([0, 1, 2], dtype=np.int64))
        ac_line.create_dataset("receivers", data=np.array([1, 2, 3], dtype=np.int64))
        ac_features = np.random.rand(10, 3)
        ac_features[9, :] = 1.0
        ac_line.create_dataset("features", data=ac_features)

        transformer = edges.create_group("transformer")
        transformer.create_dataset("senders", data=np.array([0], dtype=np.int64))
        transformer.create_dataset("receivers", data=np.array([3], dtype=np.int64))
        tr_features = np.random.rand(12, 1)
        tr_features[9, :] = 1.0
        transformer.create_dataset("features", data=tr_features)

        gen_link = edges.create_group("generator_link")
        gen_link.create_dataset("senders", data=np.array([0, 1, 2], dtype=np.int64))
        gen_link.create_dataset("receivers", data=np.array([0, 1, 2], dtype=np.int64))

        load_link = edges.create_group("load_link")
        load_link.create_dataset("senders", data=np.array([0, 1], dtype=np.int64))
        load_link.create_dataset("receivers", data=np.array([1, 3], dtype=np.int64))

        shunt_link = edges.create_group("shunt_link")
        shunt_link.create_dataset("senders", data=np.array([], dtype=np.int64))
        shunt_link.create_dataset("receivers", data=np.array([], dtype=np.int64))

        context = grid.create_group("context")
        context.create_dataset("baseMVA", data=np.array([100.0]))

        meta = scenario.create_group("metadata")
        meta.attrs["n_contingencies"] = 2

    branch_mapping = {
        "1": ["ac_line", 0],
        "2": ["ac_line", 1],
        "3": ["ac_line", 2],
        "4": ["transformer", 0],
    }
    _write_branch_mapping(osp.dirname(path), branch_mapping)


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


def test_standalone_hdf5_loading():
    h5_path = "tests/test_data/chunk_0001.h5"
    if not os.path.exists(h5_path):
        pytest.skip(f"Test data {h5_path} not found")

    with h5py.File(h5_path, "r") as f:
        scenario_key = list(f.keys())[0]
        scenario = f[scenario_key]
        data = process_hdf5_scenario(scenario, scenario_key)

    assert data is not None
    if isinstance(data, list):
        data = data[0]
    assert isinstance(data.baseMVA, torch.Tensor)
    assert data.baseMVA.ndim == 1


def test_round_trip_alignment(tmp_path):
    # We try serializing the same data using each schema we know about, and validate
    # that the resulting OPFDatasets are all functionally the same.
    mock_data = create_mock_scenario_dict()
    json_path = str(tmp_path / "temp_test.json")
    h5_path = str(tmp_path / "temp_test.h5")

    save_as_json(mock_data, json_path)
    save_as_hdf5(mock_data, h5_path)

    json_hdata = process_json_file(json_path)
    with h5py.File(h5_path, "r") as f:
        scenario_key = "scenario_1"
        scenario = f[scenario_key]
        hdf5_hdata = process_hdf5_scenario(scenario, scenario_key)

    json_base_mva = torch.as_tensor(json_hdata.baseMVA).view(-1).to(torch.float32)
    hdf5_base_mva = torch.as_tensor(hdf5_hdata.baseMVA).view(-1).to(torch.float32)
    assert torch.allclose(json_base_mva, hdf5_base_mva)
    assert torch.allclose(json_hdata.objective.to(torch.float32), hdf5_hdata.objective.to(torch.float32))

    for node_type in json_hdata.node_types:
        assert node_type in hdf5_hdata.node_types
        if hasattr(json_hdata[node_type], "x"):
            jx = json_hdata[node_type].x.to(torch.float32)
            hx = hdf5_hdata[node_type].x.to(torch.float32)
            assert jx.shape == hx.shape, f"Shape mismatch for {node_type}: JSON {jx.shape}, HDF5 {hx.shape}"
            if node_type != "generator":
                assert torch.allclose(jx, hx, atol=1e-5)
        if hasattr(json_hdata[node_type], "y"):
            jy = json_hdata[node_type].y.to(torch.float32)
            hy = hdf5_hdata[node_type].y.to(torch.float32)
            assert torch.allclose(jy, hy, atol=1e-5)

    for edge_type in json_hdata.edge_types:
        assert edge_type in hdf5_hdata.edge_types
        assert torch.equal(json_hdata[edge_type].edge_index, hdf5_hdata[edge_type].edge_index)

        json_edge = json_hdata[edge_type]
        hdf5_edge = hdf5_hdata[edge_type]

        if "edge_attr" in json_edge.keys() and "edge_attr" in hdf5_edge.keys():
            jx = json_edge["edge_attr"].to(torch.float32)
            hx = hdf5_edge["edge_attr"].to(torch.float32)
            if jx.numel() == 0 and hx.numel() == 0:
                continue
            assert torch.allclose(jx, hx, atol=1e-5)
        if "edge_label" in json_edge.keys() and "edge_label" in hdf5_edge.keys():
            jy = json_edge["edge_label"].to(torch.float32)
            hy = hdf5_edge["edge_label"].to(torch.float32)
            if jy.numel() == 0 and hy.numel() == 0:
                continue
            assert torch.allclose(jy, hy, atol=1e-5)


def test_contingency_loading(tmp_path):
    root = str(tmp_path / "data_contingency_test")
    case_name = "pglib_opf_case14_ieee"
    os.makedirs(root, exist_ok=True)

    raw_dir = osp.join(root, "OPFData", "raw", "dataset_release_1")
    results_dir = osp.join(raw_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    h5_path = osp.join(raw_dir, "task_000001.h5")
    create_fake_contingency_h5(h5_path)

    with h5py.File(h5_path, "r") as f:
        data_list = []
        for scenario_key in f.keys():
            scenario = f[scenario_key]
            res = process_hdf5_scenario(scenario, scenario_key)
            if isinstance(res, list):
                data_list.extend(res)
            elif res is not None:
                data_list.append(res)
    assert len(data_list) == 2
    objectives = sorted([d.objective.item() for d in data_list])
    assert objectives == [1234.5, 6789.0]

    assert data_list[0].n_contingencies == 2
    assert data_list[0]["load"].x.shape[1] == 2
    assert "y" in data_list[0]["load"]
    assert data_list[0]["load"].y.shape[1] == 2

    class MockOPFDataset(OPFDataset):
        @property
        def raw_file_names(self):
            return ["task_000001.h5"]

        def download(self):
            pass

    dataset = MockOPFDataset(root=root, case_name=case_name, group_id=0, force_reload=True)
    assert len(dataset) == 2
    assert dataset[0].objective.item() in [1234.5, 6789.0]
    assert dataset[0].n_contingencies == 2


def _make_heterodata(
    n_buses,
    n_gens,
    gen_bus_map,
    n_loads=0,
    load_bus_map=None,
    load_demands=None,
    load_served=None,
    baseMVA=100.0,
    gen_y=None,
):
    """Helper to create a minimal HeteroData for slack generator tests."""
    hdata = HeteroData()
    hdata["bus"].x = torch.randn(n_buses, 7, dtype=torch.float32)
    hdata["generator"].x = torch.randn(n_gens, 11, dtype=torch.float32)
    hdata.baseMVA = torch.tensor([baseMVA], dtype=torch.float32)

    if gen_y is not None:
        hdata["generator"].y = gen_y
    elif n_gens > 0:
        hdata["generator"].y = torch.randn(n_gens, 2, dtype=torch.float32)

    if n_gens > 0 and gen_bus_map:
        gen_indices = torch.arange(n_gens, dtype=torch.long)
        bus_indices = torch.tensor(gen_bus_map, dtype=torch.long)
        hdata["generator", "generator_link", "bus"].edge_index = torch.stack(
            [gen_indices, bus_indices],
            dim=0,
        )
        hdata["bus", "generator_link", "generator"].edge_index = torch.stack(
            [bus_indices, gen_indices],
            dim=0,
        )

    if n_loads > 0:
        if load_demands is not None:
            hdata["load"].x = load_demands
        else:
            hdata["load"].x = torch.randn(n_loads, 2, dtype=torch.float32).abs()

        if load_served is not None:
            hdata["load"].y = load_served

        if load_bus_map is not None:
            load_indices = torch.arange(n_loads, dtype=torch.long)
            load_buses = torch.tensor(load_bus_map, dtype=torch.long)
            hdata["load", "load_link", "bus"].edge_index = torch.stack(
                [load_indices, load_buses],
                dim=0,
            )
            hdata["bus", "load_link", "load"].edge_index = torch.stack(
                [load_buses, load_indices],
                dim=0,
            )

    return hdata


def test_add_slack_generators_basic():
    """Test correct shapes and all-NaN targets when no load.y exists."""
    n_buses, n_gens = 3, 2
    hdata = _make_heterodata(n_buses, n_gens, gen_bus_map=[0, 1], n_loads=2, load_bus_map=[0, 1])

    result = add_slack_generators(hdata)

    assert result is hdata
    assert hdata["generator"].x.shape == (n_gens + 2 * n_buses, 11)
    assert hdata["generator"].y.shape == (n_gens + 2 * n_buses, 2)

    slack_y = hdata["generator"].y[n_gens:]
    assert torch.all(torch.isnan(slack_y))

    fwd_edges = hdata["generator", "generator_link", "bus"].edge_index
    rev_edges = hdata["bus", "generator_link", "generator"].edge_index
    assert fwd_edges.shape[1] == n_gens + 2 * n_buses
    assert rev_edges.shape[1] == n_gens + 2 * n_buses


def test_add_slack_generators_features():
    """Test that positive and negative slack features are set correctly."""
    n_buses, n_gens = 3, 1
    hdata = _make_heterodata(n_buses, n_gens, gen_bus_map=[0])
    add_slack_generators(hdata)

    pos_slacks = hdata["generator"].x[n_gens:n_gens + n_buses]
    neg_slacks = hdata["generator"].x[n_gens + n_buses:]

    assert torch.all(pos_slacks[:, 0] == 100.0)
    assert torch.all(pos_slacks[:, 1] == 0.0)
    assert torch.all(pos_slacks[:, 2] == 0.0)
    assert torch.all(pos_slacks[:, 3] == 9999.0)
    assert torch.all(pos_slacks[:, 4] == 0.0)
    assert torch.all(pos_slacks[:, 5] == 0.0)
    assert torch.all(pos_slacks[:, 6] == 9999.0)
    assert torch.all(pos_slacks[:, 7] == 1.0)
    assert torch.all(pos_slacks[:, 8] == 0.0)
    assert torch.all(pos_slacks[:, 9] == 300000.0)
    assert torch.all(pos_slacks[:, 10] == 0.0)

    assert torch.all(neg_slacks[:, 0] == 100.0)
    assert torch.all(neg_slacks[:, 2] == -9999.0)
    assert torch.all(neg_slacks[:, 3] == 0.0)
    assert torch.all(neg_slacks[:, 5] == -9999.0)
    assert torch.all(neg_slacks[:, 6] == 0.0)
    assert torch.all(neg_slacks[:, 7] == 1.0)
    assert torch.all(neg_slacks[:, 9] == -300000.0)


def test_add_slack_generators_with_load_shedding():
    """Test targets at load buses reflect shed amounts."""
    n_buses = 3
    load_demands = torch.tensor([[50.0, 20.0], [30.0, 10.0]], dtype=torch.float32)
    load_served = torch.tensor([[40.0, 15.0], [25.0, 8.0]], dtype=torch.float32)

    hdata = _make_heterodata(
        n_buses,
        n_gens=1,
        gen_bus_map=[0],
        n_loads=2,
        load_bus_map=[0, 1],
        load_demands=load_demands,
        load_served=load_served,
    )
    add_slack_generators(hdata)

    n_existing = 1
    pos_slack_y = hdata["generator"].y[n_existing:n_existing + n_buses]
    neg_slack_y = hdata["generator"].y[n_existing + n_buses:]

    assert torch.allclose(pos_slack_y[0], torch.tensor([10.0, 5.0]))
    assert torch.allclose(pos_slack_y[1], torch.tensor([5.0, 2.0]))
    assert torch.all(torch.isnan(pos_slack_y[2]))
    assert torch.all(torch.isnan(neg_slack_y))


def test_add_slack_generators_multiple_loads_same_bus():
    """Test that sheds from multiple loads at the same bus are summed."""
    n_buses = 2
    load_demands = torch.tensor([[50.0, 20.0], [30.0, 10.0], [40.0, 15.0]], dtype=torch.float32)
    load_served = torch.tensor([[45.0, 18.0], [28.0, 9.0], [35.0, 12.0]], dtype=torch.float32)

    hdata = _make_heterodata(
        n_buses,
        n_gens=1,
        gen_bus_map=[0],
        n_loads=3,
        load_bus_map=[0, 0, 1],
        load_demands=load_demands,
        load_served=load_served,
    )
    add_slack_generators(hdata)

    n_existing = 1
    pos_slack_y = hdata["generator"].y[n_existing:n_existing + n_buses]

    assert torch.allclose(pos_slack_y[0], torch.tensor([7.0, 3.0]))
    assert torch.allclose(pos_slack_y[1], torch.tensor([5.0, 3.0]))


def test_add_slack_generators_no_existing_generators():
    """Test with 0 existing generators - edges should be created from scratch."""
    n_buses = 2
    hdata = HeteroData()
    hdata["bus"].x = torch.randn(n_buses, 7, dtype=torch.float32)
    hdata["generator"].x = torch.empty(0, 11, dtype=torch.float32)
    hdata.baseMVA = torch.tensor([100.0], dtype=torch.float32)

    add_slack_generators(hdata)

    assert hdata["generator"].x.shape == (2 * n_buses, 11)
    assert hdata["generator"].y.shape == (2 * n_buses, 2)

    fwd = hdata["generator", "generator_link", "bus"].edge_index
    rev = hdata["bus", "generator_link", "generator"].edge_index
    assert fwd.shape == (2, 2 * n_buses)
    assert rev.shape == (2, 2 * n_buses)


def test_add_slack_generators_no_shedding():
    """Test that load buses with no shedding get [0, 0] targets."""
    n_buses = 2
    load_demands = torch.tensor([[50.0, 20.0]], dtype=torch.float32)
    load_served = torch.tensor([[50.0, 20.0]], dtype=torch.float32)

    hdata = _make_heterodata(
        n_buses,
        n_gens=1,
        gen_bus_map=[0],
        n_loads=1,
        load_bus_map=[0],
        load_demands=load_demands,
        load_served=load_served,
    )
    add_slack_generators(hdata)

    n_existing = 1
    pos_slack_y = hdata["generator"].y[n_existing:n_existing + n_buses]

    assert torch.allclose(pos_slack_y[0], torch.tensor([0.0, 0.0]))
    assert torch.all(torch.isnan(pos_slack_y[1]))


def test_add_slack_generators_edge_indices():
    """Test that slack gen edge indices map correctly to buses."""
    n_buses, n_gens = 3, 2
    hdata = _make_heterodata(n_buses, n_gens, gen_bus_map=[0, 1])
    add_slack_generators(hdata)

    fwd = hdata["generator", "generator_link", "bus"].edge_index
    new_fwd = fwd[:, n_gens:]
    new_gen_indices = new_fwd[0]
    new_bus_indices = new_fwd[1]

    assert torch.equal(new_gen_indices[:n_buses], torch.tensor([2, 3, 4]))
    assert torch.equal(new_bus_indices[:n_buses], torch.tensor([0, 1, 2]))

    assert torch.equal(new_gen_indices[n_buses:], torch.tensor([5, 6, 7]))
    assert torch.equal(new_bus_indices[n_buses:], torch.tensor([0, 1, 2]))


def test_add_slack_generators_basemva():
    """Test that mbase is read from hdata.baseMVA."""
    n_buses = 2
    hdata = _make_heterodata(n_buses, n_gens=1, gen_bus_map=[0], baseMVA=250.0)
    add_slack_generators(hdata)

    pos_slacks = hdata["generator"].x[1:1 + n_buses]
    neg_slacks = hdata["generator"].x[1 + n_buses:]

    assert torch.all(pos_slacks[:, 0] == 250.0)
    assert torch.all(neg_slacks[:, 0] == 250.0)


def _make_heterodata_with_edges(
    n_buses=4,
    n_gens=3,
    n_loads=2,
    ac_line_senders=None,
    ac_line_receivers=None,
    tr_senders=None,
    tr_receivers=None,
    ac_edge_label=None,
    tr_edge_label=None,
):
    """Helper to create HeteroData with branch edges for contingency tests."""
    hdata = HeteroData()
    hdata["bus"].x = torch.randn(n_buses, 7, dtype=torch.float32)
    hdata["generator"].x = torch.randn(n_gens, 11, dtype=torch.float32)
    hdata["generator"].y = torch.randn(n_gens, 2, dtype=torch.float32)
    hdata["load"].x = torch.randn(n_loads, 2, dtype=torch.float32)
    hdata.baseMVA = torch.tensor([100.0], dtype=torch.float32)

    gen_idx = torch.arange(n_gens, dtype=torch.long)
    bus_idx = torch.arange(n_gens, dtype=torch.long)
    hdata["generator", "generator_link", "bus"].edge_index = torch.stack([gen_idx, bus_idx], dim=0)
    hdata["bus", "generator_link", "generator"].edge_index = torch.stack([bus_idx, gen_idx], dim=0)

    if ac_line_senders is None:
        ac_line_senders = [0, 1, 2]
        ac_line_receivers = [1, 2, 3]
    s = torch.tensor(ac_line_senders, dtype=torch.long)
    r = torch.tensor(ac_line_receivers, dtype=torch.long)
    n_ac = s.size(0)
    hdata["bus", "ac_line", "bus"].edge_index = torch.stack([s, r], dim=0)
    hdata["bus", "ac_line", "bus"].edge_attr = torch.randn(n_ac, 9, dtype=torch.float32)
    if ac_edge_label is not None:
        hdata["bus", "ac_line", "bus"].edge_label = ac_edge_label
    else:
        hdata["bus", "ac_line", "bus"].edge_label = torch.randn(n_ac, 4, dtype=torch.float32)

    if tr_senders is not None:
        ts = torch.tensor(tr_senders, dtype=torch.long)
        tr_recv = torch.tensor(tr_receivers, dtype=torch.long)
        n_tr = ts.size(0)
        hdata["bus", "transformer", "bus"].edge_index = torch.stack([ts, tr_recv], dim=0)
        hdata["bus", "transformer", "bus"].edge_attr = torch.randn(n_tr, 11, dtype=torch.float32)
        if tr_edge_label is not None:
            hdata["bus", "transformer", "bus"].edge_label = tr_edge_label
        else:
            hdata["bus", "transformer", "bus"].edge_label = torch.randn(n_tr, 4, dtype=torch.float32)

    return hdata


def test_parse_contingency_n1_branch():
    """N-1 branch contingency: branch mapping resolves to the correct ac_line index."""
    cont_ids = np.array(["1"], dtype="S32")
    cont_types = np.array([0], dtype=np.int8)
    cont_names = np.array(["LINE_2_3_1"], dtype="S64")

    branch_mapping = {"1": ["ac_line", 1]}
    parsed = parse_contingency(0, cont_ids, cont_types, cont_names, branch_mapping)

    assert parsed.name == "LINE_2_3_1"
    assert parsed.generator_indices == []
    assert parsed.ac_line_indices == [1]
    assert parsed.transformer_indices == []


def test_parse_contingency_n1_generator():
    """N-1 generator contingency id=3 resolves to generator_indices == [2]."""
    cont_ids = np.array(["3"], dtype="S32")
    cont_types = np.array([1], dtype=np.int8)
    cont_names = np.array(["GEN_3"], dtype="S64")

    parsed = parse_contingency(0, cont_ids, cont_types, cont_names, {})

    assert parsed.generator_indices == [2]
    assert parsed.ac_line_indices == []
    assert parsed.transformer_indices == []


def test_parse_contingency_nk():
    """N-k contingency gen:2;gen:3;line:96 resolves generators and branch via mapping."""
    cont_ids = np.array(["gen:2;gen:3;line:96"], dtype="S64")
    cont_types = np.array([1], dtype=np.int8)
    cont_names = np.array(["NK_COMBO"], dtype="S64")

    branch_mapping = {"96": ["ac_line", 0]}
    parsed = parse_contingency(0, cont_ids, cont_types, cont_names, branch_mapping)

    assert sorted(parsed.generator_indices) == [1, 2]
    assert parsed.ac_line_indices == [0]


def test_parse_contingency_missing_branch_key():
    """Missing branch key in mapping raises KeyError (fail loudly)."""
    cont_ids = np.array(["1"], dtype="S32")
    cont_types = np.array([0], dtype=np.int8)
    cont_names = np.array(["LINE_2_3_1"], dtype="S64")

    with pytest.raises(KeyError):
        parse_contingency(0, cont_ids, cont_types, cont_names, {})


def test_parse_contingency_branch_on_transformer():
    """N-1 branch contingency where mapping indicates a transformer."""
    cont_ids = np.array(["99"], dtype="S32")
    cont_types = np.array([0], dtype=np.int8)
    cont_names = np.array(["LINE_1_4_99"], dtype="S64")

    branch_mapping = {"99": ["transformer", 0]}
    parsed = parse_contingency(0, cont_ids, cont_types, cont_names, branch_mapping)

    assert parsed.ac_line_indices == []
    assert parsed.transformer_indices == [0]


def test_apply_contingency_branch_removal():
    """Remove an ac_line edge, verify dimensions shrink correctly."""
    hdata = _make_heterodata_with_edges()
    n_ac_before = hdata[("bus", "ac_line", "bus")].edge_index.size(1)

    contingency = ParsedContingency(name="test", ac_line_indices=[1])
    apply_contingency(hdata, contingency)

    assert hdata[("bus", "ac_line", "bus")].edge_index.size(1) == n_ac_before - 1
    assert hdata[("bus", "ac_line", "bus")].edge_attr.size(0) == n_ac_before - 1
    assert hdata[("bus", "ac_line", "bus")].edge_label.size(0) == n_ac_before - 1


def test_apply_contingency_transformer_removal():
    """Remove a transformer edge."""
    hdata = _make_heterodata_with_edges(tr_senders=[0, 1], tr_receivers=[3, 2])
    n_tr_before = hdata[("bus", "transformer", "bus")].edge_index.size(1)

    contingency = ParsedContingency(name="test", transformer_indices=[0])
    apply_contingency(hdata, contingency)

    assert hdata[("bus", "transformer", "bus")].edge_index.size(1) == n_tr_before - 1


def test_apply_contingency_generator_removal():
    """Remove a generator, verify node count and edge reindexing."""
    hdata = _make_heterodata_with_edges()
    n_gens_before = hdata["generator"].x.size(0)

    contingency = ParsedContingency(name="test", generator_indices=[1])
    apply_contingency(hdata, contingency)

    assert hdata["generator"].x.size(0) == n_gens_before - 1
    assert hdata["generator"].y.size(0) == n_gens_before - 1

    fwd = hdata["generator", "generator_link", "bus"].edge_index
    assert fwd.size(1) == n_gens_before - 1
    assert fwd[0, 0].item() == 0
    assert fwd[0, 1].item() == 1

    rev = hdata["bus", "generator_link", "generator"].edge_index
    assert rev.size(1) == n_gens_before - 1
    assert rev[1, 0].item() == 0
    assert rev[1, 1].item() == 1


def test_apply_contingency_parallel_circuits():
    """Parallel circuits: mapping resolves the correct edge, which is then removed."""
    hdata = _make_heterodata_with_edges(
        ac_line_senders=[0, 0, 1],
        ac_line_receivers=[1, 1, 2],
    )

    cont_ids = np.array(["1"], dtype="S32")
    cont_types = np.array([0], dtype=np.int8)
    cont_names = np.array(["LINE_1_2_1"], dtype="S64")

    branch_mapping = {"1": ["ac_line", 0]}
    parsed = parse_contingency(0, cont_ids, cont_types, cont_names, branch_mapping)

    assert parsed.ac_line_indices == [0]

    apply_contingency(hdata, parsed)
    assert hdata[("bus", "ac_line", "bus")].edge_index.size(1) == 2


def test_contingency_e2e_topology(tmp_path):
    """Verify process_hdf5_scenario produces HeteroData with correct post-contingency topology."""
    h5_path = str(tmp_path / "contingency_e2e.h5")
    create_fake_contingency_h5(h5_path)

    with h5py.File(h5_path, "r") as f:
        data_list = []
        for scenario_key in f.keys():
            scenario = f[scenario_key]
            res = process_hdf5_scenario(scenario, scenario_key)
            if isinstance(res, list):
                data_list.extend(res)
            elif res is not None:
                data_list.append(res)
    assert len(data_list) == 2

    by_id = {d.scenario_id: d for d in data_list}

    cont1 = by_id["scenario_1_contingency_000001"]
    assert cont1[("bus", "ac_line", "bus")].edge_index.size(1) == 2

    cont2 = by_id["scenario_1_contingency_000002"]
    n_buses = cont2["bus"].x.size(0)
    assert cont2["generator"].x.size(0) == 2 + 2 * n_buses
