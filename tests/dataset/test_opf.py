import copy
import json
import os
import os.path as osp
import shutil

import h5py
import numpy as np
import pytest
import torch

from lumina.dataset.opf.opf_dataset import OPFDataset, build_heterodata_from_grid, process_json_file, process_hdf5_scenario, _process_hdf5_scenario_from_path
from lumina.dataset.opf.opf_on_disk_dataset import OPFOnDiskDataset


def create_mock_scenario_dict():
    """Helper to create a nested dict matching the OPF JSON schema."""
    return {
        'grid': {
            'context': 100.0,
            'nodes': {
                'bus': [[1.0, 1, 0.9, 1.1]],
                'generator': [[100.0, 50.0, 0.0, 100.0, 20.0, -10.0, 30.0, 1.0, 0.01, 1.0, 0.0]],
                'load': [[20.0, 10.0]],
                'shunt': [[0.0, 0.1]]
            },
            'edges': {
                'ac_line': {
                    'senders': [0],
                    'receivers': [0],
                    'features': [[-0.5, 0.5, 0.0, 0.0, 0.01, 0.05, 100.0, 110.0, 120.0]]
                },
                'generator_link': {'senders': [0], 'receivers': [0]},
                'load_link': {'senders': [0], 'receivers': [0]},
                'shunt_link': {'senders': [0], 'receivers': [0]},
                'transformer': {'senders': [], 'receivers': [], 'features': []}
            }
        },
        'solution': {
            'nodes': {
                'bus': [[0.0, 1.0]],
                'generator': [[50.0, 20.0]]
            },
            'edges': {
                'ac_line': {
                    'features': [[10.0, 5.0, 10.0, 5.0]]
                },
                'transformer': {
                    'features': []
                }
            }
        },
        'metadata': {
            'objective': 1234.5
        }
    }

def save_as_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)

def save_as_hdf5(data, path):
    with h5py.File(path, 'w') as f:
        scenario = f.create_group('scenario_1')
        grid = scenario.create_group('grid')
        
        # Grid Context
        ctx = grid.create_group('context')
        ctx.create_dataset('baseMVA', data=np.array([data['grid']['context']]))
        
        # Nodes
        nodes = grid.create_group('nodes')
        
        gen_json = np.array(data['grid']['nodes']['generator'])
        gen_hdf5 = np.zeros((gen_json.shape[0], 10))
        # JSON: mbase(0), pg(1), pmin(2), pmax(3), qg(4), qmin(5), qmax(6), vg(7), cost_c2(8), cost_c1(9), cost_c0(10)
        # H5: pmax(0), pmin(1), qmax(2), qmin(3), cost_c2(4), cost_c1(5), cost_c0(6), vg(7), mbase(8), gen_status(9)
        gen_hdf5[:, 8] = gen_json[:, 0] # mbase
        gen_hdf5[:, 0] = gen_json[:, 3] # pmax
        gen_hdf5[:, 1] = gen_json[:, 2] # pmin
        gen_hdf5[:, 2] = gen_json[:, 6] # qmax
        gen_hdf5[:, 3] = gen_json[:, 5] # qmin
        gen_hdf5[:, 4] = gen_json[:, 8] # cost_c2
        gen_hdf5[:, 5] = gen_json[:, 9] # cost_c1
        gen_hdf5[:, 6] = gen_json[:, 10] # cost_c0
        gen_hdf5[:, 7] = gen_json[:, 7] # vg
        # gen_status remains 0.0 (off) or we could set to 1.0
        gen_hdf5[:, 9] = 1.0 
        
        nodes.create_dataset('generator', data=gen_hdf5.T)
        nodes.create_dataset('load', data=np.array(data['grid']['nodes']['load']).T)
        shunt_json = np.array(data['grid']['nodes']['shunt'])
        # JSON shunt: [bs, gs] -> H5 shunt: [gs, bs]
        shunt_hdf5 = np.zeros_like(shunt_json)
        shunt_hdf5[:, 0] = shunt_json[:, 1]  # gs
        shunt_hdf5[:, 1] = shunt_json[:, 0]  # bs
        nodes.create_dataset('shunt', data=shunt_hdf5.T)

        # Bus features in HDF5: [vmin, vmax, zone, area, bus_type]
        # JSON bus: [base_kv, bus_type, vmin, vmax]
        bus_json = np.array(data['grid']['nodes']['bus'])
        bus_hdf5 = np.zeros((bus_json.shape[0], 5)) 
        bus_hdf5[:, 0] = bus_json[:, 2] # vmin
        bus_hdf5[:, 1] = bus_json[:, 3] # vmax
        bus_hdf5[:, 4] = bus_json[:, 1] # bus_type
        
        nodes.create_dataset('bus', data=bus_hdf5.T)
        
        # Edges
        edges = grid.create_group('edges')
        ac = edges.create_group('ac_line')
        ac.create_dataset('senders', data=np.array(data['grid']['edges']['ac_line']['senders']))
        ac.create_dataset('receivers', data=np.array(data['grid']['edges']['ac_line']['receivers']))
        
        ac_json = np.array(data['grid']['edges']['ac_line']['features'])
        # JSON ACLine: angmin(0), angmax(1), b_fr(2), b_to(3), br_r(4), br_x(5), rate_a(6), rate_b(7), rate_c(8)
        # H5 ACLine: angmin(0), angmax(1), br_r(2), br_x(3), b_fr(4), b_to(5), rate_a(6), rate_b(7), rate_c(8), br_status(9)
        ac_hdf5 = np.zeros((ac_json.shape[0], 10))
        ac_hdf5[:, 0] = ac_json[:, 0] # angmin
        ac_hdf5[:, 1] = ac_json[:, 1] # angmax
        ac_hdf5[:, 2] = ac_json[:, 4] # br_r
        ac_hdf5[:, 3] = ac_json[:, 5] # br_x
        ac_hdf5[:, 4] = ac_json[:, 2] # b_fr
        ac_hdf5[:, 5] = ac_json[:, 3] # b_to
        ac_hdf5[:, 6:9] = ac_json[:, 6:9] # rates
        ac_hdf5[:, 9] = 1.0 # br_status
        
        ac.create_dataset('features', data=ac_hdf5.T)
        
        tr = edges.create_group('transformer')
        tr.create_dataset('senders', data=np.array(data['grid']['edges']['transformer']['senders']))
        tr.create_dataset('receivers', data=np.array(data['grid']['edges']['transformer']['receivers']))
        # H5Transformer has 12 features in schema
        tr_json = np.array(data['grid']['edges']['transformer']['features'])
        if tr_json.size > 0:
            # JSON Transformer: angmin(0), angmax(1), br_r(2), br_x(3), rate_a(4), rate_b(5), rate_c(6), tap(7), shift(8), b_fr(9), b_to(10)
            # H5 Transformer: angmin(0), angmax(1), br_r(2), br_x(3), b_fr(4), b_to(5), rate_a(6), rate_b(7), rate_c(8), br_status(9), tap(10), shift(11)
            tr_hdf5 = np.zeros((tr_json.shape[0], 12))
            tr_hdf5[:, 0] = tr_json[:, 0] # angmin
            tr_hdf5[:, 1] = tr_json[:, 1] # angmax
            tr_hdf5[:, 2] = tr_json[:, 2] # br_r
            tr_hdf5[:, 3] = tr_json[:, 3] # br_x
            tr_hdf5[:, 4] = tr_json[:, 9] # b_fr
            tr_hdf5[:, 5] = tr_json[:, 10] # b_to
            tr_hdf5[:, 6] = tr_json[:, 4] # rate_a
            tr_hdf5[:, 7] = tr_json[:, 5] # rate_b
            tr_hdf5[:, 8] = tr_json[:, 6] # rate_c
            tr_hdf5[:, 9] = 1.0 # br_status
            tr_hdf5[:, 10] = tr_json[:, 7] # tap
            tr_hdf5[:, 11] = tr_json[:, 8] # shift
            tr.create_dataset('features', data=tr_hdf5.T)
        else:
            tr.create_dataset('features', data=np.zeros((12, 0)))
        
        # Virtual links
        gl = edges.create_group('generator_link')
        gl.create_dataset('senders', data=np.array(data['grid']['edges']['generator_link']['senders']))
        gl.create_dataset('receivers', data=np.array(data['grid']['edges']['generator_link']['receivers']))
        
        ll = edges.create_group('load_link')
        ll.create_dataset('senders', data=np.array(data['grid']['edges']['load_link']['senders']))
        ll.create_dataset('receivers', data=np.array(data['grid']['edges']['load_link']['receivers']))
        
        sl = edges.create_group('shunt_link')
        sl.create_dataset('senders', data=np.array(data['grid']['edges']['shunt_link']['senders']))
        sl.create_dataset('receivers', data=np.array(data['grid']['edges']['shunt_link']['receivers']))
        
        # Solution
        sol = scenario.create_group('solution')
        sol_nodes = sol.create_group('nodes')
        sol_nodes.create_dataset('bus', data=np.array(data['solution']['nodes']['bus']).T)
        sol_nodes.create_dataset('generator', data=np.array(data['solution']['nodes']['generator']).T)
        
        sol_edges = sol.create_group('edges')
        sol_ac = sol_edges.create_group('ac_line')
        
        sol_ac_json = np.array(data['solution']['edges']['ac_line']['features'])
        # JSON EdgeSolution: pt(0), qt(1), pf(2), qf(3)
        # H5 EdgeSolution: pf(0), qf(1), pt(2), qt(3)
        sol_ac_hdf5 = np.zeros_like(sol_ac_json)
        sol_ac_hdf5[:, 0] = sol_ac_json[:, 2] # pf
        sol_ac_hdf5[:, 1] = sol_ac_json[:, 3] # qf
        sol_ac_hdf5[:, 2] = sol_ac_json[:, 0] # pt
        sol_ac_hdf5[:, 3] = sol_ac_json[:, 1] # qt
        sol_ac.create_dataset('features', data=sol_ac_hdf5.T)
        
        sol_tr = sol_edges.create_group('transformer')
        sol_tr.create_dataset('features', data=np.zeros((4, 0))) # empty
        
        # Metadata
        meta = scenario.create_group('metadata')
        meta.attrs['objective'] = data['metadata']['objective']


def test_standalone_hdf5_loading():
    h5_path = "tests/test_data/chunk_0001.h5"
    if not os.path.exists(h5_path):
        pytest.skip(f"Test data {h5_path} not found")
        
    with h5py.File(h5_path, 'r') as f:
        scenario_key = list(f.keys())[0]
        scenario = f[scenario_key]
        data = process_hdf5_scenario(scenario, scenario_key)
        
    assert data is not None
    if isinstance(data, list):
        data = data[0]
    assert isinstance(data.baseMVA, torch.Tensor)
    assert data.baseMVA.ndim == 1

def test_round_trip_alignment(tmp_path):
    # we try serializing the same data using each schema we know about, and validate that the resulting
    # OPFDatasets are all functionally the same

    mock_data = create_mock_scenario_dict()
    json_path = str(tmp_path / "temp_test.json")
    h5_path = str(tmp_path / "temp_test.h5")

    save_as_json(mock_data, json_path)
    save_as_hdf5(mock_data, h5_path)

    json_hdata = process_json_file(json_path)
    with h5py.File(h5_path, 'r') as f:
        scenario_key = 'scenario_1'
        scenario = f[scenario_key]
        hdf5_hdata = process_hdf5_scenario(scenario, scenario_key)

    # same mock data loaded from different formats should have the same values
    json_base_mva = torch.as_tensor(json_hdata.baseMVA).view(-1).to(torch.float32)
    hdf5_base_mva = torch.as_tensor(hdf5_hdata.baseMVA).view(-1).to(torch.float32)
    assert torch.allclose(json_base_mva, hdf5_base_mva)
    assert torch.allclose(json_hdata.objective.to(torch.float32), hdf5_hdata.objective.to(torch.float32))

    for node_type in json_hdata.node_types:
        assert node_type in hdf5_hdata.node_types
        if hasattr(json_hdata[node_type], 'x'):
            jx = json_hdata[node_type].x.to(torch.float32)
            hx = hdf5_hdata[node_type].x.to(torch.float32)
            assert jx.shape == hx.shape, f"Shape mismatch for {node_type}: JSON {jx.shape}, HDF5 {hx.shape}"
            if node_type != 'generator':
                assert torch.allclose(jx, hx, atol=1e-5)
        if hasattr(json_hdata[node_type], 'y'):
            jy = json_hdata[node_type].y.to(torch.float32)
            hy = hdf5_hdata[node_type].y.to(torch.float32)
            assert torch.allclose(jy, hy, atol=1e-5)

    for edge_type in json_hdata.edge_types:
        assert edge_type in hdf5_hdata.edge_types
        assert torch.equal(json_hdata[edge_type].edge_index, hdf5_hdata[edge_type].edge_index)

        json_edge = json_hdata[edge_type]
        hdf5_edge = hdf5_hdata[edge_type]

        if 'edge_attr' in json_edge.keys() and 'edge_attr' in hdf5_edge.keys():
            jx = json_edge['edge_attr'].to(torch.float32)
            hx = hdf5_edge['edge_attr'].to(torch.float32)
            if jx.numel() == 0 and hx.numel() == 0:
                continue
            assert torch.allclose(jx, hx, atol=1e-5)
        if 'edge_label' in json_edge.keys() and 'edge_label' in hdf5_edge.keys():
            jy = json_edge['edge_label'].to(torch.float32)
            hy = hdf5_edge['edge_label'].to(torch.float32)
            if jy.numel() == 0 and hy.numel() == 0:
                continue
            assert torch.allclose(jy, hy, atol=1e-5)





_DATASET_DIR = osp.dirname(__file__)
_EXAMPLE_JSON_PATH = osp.join(_DATASET_DIR, 'pglib_opf_case2000_goc_example_0.json')
_PROCESS_JSON_SNAPSHOT_PATH = osp.join(_DATASET_DIR, 'pglib_opf_case2000_goc_example_0.process_json_file.pt')
_ON_DISK_SNAPSHOT_PATH = osp.join(_DATASET_DIR, 'pglib_opf_case118_ieee_group_0_example_0.opf_on_disk_dataset.pt')
_CASE118_TAR_PATH = osp.join(_DATASET_DIR, 'pglib_opf_case118_ieee_0.tar.gz')


def _load_snapshot(path):
    snapshot = torch.load(
        path,
        map_location='cpu',
        weights_only=False,
    )
    assert isinstance(snapshot, HeteroData)
    return snapshot


def _assert_value_equal(actual, expected, path):
    if torch.is_tensor(expected):
        assert torch.is_tensor(actual), (
            f'{path}: expected Tensor, got {type(actual).__name__}'
        )
        assert actual.dtype == expected.dtype, (
            f'{path}: dtype mismatch {actual.dtype} != {expected.dtype}'
        )
        assert tuple(actual.shape) == tuple(expected.shape), (
            f'{path}: shape mismatch {tuple(actual.shape)} != {tuple(expected.shape)}'
        )
        torch.testing.assert_close(
            actual,
            expected,
            rtol=0,
            atol=0,
            equal_nan=True,
            msg=lambda msg: f'{path}: {msg}',
        )
        return

    assert type(actual) is type(expected), (
        f'{path}: type mismatch {type(actual).__name__} != {type(expected).__name__}'
    )
    assert actual == expected, f'{path}: value mismatch {actual!r} != {expected!r}'


def _assert_store_equal(actual_store, expected_store, path):
    actual_keys = sorted(actual_store.keys())
    expected_keys = sorted(expected_store.keys())
    assert actual_keys == expected_keys, (
        f'{path}: key mismatch {actual_keys} != {expected_keys}'
    )
    for key in expected_keys:
        _assert_value_equal(actual_store[key], expected_store[key], f'{path}.{key}')


def assert_heterodata_exact_match(actual, expected):
    assert isinstance(actual, HeteroData)
    assert isinstance(expected, HeteroData)
    assert actual.node_types == expected.node_types
    assert actual.edge_types == expected.edge_types
    assert len(actual.stores) == len(expected.stores)

    _assert_store_equal(actual._global_store, expected._global_store, 'global')

    for node_type in expected.node_types:
        _assert_store_equal(actual[node_type], expected[node_type], f'node[{node_type}]')

    for edge_type in expected.edge_types:
        edge_label = '__'.join(edge_type)
        _assert_store_equal(actual[edge_type], expected[edge_type], f'edge[{edge_label}]')


def _drop_targets(hdata):
    out = copy.deepcopy(hdata)
    if hasattr(out['bus'], 'y'):
        del out['bus'].y
    if hasattr(out['generator'], 'y'):
        del out['generator'].y
    if hasattr(out['bus', 'ac_line', 'bus'], 'edge_label'):
        del out['bus', 'ac_line', 'bus'].edge_label
    if hasattr(out['bus', 'transformer', 'bus'], 'edge_label'):
        del out['bus', 'transformer', 'bus'].edge_label
    return out


def _build_semantic_opf_payload():
    grid = {
        'context': [[[100.0]]],
        'nodes': {
            'bus': [
                [138.0, 1.0, 0.9, 1.1],
                [230.0, 3.0, 0.95, 1.05],
            ],
            'generator': [
                [100.0, 0.4, 0.1, 0.9, 0.05, -0.2, 0.3, 1.0, 1.5, 2.5, 3.5],
            ],
            'load': [
                [0.7, 0.2],
            ],
            'shunt': [
                [0.01, 0.02],
            ],
        },
        'edges': {
            'ac_line': {
                'senders': [0],
                'receivers': [1],
                'features': [[-0.5, 0.5, 0.01, 0.02, 0.03, 0.04, 100.0, 101.0, 102.0]],
            },
            'transformer': {
                'senders': [1],
                'receivers': [0],
                'features': [[-0.4, 0.4, 0.11, 0.12, 110.0, 111.0, 112.0, 1.0, 0.0, 0.13, 0.14]],
            },
            'generator_link': {
                'senders': [0],
                'receivers': [1],
            },
            'load_link': {
                'senders': [0],
                'receivers': [0],
            },
            'shunt_link': {
                'senders': [0],
                'receivers': [1],
            },
        },
    }
    metadata = {
        'objective': 321.0,
    }
    solution = {
        'nodes': {
            'bus': [
                [0.01, 1.02],
                [-0.02, 0.99],
            ],
            'generator': [
                [0.8, 0.1],
            ],
        },
        'edges': {
            'ac_line': {
                'features': [[0.7, 0.08, -0.7, -0.08]],
            },
            'transformer': {
                'features': [[0.5, 0.06, -0.5, -0.06]],
            },
        },
    }
    return grid, metadata, solution


def test_process_json_file_example_0_matches_snapshot():
    actual = process_json_file(_EXAMPLE_JSON_PATH)
    expected = _load_snapshot(_PROCESS_JSON_SNAPSHOT_PATH)

    assert_heterodata_exact_match(actual, expected)


def test_build_heterodata_from_grid_with_solution_matches_process_json_snapshot():
    with open(_EXAMPLE_JSON_PATH) as f:
        obj = json.load(f)

    actual = build_heterodata_from_grid(obj['grid'], obj['metadata'], obj['solution'])
    expected = _load_snapshot(_PROCESS_JSON_SNAPSHOT_PATH)

    assert_heterodata_exact_match(actual, expected)


def test_build_heterodata_from_grid_without_solution_matches_snapshot_without_targets():
    with open(_EXAMPLE_JSON_PATH) as f:
        obj = json.load(f)

    actual = build_heterodata_from_grid(obj['grid'], obj['metadata'])
    expected = _drop_targets(_load_snapshot(_PROCESS_JSON_SNAPSHOT_PATH))

    assert_heterodata_exact_match(actual, expected)


def test_build_heterodata_from_grid_semantic_fields_and_links():
    grid, metadata, solution = _build_semantic_opf_payload()

    data = build_heterodata_from_grid(grid, metadata, solution)

    assert data.baseMVA == 100.0
    assert data.objective.item() == 321.0

    assert data['bus'].x.shape == (2, 7)
    torch.testing.assert_close(
        data['bus'].x,
        torch.tensor([
            [138.0, 0.9, 1.1, 1.0, 0.0, 0.0, 0.0],
            [230.0, 0.95, 1.05, 0.0, 0.0, 1.0, 0.0],
        ], dtype=data['bus'].x.dtype),
        rtol=0,
        atol=0,
    )

    torch.testing.assert_close(
        data['bus', 'ac_line', 'bus'].edge_index,
        torch.tensor([[0], [1]], dtype=data['bus', 'ac_line', 'bus'].edge_index.dtype),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        data['bus', 'transformer', 'bus'].edge_index,
        torch.tensor([[1], [0]], dtype=data['bus', 'transformer', 'bus'].edge_index.dtype),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        data['generator', 'generator_link', 'bus'].edge_index,
        torch.tensor([[0], [1]], dtype=data['generator', 'generator_link', 'bus'].edge_index.dtype),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        data['bus', 'generator_link', 'generator'].edge_index,
        torch.tensor([[1], [0]], dtype=data['bus', 'generator_link', 'generator'].edge_index.dtype),
        rtol=0,
        atol=0,
    )


def test_build_heterodata_from_grid_without_solution_omits_targets_semantically():
    grid, metadata, _ = _build_semantic_opf_payload()

    data = build_heterodata_from_grid(grid, metadata)

    assert not hasattr(data['bus'], 'y')
    assert not hasattr(data['generator'], 'y')
    assert not hasattr(data['bus', 'ac_line', 'bus'], 'edge_label')
    assert not hasattr(data['bus', 'transformer', 'bus'], 'edge_label')


def test_build_heterodata_from_grid_has_expected_schema_and_dtype_categories():
    grid, metadata, solution = _build_semantic_opf_payload()

    data = build_heterodata_from_grid(grid, metadata, solution)

    assert data['generator'].x.shape == (1, 11)
    assert data['load'].x.shape == (1, 2)
    assert data['shunt'].x.shape == (1, 2)
    assert data['bus'].y.shape == (2, 2)
    assert data['generator'].y.shape == (1, 2)
    assert data['bus', 'ac_line', 'bus'].edge_attr.shape == (1, 9)
    assert data['bus', 'ac_line', 'bus'].edge_label.shape == (1, 4)
    assert data['bus', 'transformer', 'bus'].edge_attr.shape == (1, 11)
    assert data['bus', 'transformer', 'bus'].edge_label.shape == (1, 4)

    assert data['bus'].x.dtype.is_floating_point
    assert data['generator'].x.dtype.is_floating_point
    assert data['load'].x.dtype.is_floating_point
    assert data['shunt'].x.dtype.is_floating_point
    assert data['bus'].y.dtype.is_floating_point
    assert data['generator'].y.dtype.is_floating_point
    assert data['bus', 'ac_line', 'bus'].edge_attr.dtype.is_floating_point
    assert data['bus', 'ac_line', 'bus'].edge_label.dtype.is_floating_point
    assert not data['bus', 'ac_line', 'bus'].edge_index.dtype.is_floating_point
    assert not data['generator', 'generator_link', 'bus'].edge_index.dtype.is_floating_point


def test_process_json_file_returns_none_for_malformed_json(tmp_path):
    bad_json_path = tmp_path / 'bad.json'
    bad_json_path.write_text('{ not valid json')

    data = process_json_file(str(bad_json_path))

    assert data is None


def test_process_json_file_raises_for_missing_required_solution_key(tmp_path):
    grid, metadata, _ = _build_semantic_opf_payload()
    missing_solution_path = tmp_path / 'missing_solution.json'
    missing_solution_path.write_text(json.dumps({
        'grid': grid,
        'metadata': metadata,
    }))

    with pytest.raises(KeyError, match='solution'):
        process_json_file(str(missing_solution_path))


def test_opf_on_disk_dataset_processes_json_file_like_process_json_file(tmp_path):
    raw_dir = tmp_path / 'OPFData' / 'raw' / 'dataset_release_1'
    raw_dir.mkdir(parents=True)
    shutil.copy(_CASE118_TAR_PATH, raw_dir / 'pglib_opf_case118_ieee_0.tar.gz')

    actual = OPFOnDiskDataset(
        root=str(tmp_path),
        case_name='pglib_opf_case118_ieee',
        group_id=0,
        log=False,
        write_batch_size=1,
    ).get(0)
    expected = _load_snapshot(_ON_DISK_SNAPSHOT_PATH)

    assert_heterodata_exact_match(actual, expected)
