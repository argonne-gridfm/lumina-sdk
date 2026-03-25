import json
import os
import os.path as osp

import h5py
import numpy as np
import pytest
import torch

from lumina.dataset.opf.opf_dataset import OPFDataset, process_json_file, process_hdf5_file


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


def _write_branch_mapping(directory, mapping):
    """Write a branch_mapping.json file into the given directory."""
    mapping_path = osp.join(directory, 'branch_mapping.json')
    with open(mapping_path, 'w') as f:
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

    with h5py.File(path, 'w') as f:
        scenario = f.create_group(scenario_id)

        base_sol = scenario.create_group('base_solution')
        opf_sol = base_sol.create_group('opf')
        sol_nodes = opf_sol.create_group('nodes')
        sol_nodes.create_dataset('bus', data=np.random.rand(2, n_buses))
        sol_nodes.create_dataset('generator', data=np.random.rand(2, n_gens))
        sol_nodes.create_dataset('load', data=np.random.rand(2, n_loads))
        opf_sol.attrs['objective'] = 1111.1

        # contingency definitions
        cg = scenario.create_group('contingencies')
        cg.create_dataset('ids', data=np.array(['1', '2'], dtype='S32'))
        cg.create_dataset('types', data=np.array([0, 1], dtype=np.int8))
        cg.create_dataset('names', data=np.array(['LINE_1_2_1', 'GEN_2'], dtype='S64'))

        # post-contingency solutions
        pc = scenario.create_group('post_contingency')

        # Contingency 1: branch LINE_1_2_1 tripped → ac_line edge 0→1 has zero flow
        cont1 = pc.create_group('contingency_000001')
        cont1.attrs['opf_converged'] = 1
        cont1_opf = cont1.create_group('opf')
        cont1_opf.attrs['objective'] = 1234.5
        c1_nodes = cont1_opf.create_group('nodes')
        c1_nodes.create_dataset('bus', data=np.random.rand(2, n_buses))
        c1_nodes.create_dataset('generator', data=np.random.rand(2, n_gens))
        c1_nodes.create_dataset('load', data=np.random.rand(2, n_loads))
        # Solution edges: tripped branch has zero flow
        c1_edges = cont1_opf.create_group('edges')
        c1_ac = c1_edges.create_group('ac_line')
        # 3 ac_line edges: edge 0 (tripped) has zero flow, edges 1-2 have nonzero
        # H5 edge solution order: pf, qf, pt, qt
        ac_sol_data = np.array([
            [0.0, 0.0, 0.0, 0.0],   # tripped edge: zero flow
            [10.0, 5.0, -9.8, -4.9], # live edge
            [8.0, 3.0, -7.8, -2.9],  # live edge
        ]).T  # shape (4, 3) for H5 format
        c1_ac.create_dataset('features', data=ac_sol_data)
        c1_tr = c1_edges.create_group('transformer')
        tr_sol_data = np.array([[5.0, 2.0, -4.9, -1.9]]).T  # 1 transformer, shape (4, 1)
        c1_tr.create_dataset('features', data=tr_sol_data)

        # Contingency 2: generator 2 tripped
        cont2 = pc.create_group('contingency_000002')
        cont2.attrs['opf_converged'] = 1
        cont2_opf = cont2.create_group('opf')
        cont2_opf.attrs['objective'] = 6789.0
        c2_nodes = cont2_opf.create_group('nodes')
        c2_nodes.create_dataset('bus', data=np.random.rand(2, n_buses))
        c2_nodes.create_dataset('generator', data=np.random.rand(2, n_gens))
        c2_nodes.create_dataset('load', data=np.random.rand(2, n_loads))
        c2_edges = cont2_opf.create_group('edges')
        c2_ac = c2_edges.create_group('ac_line')
        c2_ac.create_dataset('features', data=np.random.rand(4, 3))  # 3 ac_line edges
        c2_tr = c2_edges.create_group('transformer')
        c2_tr.create_dataset('features', data=np.random.rand(4, 1))  # 1 transformer edge

        # grid group
        grid = scenario.create_group('grid')
        nodes = grid.create_group('nodes')
        # bus data: H5Bus order [vmin, vmax, zone, area, bus_type] — 5 features, n_buses columns
        bus_data = np.zeros((5, n_buses))
        bus_data[0, :] = 0.9   # vmin
        bus_data[1, :] = 1.1   # vmax
        bus_data[2, :] = 1     # zone
        bus_data[3, :] = 1     # area
        bus_data[4, :] = [3, 1, 2, 1]  # bus_type (ref, PQ, PV, PQ)
        nodes.create_dataset('bus', data=bus_data)

        # generator data: 10 features (H5Generator order), n_gens columns
        gen_data = np.random.rand(10, n_gens)
        gen_data[9, :] = 1.0  # gen_status = on
        nodes.create_dataset('generator', data=gen_data)

        # load data: pd, qd, weight_p, weight_q — 4 features, n_loads columns
        load_data = np.random.rand(4, n_loads)
        nodes.create_dataset('load', data=load_data)

        # edge data
        edges = grid.create_group('edges')

        # ac_line: 3 edges: bus 0→1, bus 1→2, bus 2→3
        ac_line = edges.create_group('ac_line')
        ac_line.create_dataset('senders', data=np.array([0, 1, 2], dtype=np.int64))
        ac_line.create_dataset('receivers', data=np.array([1, 2, 3], dtype=np.int64))
        ac_features = np.random.rand(10, 3)  # H5ACLine: 10 features
        ac_features[9, :] = 1.0  # br_status
        ac_line.create_dataset('features', data=ac_features)

        # transformer: 1 edge: bus 0→3
        transformer = edges.create_group('transformer')
        transformer.create_dataset('senders', data=np.array([0], dtype=np.int64))
        transformer.create_dataset('receivers', data=np.array([3], dtype=np.int64))
        tr_features = np.random.rand(12, 1)  # H5Transformer: 12 features
        tr_features[9, :] = 1.0  # br_status
        transformer.create_dataset('features', data=tr_features)

        # virtual links
        gen_link = edges.create_group('generator_link')
        gen_link.create_dataset('senders', data=np.array([0, 1, 2], dtype=np.int64))
        gen_link.create_dataset('receivers', data=np.array([0, 1, 2], dtype=np.int64))

        load_link = edges.create_group('load_link')
        load_link.create_dataset('senders', data=np.array([0, 1], dtype=np.int64))
        load_link.create_dataset('receivers', data=np.array([1, 3], dtype=np.int64))

        shunt_link = edges.create_group('shunt_link')
        shunt_link.create_dataset('senders', data=np.array([], dtype=np.int64))
        shunt_link.create_dataset('receivers', data=np.array([], dtype=np.int64))

        context = grid.create_group('context')
        context.create_dataset('baseMVA', data=np.array([100.0]))

        meta = scenario.create_group('metadata')
        meta.attrs['n_contingencies'] = 2

    # Write branch_mapping.json alongside the HDF5 file.
    # Branch 1 = ac_line index 0, branches 2,3 = ac_line indices 1,2,
    # branch 4 = transformer index 0.
    branch_mapping = {
        "1": ["ac_line", 0],
        "2": ["ac_line", 1],
        "3": ["ac_line", 2],
        "4": ["transformer", 0],
    }
    _write_branch_mapping(osp.dirname(path), branch_mapping)


def test_standalone_hdf5_loading():
    h5_path = "tests/test_data/chunk_0001.h5"
    if not os.path.exists(h5_path):
        pytest.skip(f"Test data {h5_path} not found")
        
    data_list = process_hdf5_file(h5_path, n_jobs=2)
    assert len(data_list) > 0
    assert isinstance(data_list[0].baseMVA, torch.Tensor)
    assert data_list[0].baseMVA.ndim == 1

def test_round_trip_alignment(tmp_path):
    # we try serializing the same data using each schema we know about, and validate that the resulting
    # OPFDatasets are all functionally the same

    mock_data = create_mock_scenario_dict()
    json_path = str(tmp_path / "temp_test.json")
    h5_path = str(tmp_path / "temp_test.h5")

    save_as_json(mock_data, json_path)
    save_as_hdf5(mock_data, h5_path)

    json_hdata = process_json_file(json_path)
    hdf5_hdata = process_hdf5_file(h5_path)[0]

    # same mock data loaded from different formats should have the same values
    assert torch.allclose(json_hdata.baseMVA.to(torch.float32), hdf5_hdata.baseMVA.to(torch.float32))
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
            assert torch.allclose(jx, hx, atol=1e-5)
        if 'edge_label' in json_edge.keys() and 'edge_label' in hdf5_edge.keys():
            jy = json_edge['edge_label'].to(torch.float32)
            hy = hdf5_edge['edge_label'].to(torch.float32)
            assert torch.allclose(jy, hy, atol=1e-5)

def test_contingency_loading(tmp_path):
    root = str(tmp_path / 'data_contingency_test')
    case_name = 'pglib_opf_case14_ieee'
    os.makedirs(root, exist_ok=True)

    raw_dir = osp.join(root, 'OPFData', 'raw', 'dataset_release_1')
    results_dir = osp.join(raw_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    h5_path = osp.join(raw_dir, 'task_000001.h5')
    create_fake_contingency_h5(h5_path)

    data_list = process_hdf5_file(h5_path)
    assert len(data_list) == 2
    objectives = sorted([d.objective.item() for d in data_list])
    assert objectives == [1234.5, 6789.0]

    assert data_list[0].n_contingencies == 2
    assert data_list[0]['load'].x.shape[1] == 2
    # Check load solution presence
    assert 'y' in data_list[0]['load']
    assert data_list[0]['load'].y.shape[1] == 2

    class MockOPFDataset(OPFDataset):
        @property
        def raw_file_names(self):
            # When using .h5 files directly in raw_dir, process() detects them
            return ['task_000001.h5']

        def download(self):
            pass

    dataset = MockOPFDataset(root=root, case_name=case_name, group_id=0, force_reload=True)
    assert len(dataset) == 2
    assert dataset[0].objective.item() in [1234.5, 6789.0]
    assert dataset[0].n_contingencies == 2


# ---- Slack Generator Tests ----

from torch_geometric.data import HeteroData
from lumina.dataset.opf.contingency import add_slack_generators


def _make_heterodata(n_buses, n_gens, gen_bus_map, n_loads=0, load_bus_map=None,
                     load_demands=None, load_served=None, baseMVA=100.0,
                     gen_y=None):
    """Helper to create a minimal HeteroData for slack generator tests.

    Args:
        n_buses: Number of bus nodes.
        n_gens: Number of existing generators.
        gen_bus_map: List of bus indices each generator connects to.
        n_loads: Number of load nodes.
        load_bus_map: List of bus indices each load connects to.
        load_demands: Tensor [n_loads, 2] of (pd, qd) or None.
        load_served: Tensor [n_loads, 2] of (pd_served, qd_served) or None.
        baseMVA: System base power.
        gen_y: Tensor [n_gens, 2] of generator targets or None.
    """
    hdata = HeteroData()
    hdata['bus'].x = torch.randn(n_buses, 7, dtype=torch.float32)
    hdata['generator'].x = torch.randn(n_gens, 11, dtype=torch.float32)
    hdata.baseMVA = torch.tensor([baseMVA], dtype=torch.float32)

    if gen_y is not None:
        hdata['generator'].y = gen_y
    elif n_gens > 0:
        hdata['generator'].y = torch.randn(n_gens, 2, dtype=torch.float32)

    # Generator links
    if n_gens > 0 and gen_bus_map:
        gen_indices = torch.arange(n_gens, dtype=torch.long)
        bus_indices = torch.tensor(gen_bus_map, dtype=torch.long)
        hdata['generator', 'generator_link', 'bus'].edge_index = torch.stack(
            [gen_indices, bus_indices], dim=0
        )
        hdata['bus', 'generator_link', 'generator'].edge_index = torch.stack(
            [bus_indices, gen_indices], dim=0
        )

    # Loads
    if n_loads > 0:
        if load_demands is not None:
            hdata['load'].x = load_demands
        else:
            hdata['load'].x = torch.randn(n_loads, 2, dtype=torch.float32).abs()

        if load_served is not None:
            hdata['load'].y = load_served

        if load_bus_map is not None:
            load_indices = torch.arange(n_loads, dtype=torch.long)
            load_buses = torch.tensor(load_bus_map, dtype=torch.long)
            hdata['load', 'load_link', 'bus'].edge_index = torch.stack(
                [load_indices, load_buses], dim=0
            )
            hdata['bus', 'load_link', 'load'].edge_index = torch.stack(
                [load_buses, load_indices], dim=0
            )

    return hdata


def test_add_slack_generators_basic():
    """Test correct shapes and all-NaN targets when no load.y exists."""
    n_buses, n_gens = 3, 2
    hdata = _make_heterodata(n_buses, n_gens, gen_bus_map=[0, 1],
                             n_loads=2, load_bus_map=[0, 1])
    # No load.y set -> all slack targets should be NaN

    result = add_slack_generators(hdata)

    assert result is hdata  # in-place modification
    assert hdata['generator'].x.shape == (n_gens + 2 * n_buses, 11)
    assert hdata['generator'].y.shape == (n_gens + 2 * n_buses, 2)

    # All slack targets should be NaN (no load.y)
    slack_y = hdata['generator'].y[n_gens:]
    assert torch.all(torch.isnan(slack_y))

    # Check generator_link edges grew
    fwd_edges = hdata['generator', 'generator_link', 'bus'].edge_index
    rev_edges = hdata['bus', 'generator_link', 'generator'].edge_index
    assert fwd_edges.shape[1] == n_gens + 2 * n_buses
    assert rev_edges.shape[1] == n_gens + 2 * n_buses


def test_add_slack_generators_features():
    """Test that positive and negative slack features are set correctly."""
    n_buses, n_gens = 3, 1
    hdata = _make_heterodata(n_buses, n_gens, gen_bus_map=[0])
    add_slack_generators(hdata)

    pos_slacks = hdata['generator'].x[n_gens:n_gens + n_buses]
    neg_slacks = hdata['generator'].x[n_gens + n_buses:]

    # Positive slack features
    assert torch.all(pos_slacks[:, 0] == 100.0)   # mbase
    assert torch.all(pos_slacks[:, 1] == 0.0)     # pg
    assert torch.all(pos_slacks[:, 2] == 0.0)     # pmin
    assert torch.all(pos_slacks[:, 3] == 9999.0)  # pmax
    assert torch.all(pos_slacks[:, 4] == 0.0)     # qg
    assert torch.all(pos_slacks[:, 5] == 0.0)     # qmin
    assert torch.all(pos_slacks[:, 6] == 9999.0)  # qmax
    assert torch.all(pos_slacks[:, 7] == 1.0)     # vg
    assert torch.all(pos_slacks[:, 8] == 0.0)     # cost_c2
    assert torch.all(pos_slacks[:, 9] == 300000.0)  # cost_c1
    assert torch.all(pos_slacks[:, 10] == 0.0)    # cost_c0

    # Negative slack features
    assert torch.all(neg_slacks[:, 0] == 100.0)
    assert torch.all(neg_slacks[:, 2] == -9999.0)  # pmin
    assert torch.all(neg_slacks[:, 3] == 0.0)      # pmax
    assert torch.all(neg_slacks[:, 5] == -9999.0)  # qmin
    assert torch.all(neg_slacks[:, 6] == 0.0)      # qmax
    assert torch.all(neg_slacks[:, 7] == 1.0)      # vg
    assert torch.all(neg_slacks[:, 9] == -300000.0) # cost_c1


def test_add_slack_generators_with_load_shedding():
    """Test targets at load buses reflect shed amounts."""
    n_buses = 3
    load_demands = torch.tensor([[50.0, 20.0], [30.0, 10.0]], dtype=torch.float32)
    load_served = torch.tensor([[40.0, 15.0], [25.0, 8.0]], dtype=torch.float32)
    # Load 0 at bus 0: shed = [10, 5]
    # Load 1 at bus 1: shed = [5, 2]
    # Bus 2: no load -> NaN

    hdata = _make_heterodata(n_buses, n_gens=1, gen_bus_map=[0],
                             n_loads=2, load_bus_map=[0, 1],
                             load_demands=load_demands, load_served=load_served)
    add_slack_generators(hdata)

    n_existing = 1
    pos_slack_y = hdata['generator'].y[n_existing:n_existing + n_buses]
    neg_slack_y = hdata['generator'].y[n_existing + n_buses:]

    # Positive slack at bus 0: [10, 5]
    assert torch.allclose(pos_slack_y[0], torch.tensor([10.0, 5.0]))
    # Positive slack at bus 1: [5, 2]
    assert torch.allclose(pos_slack_y[1], torch.tensor([5.0, 2.0]))
    # Positive slack at bus 2: NaN (no load)
    assert torch.all(torch.isnan(pos_slack_y[2]))
    # All negative slack targets: NaN
    assert torch.all(torch.isnan(neg_slack_y))


def test_add_slack_generators_multiple_loads_same_bus():
    """Test that sheds from multiple loads at the same bus are summed."""
    n_buses = 2
    # Two loads at bus 0, one load at bus 1
    load_demands = torch.tensor([[50.0, 20.0], [30.0, 10.0], [40.0, 15.0]], dtype=torch.float32)
    load_served = torch.tensor([[45.0, 18.0], [28.0, 9.0], [35.0, 12.0]], dtype=torch.float32)
    # Load 0 at bus 0: shed [5, 2]
    # Load 1 at bus 0: shed [2, 1]
    # Load 2 at bus 1: shed [5, 3]
    # Bus 0 total: [7, 3]

    hdata = _make_heterodata(n_buses, n_gens=1, gen_bus_map=[0],
                             n_loads=3, load_bus_map=[0, 0, 1],
                             load_demands=load_demands, load_served=load_served)
    add_slack_generators(hdata)

    n_existing = 1
    pos_slack_y = hdata['generator'].y[n_existing:n_existing + n_buses]

    assert torch.allclose(pos_slack_y[0], torch.tensor([7.0, 3.0]))
    assert torch.allclose(pos_slack_y[1], torch.tensor([5.0, 3.0]))


def test_add_slack_generators_no_existing_generators():
    """Test with 0 existing generators - edges should be created from scratch."""
    n_buses = 2
    hdata = HeteroData()
    hdata['bus'].x = torch.randn(n_buses, 7, dtype=torch.float32)
    hdata['generator'].x = torch.empty(0, 11, dtype=torch.float32)
    hdata.baseMVA = torch.tensor([100.0], dtype=torch.float32)

    add_slack_generators(hdata)

    assert hdata['generator'].x.shape == (2 * n_buses, 11)
    assert hdata['generator'].y.shape == (2 * n_buses, 2)

    fwd = hdata['generator', 'generator_link', 'bus'].edge_index
    rev = hdata['bus', 'generator_link', 'generator'].edge_index
    assert fwd.shape == (2, 2 * n_buses)
    assert rev.shape == (2, 2 * n_buses)


def test_add_slack_generators_no_shedding():
    """Test that load buses with no shedding get [0, 0] targets."""
    n_buses = 2
    load_demands = torch.tensor([[50.0, 20.0]], dtype=torch.float32)
    load_served = torch.tensor([[50.0, 20.0]], dtype=torch.float32)  # fully served

    hdata = _make_heterodata(n_buses, n_gens=1, gen_bus_map=[0],
                             n_loads=1, load_bus_map=[0],
                             load_demands=load_demands, load_served=load_served)
    add_slack_generators(hdata)

    n_existing = 1
    pos_slack_y = hdata['generator'].y[n_existing:n_existing + n_buses]

    # Bus 0 has load but no shedding -> [0, 0]
    assert torch.allclose(pos_slack_y[0], torch.tensor([0.0, 0.0]))
    # Bus 1 has no load -> NaN
    assert torch.all(torch.isnan(pos_slack_y[1]))


def test_add_slack_generators_edge_indices():
    """Test that slack gen edge indices map correctly to buses."""
    n_buses, n_gens = 3, 2
    hdata = _make_heterodata(n_buses, n_gens, gen_bus_map=[0, 1])
    add_slack_generators(hdata)

    fwd = hdata['generator', 'generator_link', 'bus'].edge_index
    # New edges are the last 2*n_buses columns
    new_fwd = fwd[:, n_gens:]
    new_gen_indices = new_fwd[0]
    new_bus_indices = new_fwd[1]

    # Positive slacks: gen indices [2,3,4] -> bus [0,1,2]
    assert torch.equal(new_gen_indices[:n_buses], torch.tensor([2, 3, 4]))
    assert torch.equal(new_bus_indices[:n_buses], torch.tensor([0, 1, 2]))

    # Negative slacks: gen indices [5,6,7] -> bus [0,1,2]
    assert torch.equal(new_gen_indices[n_buses:], torch.tensor([5, 6, 7]))
    assert torch.equal(new_bus_indices[n_buses:], torch.tensor([0, 1, 2]))


def test_add_slack_generators_basemva():
    """Test that mbase is read from hdata.baseMVA."""
    n_buses = 2
    hdata = _make_heterodata(n_buses, n_gens=1, gen_bus_map=[0], baseMVA=250.0)
    add_slack_generators(hdata)

    pos_slacks = hdata['generator'].x[1:1 + n_buses]
    neg_slacks = hdata['generator'].x[1 + n_buses:]

    assert torch.all(pos_slacks[:, 0] == 250.0)
    assert torch.all(neg_slacks[:, 0] == 250.0)


# ---- Contingency Parsing / Application Tests ----

from lumina.dataset.opf.contingency import (
    ParsedContingency, parse_contingency, apply_contingency,
)


def _make_heterodata_with_edges(
    n_buses=4, n_gens=3, n_loads=2,
    ac_line_senders=None, ac_line_receivers=None,
    tr_senders=None, tr_receivers=None,
    ac_edge_label=None, tr_edge_label=None,
):
    """Helper to create HeteroData with branch edges for contingency tests."""
    hdata = HeteroData()
    hdata['bus'].x = torch.randn(n_buses, 7, dtype=torch.float32)
    hdata['generator'].x = torch.randn(n_gens, 11, dtype=torch.float32)
    hdata['generator'].y = torch.randn(n_gens, 2, dtype=torch.float32)
    hdata['load'].x = torch.randn(n_loads, 2, dtype=torch.float32)
    hdata.baseMVA = torch.tensor([100.0], dtype=torch.float32)

    # Generator links: gen i → bus i
    gen_idx = torch.arange(n_gens, dtype=torch.long)
    bus_idx = torch.arange(n_gens, dtype=torch.long)
    hdata['generator', 'generator_link', 'bus'].edge_index = torch.stack([gen_idx, bus_idx], dim=0)
    hdata['bus', 'generator_link', 'generator'].edge_index = torch.stack([bus_idx, gen_idx], dim=0)

    # AC line edges
    if ac_line_senders is None:
        ac_line_senders = [0, 1, 2]
        ac_line_receivers = [1, 2, 3]
    s = torch.tensor(ac_line_senders, dtype=torch.long)
    r = torch.tensor(ac_line_receivers, dtype=torch.long)
    n_ac = s.size(0)
    hdata['bus', 'ac_line', 'bus'].edge_index = torch.stack([s, r], dim=0)
    hdata['bus', 'ac_line', 'bus'].edge_attr = torch.randn(n_ac, 9, dtype=torch.float32)
    if ac_edge_label is not None:
        hdata['bus', 'ac_line', 'bus'].edge_label = ac_edge_label
    else:
        hdata['bus', 'ac_line', 'bus'].edge_label = torch.randn(n_ac, 4, dtype=torch.float32)

    # Transformer edges
    if tr_senders is not None:
        ts = torch.tensor(tr_senders, dtype=torch.long)
        tr_recv = torch.tensor(tr_receivers, dtype=torch.long)
        n_tr = ts.size(0)
        hdata['bus', 'transformer', 'bus'].edge_index = torch.stack([ts, tr_recv], dim=0)
        hdata['bus', 'transformer', 'bus'].edge_attr = torch.randn(n_tr, 11, dtype=torch.float32)
        if tr_edge_label is not None:
            hdata['bus', 'transformer', 'bus'].edge_label = tr_edge_label
        else:
            hdata['bus', 'transformer', 'bus'].edge_label = torch.randn(n_tr, 4, dtype=torch.float32)

    return hdata


# --- Unit tests for parse_contingency ---

def test_parse_contingency_n1_branch():
    """N-1 branch contingency: branch mapping resolves to the correct ac_line index."""
    cont_ids = np.array(['1'], dtype='S32')
    cont_types = np.array([0], dtype=np.int8)
    cont_names = np.array(['LINE_2_3_1'], dtype='S64')

    branch_mapping = {"1": ["ac_line", 1]}
    parsed = parse_contingency(0, cont_ids, cont_types, cont_names, branch_mapping)

    assert parsed.name == 'LINE_2_3_1'
    assert parsed.generator_indices == []
    assert parsed.ac_line_indices == [1]
    assert parsed.transformer_indices == []


def test_parse_contingency_n1_generator():
    """N-1 generator contingency id=3 resolves to generator_indices == [2]."""
    cont_ids = np.array(['3'], dtype='S32')
    cont_types = np.array([1], dtype=np.int8)
    cont_names = np.array(['GEN_3'], dtype='S64')

    parsed = parse_contingency(0, cont_ids, cont_types, cont_names, {})

    assert parsed.generator_indices == [2]
    assert parsed.ac_line_indices == []
    assert parsed.transformer_indices == []


def test_parse_contingency_nk():
    """N-k contingency gen:2;gen:3;line:96 resolves generators and branch via mapping."""
    cont_ids = np.array(['gen:2;gen:3;line:96'], dtype='S64')
    cont_types = np.array([1], dtype=np.int8)
    cont_names = np.array(['NK_COMBO'], dtype='S64')

    branch_mapping = {"96": ["ac_line", 0]}
    parsed = parse_contingency(0, cont_ids, cont_types, cont_names, branch_mapping)

    assert sorted(parsed.generator_indices) == [1, 2]
    assert parsed.ac_line_indices == [0]


def test_parse_contingency_missing_branch_key():
    """Missing branch key in mapping raises KeyError (fail loudly)."""
    cont_ids = np.array(['1'], dtype='S32')
    cont_types = np.array([0], dtype=np.int8)
    cont_names = np.array(['LINE_2_3_1'], dtype='S64')

    # Empty mapping — branch id '1' is not present
    with pytest.raises(KeyError):
        parse_contingency(0, cont_ids, cont_types, cont_names, {})


def test_parse_contingency_branch_on_transformer():
    """N-1 branch contingency where mapping indicates a transformer."""
    cont_ids = np.array(['99'], dtype='S32')
    cont_types = np.array([0], dtype=np.int8)
    cont_names = np.array(['LINE_1_4_99'], dtype='S64')

    branch_mapping = {"99": ["transformer", 0]}
    parsed = parse_contingency(0, cont_ids, cont_types, cont_names, branch_mapping)

    assert parsed.ac_line_indices == []
    assert parsed.transformer_indices == [0]


# --- Integration tests for apply_contingency ---

def test_apply_contingency_branch_removal():
    """Remove an ac_line edge, verify dimensions shrink correctly."""
    hdata = _make_heterodata_with_edges()
    n_ac_before = hdata[('bus', 'ac_line', 'bus')].edge_index.size(1)

    contingency = ParsedContingency(name='test', ac_line_indices=[1])
    apply_contingency(hdata, contingency)

    assert hdata[('bus', 'ac_line', 'bus')].edge_index.size(1) == n_ac_before - 1
    assert hdata[('bus', 'ac_line', 'bus')].edge_attr.size(0) == n_ac_before - 1
    assert hdata[('bus', 'ac_line', 'bus')].edge_label.size(0) == n_ac_before - 1


def test_apply_contingency_transformer_removal():
    """Remove a transformer edge."""
    hdata = _make_heterodata_with_edges(tr_senders=[0, 1], tr_receivers=[3, 2])
    n_tr_before = hdata[('bus', 'transformer', 'bus')].edge_index.size(1)

    contingency = ParsedContingency(name='test', transformer_indices=[0])
    apply_contingency(hdata, contingency)

    assert hdata[('bus', 'transformer', 'bus')].edge_index.size(1) == n_tr_before - 1


def test_apply_contingency_generator_removal():
    """Remove a generator, verify node count and edge reindexing."""
    hdata = _make_heterodata_with_edges()
    n_gens_before = hdata['generator'].x.size(0)  # 3

    contingency = ParsedContingency(name='test', generator_indices=[1])
    apply_contingency(hdata, contingency)

    assert hdata['generator'].x.size(0) == n_gens_before - 1
    assert hdata['generator'].y.size(0) == n_gens_before - 1

    # gen_to_bus edges: originally gen [0,1,2] → bus [0,1,2]; after removing gen 1:
    # gen 0 → bus 0, gen 2(now 1) → bus 2
    fwd = hdata['generator', 'generator_link', 'bus'].edge_index
    assert fwd.size(1) == n_gens_before - 1
    # Old gen 0 remaps to 0, old gen 2 remaps to 1
    assert fwd[0, 0].item() == 0  # gen 0
    assert fwd[0, 1].item() == 1  # gen 2 → remapped to 1

    rev = hdata['bus', 'generator_link', 'generator'].edge_index
    assert rev.size(1) == n_gens_before - 1
    assert rev[1, 0].item() == 0
    assert rev[1, 1].item() == 1


def test_apply_contingency_parallel_circuits():
    """Parallel circuits: mapping resolves the correct edge, which is then removed."""
    hdata = _make_heterodata_with_edges(
        ac_line_senders=[0, 0, 1],
        ac_line_receivers=[1, 1, 2],
    )

    cont_ids = np.array(['1'], dtype='S32')
    cont_types = np.array([0], dtype=np.int8)
    cont_names = np.array(['LINE_1_2_1'], dtype='S64')

    branch_mapping = {"1": ["ac_line", 0]}
    parsed = parse_contingency(0, cont_ids, cont_types, cont_names, branch_mapping)

    assert parsed.ac_line_indices == [0]

    apply_contingency(hdata, parsed)
    # 3 edges → 2 after removing one
    assert hdata[('bus', 'ac_line', 'bus')].edge_index.size(1) == 2


# --- End-to-end test ---

def test_contingency_e2e_topology(tmp_path):
    """Verify process_hdf5_file produces HeteroData with correct post-contingency topology."""
    h5_path = str(tmp_path / 'contingency_e2e.h5')
    create_fake_contingency_h5(h5_path)

    data_list = process_hdf5_file(h5_path)
    assert len(data_list) == 2

    # Find contingency_000001 (branch tripped) and contingency_000002 (gen tripped)
    by_id = {d.scenario_id: d for d in data_list}

    cont1 = by_id['scenario_1_contingency_000001']
    # Original: 3 ac_line edges. Contingency 1 trips branch LINE_1_2_1 (bus 0→1).
    # After removal: 2 ac_line edges remain.
    assert cont1[('bus', 'ac_line', 'bus')].edge_index.size(1) == 2

    cont2 = by_id['scenario_1_contingency_000002']
    # Original: 3 generators. Contingency 2 trips gen id=2 (index 1).
    # After removal: 2 real generators + 2*4 slack generators = 10
    n_buses = cont2['bus'].x.size(0)
    assert cont2['generator'].x.size(0) == 2 + 2 * n_buses
