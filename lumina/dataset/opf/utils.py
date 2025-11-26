"""
Utility functions for processing OPF datasets.

Copyright (c) 2025, Argonne National Laboratory
All rights reserved.
"""
import argparse
import os
import random
import warnings
from typing import Dict

import numpy as np
import pandapower as pp
import torch
from pandapower.pypower.idx_brch import (BR_B, BR_R, BR_X, F_BUS, RATE_A,
                                         RATE_B, RATE_C, SHIFT, T_BUS, TAP)
from pandapower.pypower.idx_bus import (BASE_KV, BS, BUS_AREA, BUS_I, BUS_TYPE,
                                        GS, PD, QD, VA, VM, VMAX, VMIN, ZONE)
from pandapower.pypower.idx_gen import (GEN_BUS, GEN_STATUS, MBASE, PG, PMAX,
                                        PMIN, QG, QMAX, QMIN, VG)
from torch import Tensor
from torch_geometric.data import HeteroData


def extract_edge_index(obj: Dict, edge_name: str) -> Tensor:
    """Extract edge index from a grid object.

    Args:
        obj (Dict): Grid object containing edges information.
        edge_name (str): Name of the edge type.

    Returns:
        Tensor: Edge index tensor.
    """
    return torch.tensor(np.array([
        obj['grid']['edges'][edge_name]['senders'],
        obj['grid']['edges'][edge_name]['receivers'],
    ]))


def extract_edge_index_rev(obj: Dict, edge_name: str) -> Tensor:
    """Extract reversed edge index from a grid object.

    Args:
        obj (Dict): Grid object containing edges information.
        edge_name (str): Name of the edge type.

    Returns:
        Tensor: Reversed edge index tensor.
    """
    return torch.tensor(np.array([
        obj['grid']['edges'][edge_name]['receivers'],
        obj['grid']['edges'][edge_name]['senders'],
    ]))


def process_matpower_file(matpower_file):
    r"""Process a single MATPOWER case file.

    Args:
        matpower_file (str): Path to the MATPOWER case file.

    Returns:
        data (HeteroData): Processed single data object.
    """

    net = pp.converter.from_mpc(matpower_file)
    ppc = pp.converter.pypower.to_ppc(net, init='flat')
    # pp.runpp(net, numba=False)

    # Ensure branch data has the required columns for pandapower
    # BR_R_ASYM=21, BR_X_ASYM=22, BR_G=23 (need at least 24 columns)
    branch = ppc['branch']
    if branch.shape[1] < 24:
        # Pad with zeros for missing columns
        missing_cols = 24 - branch.shape[1]
        padding = np.zeros((branch.shape[0], missing_cols))
        branch = np.hstack([branch, padding])
        ppc['branch'] = branch

    # Calculate the admittance matrix Y using pypower's makeYbus (more compatible)
    # NOTE: https://matpower.org/docs/ref/matpower5.0/makeYbus.html
    from pypower.api import makeYbus
    Y_sp, _, _ = makeYbus(ppc['baseMVA'], ppc['bus'], ppc['branch'])

    load_bus_indices = net.load.bus.values.astype(np.int32)
    gen_bus_indices = net.gen.bus.values.astype(np.int32)

    obj = {
        'grid': {
            'context': [[ppc['baseMVA']]],
            'nodes': {
                'bus': ppc['bus'][:, [BUS_TYPE, BASE_KV, VMIN, VMAX]].astype(np.float32),
                'generator': ppc['gen'][:, [QMAX, QMIN, VG, MBASE, PMAX, PMIN]].astype(np.float32),
                'gencost': np.pad(ppc['gencost'], ((0, 0), (0, 7 - ppc['gencost'].shape[1])), 'constant', constant_values=0).astype(np.float32),
                'load': ppc['bus'][:, [PD, QD]].astype(np.float32),
                'shunt': ppc['bus'][:, [GS, BS]].astype(np.float32)
            },
            'edges': {
                'ac_line': {
                    'senders': ppc['branch'][:, F_BUS].astype(int),
                    'receivers': ppc['branch'][:, T_BUS].astype(int),
                    # TAP, SHIFT are at indices 8, 9
                    'features': ppc['branch'][:, [BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, TAP, SHIFT]].astype(np.float32)
                },
                'transformer': {
                    'senders': ppc['branch'][ppc['branch'][:, 8] != 0][:, F_BUS].astype(int),
                    'receivers': ppc['branch'][ppc['branch'][:, 8] != 0][:, T_BUS].astype(int),
                    'features': ppc['branch'][ppc['branch'][:, 8] != 0][:, [BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, TAP, SHIFT]].astype(np.float32)
                },
                'generator_link': {
                    'senders': np.arange(ppc['gen'].shape[0]),
                    'receivers': ppc['gen'][:, GEN_BUS].astype(int)
                },
                'load_link': {
                    'senders': ppc['bus'][:, BUS_I].astype(int),
                    'receivers': ppc['bus'][:, BUS_I].astype(int)
                },
                'shunt_link': {
                    'senders': ppc['bus'][:, BUS_I].astype(int),
                    'receivers': ppc['bus'][:, BUS_I].astype(int)
                }
            }
        },
        'metadata': {
            'objective': 0.0  # not given from MATPOWER file
        }
    }
    grid = obj['grid']

    # Graph-level properties:
    data = HeteroData()
    data.Y = Y_sp
    # Store baseMVA explicitly, remove generic data.x
    # data.x = torch.tensor(grid['context']).view(-1)
    data.base_mva = ppc['baseMVA']
    data.load_bus_indices = load_bus_indices
    data.gen_bus_indices = gen_bus_indices
    # data.baseMVA = ppc['baseMVA'] # Already stored as data.base_mva
    # DEBUG: duplicated baseMVA
    data.baseMVA = ppc['baseMVA']

    # Nodes (only some have a target):
    data['bus'].x = torch.tensor(grid['nodes']['bus'])
    data['generator'].x = torch.tensor(grid['nodes']['generator'])
    data['gencost'].x = torch.tensor(grid['nodes']['gencost'])
    data['load'].x = torch.tensor(grid['nodes']['load'])
    data['shunt'].x = torch.tensor(grid['nodes']['shunt'])

    # Edges (only ac lines and transformers have features):
    data['bus', 'ac_line', 'bus'].edge_index = extract_edge_index(obj, 'ac_line')
    data['bus', 'ac_line', 'bus'].edge_attr = torch.tensor(grid['edges']['ac_line']['features'])
    # data['bus', 'ac_line', 'bus'].edge_label = torch.tensor(solution['edges']['ac_line']['features'])

    data['bus', 'transformer', 'bus'].edge_index = extract_edge_index(obj, 'transformer')
    data['bus', 'transformer', 'bus'].edge_attr = torch.tensor(grid['edges']['transformer']['features'])
    # data['bus', 'transformer', 'bus'].edge_label = torch.tensor(solution['edges']['transformer']['features'])
    # print("Warning: edge_label is not available")

    data['generator', 'generator_link', 'bus'].edge_index = extract_edge_index(obj, 'generator_link')
    data['bus', 'generator_link', 'generator'].edge_index = extract_edge_index_rev(obj, 'generator_link')

    data['load', 'load_link', 'bus'].edge_index = extract_edge_index(obj, 'load_link')
    data['bus', 'load_link', 'load'].edge_index = extract_edge_index_rev(obj, 'load_link')

    data['shunt', 'shunt_link', 'bus'].edge_index = extract_edge_index(obj, 'shunt_link')
    data['bus', 'shunt_link', 'shunt'].edge_index = extract_edge_index_rev(obj, 'shunt_link')

    return data
