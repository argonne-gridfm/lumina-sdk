"""Contingency-to-ACOPF conversion utilities for HeteroData graphs.

Functions for adding slack generators and applying contingency
topology modifications to HeteroData objects.
"""

import logging
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
from torch_geometric.data import HeteroData

from .schema import JSONGenerator

logger = logging.getLogger(__name__)

# Generator feature indices derived from the canonical JSONGenerator schema
_GEN_IDX = JSONGenerator.get_field_indices()
_N_GEN_FEATURES = len(JSONGenerator.get_feature_names())

_SLACK_LIMIT = 9999.0
_SLACK_COST = 300000.0


def add_slack_generators(hdata: HeteroData) -> HeteroData:
    """Add positive and negative slack generators at every bus in the HeteroData graph.

    Each bus receives two slack generators:
      - Positive slack: can inject power (pmin=0, pmax=9999, cost_c1=+300000)
      - Negative slack: can absorb power (pmin=-9999, pmax=0, cost_c1=-300000)

    Generator features follow the JSONGenerator schema order (11 features):
      [mbase, pg, pmin, pmax, qg, qmin, qmax, vg, cost_c2, cost_c1, cost_c0]

    Targets (.y) for each slack generator have 2 features [pg, qg]:
      - Negative slacks: always NaN (will be masked out during loss computation).
      - Positive slacks at non-load buses: NaN.
      - Positive slacks at load buses: total load shed at that bus, computed as
        sum over all loads at the bus of (pd - pd_served, qd - qd_served).
        If load.y is absent (e.g. non-contingency data), all targets are NaN.

    Modifies hdata in-place and returns it.

    Args:
        hdata: A HeteroData with at least 'bus' and 'generator' node types,
            and hdata.baseMVA set. For target computation, expects load.x
            (demands), load.y (served amounts), and load_link edges.

    Returns:
        The same HeteroData with 2*n_buses additional generator nodes,
        corresponding generator_link edges (both directions), and targets.
    """
    n_buses = hdata['bus'].x.size(0)
    n_existing_gen = hdata['generator'].x.size(0)

    # Extract baseMVA
    if hasattr(hdata, 'baseMVA'):
        base_mva_val = hdata.baseMVA
        if torch.is_tensor(base_mva_val):
            mbase = base_mva_val.item() if base_mva_val.numel() == 1 else float(base_mva_val[0])
        else:
            mbase = float(base_mva_val)
    else:
        logging.warning("hdata.baseMVA not found; defaulting mbase to 100.0 for slack generators")
        mbase = 100.0

    # --- Build slack generator features (2*n_buses, 11) ---
    slack_x = torch.zeros(2 * n_buses, _N_GEN_FEATURES, dtype=torch.float32)

    # Positive slacks: rows [0, n_buses)
    slack_x[:n_buses, _GEN_IDX['mbase']] = mbase
    slack_x[:n_buses, _GEN_IDX['pmax']] = _SLACK_LIMIT
    slack_x[:n_buses, _GEN_IDX['qmax']] = _SLACK_LIMIT
    slack_x[:n_buses, _GEN_IDX['vg']] = 1.0
    slack_x[:n_buses, _GEN_IDX['cost_c1']] = _SLACK_COST

    # Negative slacks: rows [n_buses, 2*n_buses)
    slack_x[n_buses:, _GEN_IDX['mbase']] = mbase
    slack_x[n_buses:, _GEN_IDX['pmin']] = -_SLACK_LIMIT
    slack_x[n_buses:, _GEN_IDX['qmin']] = -_SLACK_LIMIT
    slack_x[n_buses:, _GEN_IDX['vg']] = 1.0
    slack_x[n_buses:, _GEN_IDX['cost_c1']] = -_SLACK_COST

    hdata['generator'].x = torch.cat([hdata['generator'].x, slack_x], dim=0)

    # --- Build slack generator targets (2*n_buses, 2) ---
    slack_y = torch.full((2 * n_buses, 2), float('nan'), dtype=torch.float32)

    has_load_y = (
        'load' in hdata.node_types
        and hasattr(hdata['load'], 'y')
        and hdata['load'].y is not None
    )

    if has_load_y:
        load_bus_edge = _get_load_bus_mapping(hdata)
        if load_bus_edge is not None:
            load_indices = load_bus_edge[0]
            bus_indices_for_loads = load_bus_edge[1]

            load_x = hdata['load'].x  # [n_load, 2]: pd, qd
            load_y = hdata['load'].y  # [n_load, 2]: pd_served, qd_served
            p_shed = load_x[load_indices, 0] - load_y[load_indices, 0]
            q_shed = load_x[load_indices, 1] - load_y[load_indices, 1]

            # Accumulate shed per bus using scatter_add
            bus_p_shed = torch.zeros(n_buses, dtype=torch.float32)
            bus_q_shed = torch.zeros(n_buses, dtype=torch.float32)
            bus_p_shed.scatter_add_(0, bus_indices_for_loads, p_shed)
            bus_q_shed.scatter_add_(0, bus_indices_for_loads, q_shed)

            # Mark buses that have at least one load
            has_load_mask = torch.zeros(n_buses, dtype=torch.bool)
            has_load_mask[bus_indices_for_loads] = True

            # Positive slack targets: shed at load buses, NaN at non-load buses
            nan_val = torch.tensor(float('nan'), dtype=torch.float32)
            slack_y[:n_buses, 0] = torch.where(has_load_mask, bus_p_shed, nan_val)
            slack_y[:n_buses, 1] = torch.where(has_load_mask, bus_q_shed, nan_val)
    # Negative slack targets (rows n_buses:2*n_buses) remain NaN

    if hasattr(hdata['generator'], 'y') and hdata['generator'].y is not None:
        hdata['generator'].y = torch.cat([hdata['generator'].y, slack_y], dim=0)
    else:
        logging.warning(
            "generator.y not found on HeteroData; "
            "creating generator.y with NaN for existing generators"
        )
        existing_gen_y = torch.full((n_existing_gen, 2), float('nan'), dtype=torch.float32)
        hdata['generator'].y = torch.cat([existing_gen_y, slack_y], dim=0)

    # --- Build generator_link edges for slack generators ---
    bus_range = torch.arange(n_buses, dtype=torch.long)
    pos_slack_indices = torch.arange(n_existing_gen, n_existing_gen + n_buses, dtype=torch.long)
    neg_slack_indices = torch.arange(n_existing_gen + n_buses, n_existing_gen + 2 * n_buses, dtype=torch.long)

    new_gen_indices = torch.cat([pos_slack_indices, neg_slack_indices], dim=0)
    new_bus_indices = torch.cat([bus_range, bus_range], dim=0)

    new_gen_to_bus = torch.stack([new_gen_indices, new_bus_indices], dim=0)
    new_bus_to_gen = torch.stack([new_bus_indices, new_gen_indices], dim=0)

    gen_to_bus_key = ('generator', 'generator_link', 'bus')
    bus_to_gen_key = ('bus', 'generator_link', 'generator')

    if gen_to_bus_key in hdata.edge_types:
        hdata[gen_to_bus_key].edge_index = torch.cat(
            [hdata[gen_to_bus_key].edge_index, new_gen_to_bus], dim=1
        )
    else:
        logging.warning("generator_link edges not found; creating from slack generators only")
        hdata[gen_to_bus_key].edge_index = new_gen_to_bus

    if bus_to_gen_key in hdata.edge_types:
        hdata[bus_to_gen_key].edge_index = torch.cat(
            [hdata[bus_to_gen_key].edge_index, new_bus_to_gen], dim=1
        )
    else:
        hdata[bus_to_gen_key].edge_index = new_bus_to_gen

    return hdata


def _get_load_bus_mapping(hdata: HeteroData):
    """Extract load-to-bus edge mapping from HeteroData.

    Returns:
        Tensor of shape [2, n_edges] where [0] is load indices and [1] is bus indices,
        or None if no load_link edges exist.
    """
    load_to_bus_key = ('load', 'load_link', 'bus')
    bus_to_load_key = ('bus', 'load_link', 'load')

    if load_to_bus_key in hdata.edge_types:
        edge_index = hdata[load_to_bus_key].edge_index
        return edge_index  # [0]=load_idx, [1]=bus_idx
    elif bus_to_load_key in hdata.edge_types:
        edge_index = hdata[bus_to_load_key].edge_index
        # Reverse: [0]=bus_idx, [1]=load_idx -> return [load_idx, bus_idx]
        return torch.stack([edge_index[1], edge_index[0]], dim=0)
    else:
        logging.warning(
            "No load_link edges found in HeteroData; "
            "all positive slack targets will be NaN"
        )
        return None


# ---------------------------------------------------------------------------
# Contingency parsing and topology modification
# ---------------------------------------------------------------------------

@dataclass
class ParsedContingency:
    """Resolved contingency with 0-based indices ready for graph modification."""
    name: str
    generator_indices: List[int] = field(default_factory=list)
    ac_line_indices: List[int] = field(default_factory=list)
    transformer_indices: List[int] = field(default_factory=list)


def _decode(val) -> str:
    """Decode a value that may be a bytes or numpy bytes object to a str."""
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode('utf-8')
    return str(val)


def parse_contingency(
    cont_index: int,
    cont_ids: np.ndarray,
    cont_types: np.ndarray,
    cont_names: np.ndarray,
    branch_mapping: dict,
) -> ParsedContingency:
    """Parse a contingency definition from HDF5 arrays and resolve element indices.

    Generator outages are resolved by ``int(id) - 1``.  Branch outages are
    resolved via *branch_mapping*, a dict loaded from the JSON file produced
    by ``build_contingency_index.py``.

    Args:
        cont_index: 0-based index into the contingency arrays.
        cont_ids: String array from contingencies/ids.
        cont_types: Int8 array from contingencies/types.
        cont_names: String array from contingencies/names.
        branch_mapping: Dict mapping str(branch_id) -> [type_str, index].
            Produced by parsing the MATPOWER .m file.

    Returns:
        A ParsedContingency with fully resolved 0-based indices.
    """
    cid = _decode(cont_ids[cont_index])
    ctype = int(cont_types[cont_index])
    cname = _decode(cont_names[cont_index])

    result = ParsedContingency(name=cname)

    if ';' in cid:
        # N-k contingency: parse tokens for generators and branches
        for token in cid.split(';'):
            token = token.strip()
            if token.startswith('gen:'):
                result.generator_indices.append(int(token[4:]) - 1)
            elif token.startswith('line:'):
                branch_id = token[5:]
                type_str, index = branch_mapping[branch_id]
                if type_str == 'ac_line':
                    result.ac_line_indices.append(index)
                else:
                    result.transformer_indices.append(index)
            else:
                logging.warning(
                    "Contingency '%s': unknown token '%s'; skipping",
                    cname, token,
                )
    elif ctype == 1:
        # N-1 generator contingency
        result.generator_indices.append(int(cid) - 1)
    elif ctype == 0:
        # N-1 branch contingency
        type_str, index = branch_mapping[cid]
        if type_str == 'ac_line':
            result.ac_line_indices.append(index)
        else:
            result.transformer_indices.append(index)
    else:
        logging.warning(
            "Unknown contingency type=%d id='%s' name='%s'; skipping",
            ctype, cid, cname,
        )

    return result


def apply_contingency(hdata: HeteroData, contingency: ParsedContingency) -> HeteroData:
    """Remove faulted elements from HeteroData graph topology.

    Modifies hdata in-place. Branch edges are removed first, then generator
    nodes (to avoid index shifting issues).

    Args:
        hdata: The HeteroData graph to modify.
        contingency: Parsed contingency with resolved indices.

    Returns:
        The same HeteroData with faulted elements removed.
    """
    if contingency.ac_line_indices:
        _remove_branch_edges(hdata, ('bus', 'ac_line', 'bus'), contingency.ac_line_indices)

    if contingency.transformer_indices:
        _remove_branch_edges(hdata, ('bus', 'transformer', 'bus'), contingency.transformer_indices)

    if contingency.generator_indices:
        _remove_generators(hdata, contingency.generator_indices)

    return hdata


def _remove_branch_edges(hdata: HeteroData, edge_type_key, removed_indices: List[int]):
    """Remove branch edges by index."""
    store = hdata[edge_type_key]
    n_edges = store.edge_index.size(1)
    keep = torch.ones(n_edges, dtype=torch.bool)
    for idx in removed_indices:
        keep[idx] = False
    store.edge_index = store.edge_index[:, keep]
    store.edge_attr = store.edge_attr[keep]
    store.edge_label = store.edge_label[keep]


def _remove_generators(hdata: HeteroData, gen_indices: List[int]):
    """Remove generator nodes and update generator_link edges."""
    n_gens = hdata['generator'].x.size(0)
    keep = torch.ones(n_gens, dtype=torch.bool)
    for idx in gen_indices:
        keep[idx] = False

    # Remove node features and targets
    hdata['generator'].x = hdata['generator'].x[keep]
    hdata['generator'].y = hdata['generator'].y[keep]

    # Build index remap: old_idx -> new_idx
    remap = torch.cumsum(keep.long(), 0) - 1
    removed_set = set(gen_indices)

    gen_to_bus_key = ('generator', 'generator_link', 'bus')
    bus_to_gen_key = ('bus', 'generator_link', 'generator')

    ei = hdata[gen_to_bus_key].edge_index
    edge_keep = ~torch.tensor([int(ei[0, i].item()) in removed_set for i in range(ei.size(1))], dtype=torch.bool)
    ei = ei[:, edge_keep]
    ei[0] = remap[ei[0]]
    hdata[gen_to_bus_key].edge_index = ei

    ei = hdata[bus_to_gen_key].edge_index
    edge_keep = ~torch.tensor([int(ei[1, i].item()) in removed_set for i in range(ei.size(1))], dtype=torch.bool)
    ei = ei[:, edge_keep]
    ei[1] = remap[ei[1]]
    hdata[bus_to_gen_key].edge_index = ei
