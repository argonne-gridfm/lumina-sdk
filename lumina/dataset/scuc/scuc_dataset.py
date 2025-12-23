"""
SCUC dataset utilities and helpers.

Ported from legacy `data_load.py` to follow the modular layout used by the
OPF components. Provides helpers for constructing `HeteroData` objects and the
`SCUCDataset` class that persists processed samples to disk.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch_geometric.data import HeteroData, InMemoryDataset


def load_problem_instance(filepath: str) -> Tuple[Dict[str, torch.Tensor], Dict[Tuple[str, str, str], torch.Tensor], List[str], List[str]]:
    """Load a single SCUC problem instance and return typed feature dictionaries."""
    with open(filepath, "r") as f:
        data = json.load(f)

    bus_ids = list(data["Buses"].keys())
    gen_ids = list(data["Generators"].keys())

    bus_id_to_idx = {bid: i for i, bid in enumerate(bus_ids)}
    gen_id_to_idx = {gid: i for i, gid in enumerate(gen_ids)}

    bus_feats = []
    for bus_id in bus_ids:
        bus_data = data["Buses"][bus_id]
        bus_type = bus_data.get("BUS_TYPE", "load")
        base_kv = bus_data.get("BASE_KV", 0.0)
        vmax = bus_data.get("VMAX", 1.0)
        vmin = bus_data.get("VMIN", 0.95)
        load_profile = bus_data.get("Load (MW)", [0.0] * 36)
        if isinstance(load_profile, list) and len(load_profile) < 36:
            load_profile = load_profile + [0.0] * (36 - len(load_profile))
        elif not isinstance(load_profile, list):
            load_profile = [0.0] * 36
        feat = [
            {"slack": 0, "load": 1, "generator": 2}.get(bus_type.lower(), 3),
            base_kv,
            vmax,
            vmin,
        ] + load_profile
        bus_feats.append(feat)
    bus_tensor = torch.tensor(bus_feats, dtype=torch.float32)

    gen_feats = []
    for gen_id in gen_ids:
        gen_data = data["Generators"][gen_id]
        qmax = gen_data.get("QMAX", 0.0)
        qmin = gen_data.get("QMIN", 0.0)
        vg = gen_data.get("VG", 1.0)
        mbase = gen_data.get("MBASE", 100.0)
        pmax = gen_data.get("PMAX", 0.0)
        pmin = gen_data.get("PMIN", 0.0)

        mw_curve = gen_data.get("Production cost curve (MW)", [])
        cost_curve = gen_data.get("Production cost curve ($)", [])
        coeffs = [0.0, 0.0, 0.0]
        if (
            isinstance(mw_curve, list)
            and isinstance(cost_curve, list)
            and len(mw_curve) >= 3
            and len(cost_curve) >= 3
        ):
            try:
                coeffs = np.polyfit(mw_curve, cost_curve, 2).tolist()
            except Exception:
                coeffs = [0.0, 0.0, 0.0]

        feat = coeffs + [qmax, qmin, vg, mbase, pmax, pmin]
        gen_feats.append(feat)
    gen_tensor = torch.tensor(gen_feats, dtype=torch.float32)

    gen_src, gen_dst = [], []
    for gen_id, gen_data in data["Generators"].items():
        g_idx = gen_id_to_idx[gen_id]
        b_idx = bus_id_to_idx[gen_data["Bus"]]
        gen_src.append(g_idx)
        gen_dst.append(b_idx)
    gen2bus_index = torch.tensor([gen_src, gen_dst], dtype=torch.long)
    bus2gen_index = torch.tensor([gen_dst, gen_src], dtype=torch.long)

    bus_src, bus_dst = [], []
    for line_data in data["Transmission lines"].values():
        fb = bus_id_to_idx[line_data["Source bus"]]
        tb = bus_id_to_idx[line_data["Target bus"]]
        bus_src.extend([fb, tb])
        bus_dst.extend([tb, fb])
    bus2bus_index = torch.tensor([bus_src, bus_dst], dtype=torch.long)

    x_dict = {"bus": bus_tensor, "generator": gen_tensor}
    edge_index_dict = {
        ("generator", "connects_to", "bus"): gen2bus_index,
        ("bus", "connects_to_gen", "generator"): bus2gen_index,
        ("bus", "connected_to", "bus"): bus2bus_index,
    }

    return x_dict, edge_index_dict, bus_ids, gen_ids


def load_targets_scuc(scuc_folder: str, problem_filename: str, gen_ids: List[str], case_name: str) -> torch.Tensor:
    """Load SCUC solution targets for a given problem instance."""
    date_part = problem_filename.replace(".json", "").replace("-", "_")
    scuc_filename = f"scuc_{case_name}_{date_part}.json"
    scuc_path = os.path.join(scuc_folder, scuc_filename)

    with open(scuc_path, "r") as f:
        scuc_data = json.load(f)

    scuc_labels = torch.zeros((len(gen_ids), 36, 4), dtype=torch.float32)
    for i, gid in enumerate(gen_ids):
        gnode = scuc_data.get("graph", {}).get("nodes", {}).get("generator", {}).get(gid, {})
        oper = gnode.get("operational", {})

        def _pad(series):
            if isinstance(series, list):
                if len(series) < 36:
                    return series + [0.0] * (36 - len(series))
                return series
            return [0.0] * 36

        scuc_labels[i, :, 0] = torch.tensor(_pad(oper.get("Is on", [0.0] * 36)), dtype=torch.float32)
        scuc_labels[i, :, 1] = torch.tensor(_pad(oper.get("Thermal production (MW)", [0.0] * 36)), dtype=torch.float32)
        scuc_labels[i, :, 2] = torch.tensor(_pad(oper.get("Switch on", [0.0] * 36)), dtype=torch.float32)
        scuc_labels[i, :, 3] = torch.tensor(_pad(oper.get("Switch off", [0.0] * 36)), dtype=torch.float32)

    return scuc_labels


def load_combine(problem_path: str, scuc_folder: str, case_name: str) -> HeteroData:
    """Load a combined SCUC problem/solution pair as a `HeteroData` graph."""
    with open(problem_path, "r") as f:
        prob_data = json.load(f)

    bus_ids = list(prob_data["Buses"].keys())
    gen_ids = list(prob_data["Generators"].keys())
    bus_id_to_idx = {bid: i for i, bid in enumerate(bus_ids)}
    gen_id_to_idx = {gid: i for i, gid in enumerate(gen_ids)}

    bus_feats_full = []
    for bus_id in bus_ids:
        bus = prob_data["Buses"][bus_id]
        bus_type = bus.get("BUS_TYPE", "load")
        base_kv = bus.get("BASE_KV", 0.0)
        vmax = bus.get("VMAX", 1.0)
        vmin = bus.get("VMIN", 0.95)
        lp = bus.get("Load (MW)", [0.0] * 36)
        if isinstance(lp, list) and len(lp) < 36:
            lp = lp + [0.0] * (36 - len(lp))
        elif not isinstance(lp, list):
            lp = [0.0] * 36
        node_type = 0 if bus_type.lower() == "bus" else 1
        bus_feats_full.append([node_type, base_kv, vmax, vmin] + lp)
    bus_feats_full = torch.tensor(bus_feats_full, dtype=torch.float32)
    bus_feats = bus_feats_full[:, :4]
    load_feats = bus_feats_full[:, 4:]

    gen_feats = []
    for gen_id in gen_ids:
        gen = prob_data["Generators"][gen_id]
        mw_curve = gen.get("Production cost curve (MW)", [])
        cost_curve = gen.get("Production cost curve ($)", [])
        coeffs = [0.0, 0.0, 0.0]
        if (
            isinstance(mw_curve, list)
            and isinstance(cost_curve, list)
            and len(mw_curve) >= 3
            and len(cost_curve) >= 3
        ):
            try:
                coeffs = np.polyfit(mw_curve, cost_curve, 2).tolist()
            except Exception:
                coeffs = [0.0, 0.0, 0.0]

        zeros_pad = coeffs + [0.0] * (11 - 3)

        prod_curve_mw = gen.get("Production cost curve (MW)", [])
        if len(prod_curve_mw) > 0:
            pmin_prod = prod_curve_mw[0]
            pmax_prod = prod_curve_mw[-1]
        else:
            pmin_prod = 0.0
            pmax_prod = 0.0

        scuc_fields = [
            gen.get("Ramp up limit (MW)", 0.0),
            gen.get("Ramp down limit (MW)", 0.0),
            gen.get("Startup limit (MW)", 0.0),
            gen.get("Shutdown limit (MW)", 0.0),
            gen.get("Minimum uptime (h)", 0.0),
            gen.get("Minimum downtime (h)", 0.0),
            gen.get("Initial status (h)", 0.0),
            gen.get("Initial power (MW)", 0.0),
            pmin_prod,
            pmax_prod,
        ]
        gen_feats.append(zeros_pad + scuc_fields)
    gen_feats = torch.tensor(gen_feats, dtype=torch.float32)

    gen_src, gen_dst = [], []
    for gid, gen in prob_data["Generators"].items():
        gi = gen_id_to_idx[gid]
        bi = bus_id_to_idx[gen["Bus"]]
        gen_src.append(gi)
        gen_dst.append(bi)
    gen2bus = torch.tensor([gen_src, gen_dst], dtype=torch.long)
    bus2gen = torch.tensor([gen_dst, gen_src], dtype=torch.long)

    bus_src, bus_dst = [], []
    for line in prob_data["Transmission lines"].values():
        f = bus_id_to_idx[line["Source bus"]]
        t = bus_id_to_idx[line["Target bus"]]
        bus_src += [f, t]
        bus_dst += [t, f]
    bus2bus = torch.tensor([bus_src, bus_dst], dtype=torch.long)

    date_part = os.path.basename(problem_path).replace(".json", "").replace("-", "_")
    scuc_filename = f"scuc_{case_name}_{date_part}.json"
    scuc_path = os.path.join(scuc_folder, scuc_filename)

    scuc_labels = torch.zeros((len(gen_ids), 36, 4), dtype=torch.float32)
    if os.path.exists(scuc_path):
        with open(scuc_path, "r") as f:
            scuc_data = json.load(f)

        def _pad36(values):
            if isinstance(values, list):
                if len(values) < 36:
                    return values + [0.0] * (36 - len(values))
                return values
            return [0.0] * 36

        for i, gid in enumerate(gen_ids):
            gnode = scuc_data.get("graph", {}).get("nodes", {}).get("generator", {}).get(gid, {})
            oper = gnode.get("operational", {})
            scuc_labels[i, :, 0] = torch.tensor(_pad36(oper.get("Is on", [0.0] * 36)))
            scuc_labels[i, :, 1] = torch.tensor(_pad36(oper.get("Thermal production (MW)", [0.0] * 36)))
            scuc_labels[i, :, 2] = torch.tensor(_pad36(oper.get("Switch on", [0.0] * 36)))
            scuc_labels[i, :, 3] = torch.tensor(_pad36(oper.get("Switch off", [0.0] * 36)))

    data = HeteroData()
    data["bus"].x = bus_feats
    data["bus"].temp_load = load_feats
    data["generator"].x = gen_feats
    data["generator"].y = scuc_labels
    data["generator", "generator_link", "bus"].edge_index = gen2bus
    data["bus", "generator_link", "generator"].edge_index = bus2gen
    data["bus", "ac_line", "bus"].edge_index = bus2bus
    return data


def load_all_instances(problem_folder: str, scuc_folder: str, case_name: str) -> List[HeteroData]:
    """Load and collate every SCUC instance available for a case."""
    files = sorted(glob.glob(os.path.join(problem_folder, "*.json")))
    data_list = []
    for fpath in files:
        try:
            data = load_combine(fpath, scuc_folder, case_name)
            data_list.append(data)
        except Exception as exc:  # pragma: no cover - diagnostic output
            print(f"[WARN] Skipping {fpath}: {exc}")
    return data_list


class SCUCDataset(InMemoryDataset):
    """Torch Geometric dataset wrapper for SCUC heterogenous graphs."""

    def __init__(
        self,
        root: str,
        case_name: str = "case14",
        problem_root: str = "data",
        sol_root: str = "data",
        transform=None,
        pre_transform=None,
        force_reload: bool = False,
    ):
        self.case_name = case_name
        self.case_id = None
        self.problem_root = problem_root
        self.sol_root = sol_root
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        self._ensure_generator_feature_dim(target_dim=21, persist=True)

    def _pad_generator_tensor(self, tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
        current_dim = tensor.size(-1)
        if current_dim == target_dim:
            return tensor
        if current_dim < target_dim:
            pad = tensor.new_zeros(tensor.size(0), target_dim - current_dim)
            return torch.cat([tensor, pad], dim=-1)
        return tensor[:, :target_dim]

    def _ensure_generator_feature_dim(self, target_dim: int, persist: bool = False) -> None:
        if "generator" not in self.data.node_types:
            return
        self.data["generator"].x = self._pad_generator_tensor(self.data["generator"].x, target_dim)
        if persist:
            torch.save((self.data, self.slices), self.processed_paths[0])

    def get(self, idx: int):
        data = super().get(idx)
        if "generator" in data.node_types:
            data["generator"].x = self._pad_generator_tensor(data["generator"].x, target_dim=21)
        case_idx = getattr(self, "case_id", None)
        if case_idx is not None:
            data.case_id = torch.tensor([int(case_idx)], dtype=torch.long)
        else:
            data.case_id = torch.tensor([-1], dtype=torch.long)
        data.case_name = self.case_name
        return data

    @property
    def processed_file_names(self) -> List[str]:
        return [f"scuc_{self.case_name}.pt"]

    def process(self) -> None:
        problem_folder = os.path.join(self.problem_root, f"scuc_{self.case_name}", self.case_name)
        scuc_folder = os.path.join(self.sol_root, f"scuc_{self.case_name}_sol")
        data_list = load_all_instances(problem_folder, scuc_folder, self.case_name)
        if len(data_list) == 0:
            raise RuntimeError(f"No SCUC instances found for {self.case_name}!")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def metadata(self):
        return (
            ["bus", "generator"],
            [
                ("generator", "generator_link", "bus"),
                ("bus", "generator_link", "generator"),
                ("bus", "ac_line", "bus"),
            ],
        )

    @property
    def node_types(self):
        return self.data.node_types

    @property
    def edge_types(self):
        return self.data.edge_types


__all__ = [
    "SCUCDataset",
    "load_problem_instance",
    "load_targets_scuc",
    "load_combine",
    "load_all_instances",
]

