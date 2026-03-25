""" Customized OPFDataset class for the ACOPF task.

Changes:
- Add OPFMultiDataset class to combine multiple OPFDataset instances, either from same topology or different topologies.
- OPFDataset will only download and process a single group specified by group_id, remove num_groups argument.
- Add option to keep temporary files after processing.
- Parallelize data processing tasks.
- Remove temporary files after processing.
- Move pandapower utilities out of this file.

References:
- [OPFDataset](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.OPFDataset.html)
- [OPFData](https://arxiv.org/abs/2406.07234)

Copyright (c) 2025, Argonne National Laboratory
All rights reserved.
"""

import json
import os
import os.path as osp
import pickle
import shutil
import stat
import warnings
from glob import glob
from typing import Callable, Dict, List, Literal, Optional, Union

import h5py
import logging
import numpy as np
import torch
import tqdm
from joblib import Parallel, delayed, parallel_backend
from torch.utils.data import ConcatDataset
from torch_geometric.data import (HeteroData, InMemoryDataset, download_url,
                                  extract_tar)

from lumina.utils.graph_utils import OPFHomoWrapper

from contingency import add_slack_generators, parse_contingency, apply_contingency
from utils import extract_edge_index, extract_edge_index_rev
from schema import (
    JSONBus, JSONGenerator, JSONLoad, JSONShunt, JSONACLine, JSONTransformer,
    JSONBusSolution, JSONGeneratorSolution, JSONEdgeSolution,
    H5Bus, H5Generator, H5Load, H5Shunt, H5ACLine, H5Transformer,
    H5BusSolution, H5GeneratorSolution, H5EdgeSolution,
    ContingencyH5Load, ContingencyH5LoadSolution
)


class OPFDataset(InMemoryDataset):
    r"""The heterogeneous OPF data from the `"Large-scale Datasets for AC
    Optimal Power Flow with Topological Perturbations"
    <https://arxiv.org/abs/2406.07234>`_ paper.

    :class:`OPFDataset` is a large-scale dataset of solved optimal power flow
    problems, derived from the
    `pglib-opf <https://github.com/power-grid-lib/pglib-opf>`_ dataset.

    The physical topology of the grid is represented by the :obj:`"bus"` node
    type, and the connecting AC lines and transformers. Additionally,
    :obj:`"generator"`, :obj:`"load"`, and :obj:`"shunt"` nodes are connected
    to :obj:`"bus"` nodes using a dedicated edge type each, *e.g.*,
    :obj:`"generator_link"`.

    Edge direction corresponds to the properties of the line, *e.g.*,
    :obj:`b_fr` is the line charging susceptance at the :obj:`from`
    (source/sender) bus.

    Args:
        root (str): Root directory where the dataset should be saved.
        case_name (str, optional): The name of the original pglib-opf case.
            (default: :obj:`"pglib_opf_case14_ieee"`)
        group_id (int, optional): The specific group to load. Each group
            contains 15,000 samples. Valid values are [0, 19].
            (default: :obj:`0`)
        topological_perturbations (bool, optional): Whether to use the dataset
            with added topological perturbations. (default: :obj:`False`)
        transform (callable, optional): A function/transform that takes in
            a :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes
            in a :obj:`torch_geometric.data.HeteroData` object and returns
            a transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in a
            :obj:`torch_geometric.data.HeteroData` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
        keep_temp (bool, optional): Whether to keep the temporary files
            after processing. (default: :obj:`False`)
        n_jobs (int, optional): The number of jobs to use for parallel
            processing. If set to :obj:`-1`, all available cores will be used.
            NOTE: for larger dataset, it is recommended to set this to a lower positive value
            to avoid memory issues. (default: :obj:`-1`)
        local_raw_folder (str, optional): Local folder to look for raw files.
            If :obj:`None`, files will be downloaded from the internet.
            (default: :obj:`None`)

    Examples:
        >>> from lumina.dataset.opf.opf_dataset import OPFDataset
        >>> dataset = OPFDataset(root='./data', case_name='pglib_opf_case14_ieee')
        >>> # By default, only first group (i.e., group 0) is loaded, if you want to load multiple groups,
        >>> #   please use OPFMultiDataset instead:
        >>> dataset = OPFMultiDataset.from_case_groups(root='./data', case_name='pglib_opf_case14_ieee', group_ids=[0,1,2])
    """
    url = "https://storage.googleapis.com/gridopt-dataset"

    def __init__(
        self,
        root: str,
        case_name: Literal[
            'pglib_opf_case14_ieee',
            'pglib_opf_case30_ieee',
            'pglib_opf_case57_ieee',
            'pglib_opf_case118_ieee',
            'pglib_opf_case500_goc',
            'pglib_opf_case2000_goc',
            'pglib_opf_case4661_sdet',
            'pglib_opf_case6470_rte',
            'pglib_opf_case10000_goc',
            'pglib_opf_case13659_pegase',
        ] = 'pglib_opf_case14_ieee',
        group_id: int = 0,
        topological_perturbations: bool = False,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        keep_temp: bool = False,
        n_jobs: int = -1,
        local_raw_folder: str = None,
    ) -> None:

        self.case_name = case_name
        self.group_id = group_id
        self.topological_perturbations = topological_perturbations

        self._raw_root = osp.join(root, 'OPFData/raw')
        self._processed_root = osp.join(root, 'OPFData/processed')
        self._release = 'dataset_release_1'
        if topological_perturbations:
            self._release += '_nminusone'
        self.n_jobs = n_jobs
        self.keep_temp = keep_temp

        # TODO: add admittance matrix Y to the dataset - This may be used in multi-case evaluation
        # current_dir = os.path.dirname(os.path.abspath(__file__))
        # matpower_file = osp.join(current_dir, "../data/pglib", f"{case_name}.m")
        # net = pp.converter.from_mpc(matpower_file)
        # ppc = pp.converter.pypower.to_ppc(net, init='flat')
        # self.Y, _, _ = pp.makeYbus_pypower(ppc['baseMVA'], ppc['bus'], ppc['branch'])
        # self.Y_real = torch.tensor(self.Y.real.todense())
        # self.Y_imag = torch.tensor(self.Y.imag.todense())
        # self.load_bus_indices = net.load.bus.values.astype(np.int32)

        self.local_raw_folder = local_raw_folder
        # NOTE: processing steps:
        #   1. check downloaded: if raw files are not ready, download from url
        #   2. check processed: if processed files are not exist, process raw files
        super().__init__(root,
                         transform,
                         pre_transform,
                         pre_filter,
                         force_reload=force_reload)

        # Load only the specified group
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

        if osp.exists(self._processed_root):
            try:
                current_mode = os.stat(self._processed_root).st_mode
                os.chmod(self._processed_root, current_mode | stat.S_IWGRP)
            except OSError as exc:
                warnings.warn(f'Failed to set group write permission on {self._processed_root}: {exc}')

    @property
    def raw_dir(self) -> str:
        r""" Raw data folder """
        return osp.join(self._raw_root, self._release)

    @property
    def processed_dir(self) -> str:
        r""" Processed data folder. """
        return osp.join(self._processed_root, self._release, self.case_name)

    @property
    def tmp_dir(self) -> str:
        # NOTE: new class attribute
        r""" Temporary data folder. """
        return osp.join(self.raw_dir,
                        "gridopt-dataset-tmp",
                        self._release,
                        self.case_name)

    @property
    def raw_file_names(self) -> List[str]:
        r""" Raw file names, which are stored locally. """
        return [f'{self.case_name}_{self.group_id}.tar.gz']

    @property
    def processed_file_names(self) -> List[str]:
        r""" Processed file names, which are stored locally. """
        return [f'group_{self.group_id}.pt']

    def download(self) -> None:
        r""" Download .tar.gz files """
        print("Download files")

        # NOTE: download a single file for the specified group_id
        self.download_and_extract(self.raw_file_names[0])

        # NOTE: keep the parallel code for future reference
        # results = Parallel(n_jobs=self.n_jobs, backend="multiprocessing")(
        #     delayed(self.download_and_extract)(name)
        #     for name in self.raw_file_names)

        print(f"Downloaded {self.raw_file_names[0]} to {self.raw_dir}")

    def download_and_extract(self, name: str) -> None:
        r""" Download and extract a .tar.gz file. """
        url = f'{self.url}/{self._release}/{name}'
        path = download_url(url, self.raw_dir)
        extract_tar(path, self.raw_dir)

    def process(self) -> None:
        r""" Process the raw files into a single file. """
        h5_files = [f for f in self.raw_paths if f.endswith('.h5')]
        
        if h5_files:
            print(f"HDF5 files detected: {h5_files}")
            try:
                self.process_hdf5_group(self.group_id)
            except Exception as e:
                print(f"Error processing HDF5 group {self.group_id}: {e}")
                raise e
            return

        if not osp.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        try:
            self.process_json_group(self.group_id)
        except Exception as e:
            print(f"Error processing group {self.group_id}: {e}")
            raise e

        print(f"Processed group {self.group_id}")

        # NOTE: remove tmp_dir content to save local space
        if not self.keep_temp:
            shutil.rmtree(osp.join(self.raw_dir, 'gridopt-dataset-tmp'))

    def _post_process_and_save(self, data_list: List[Optional[Union[HeteroData, List[HeteroData]]]], group_id: int):
        """Helper to filter, transform, collate and save processed data."""
        flattened_list = []
        for item in data_list:
            if item is None:
                continue
            if isinstance(item, list):
                flattened_list.extend(item)
            else:
                flattened_list.append(item)
        
        data_list = flattened_list

        if self.pre_filter is not None or self.pre_transform is not None:
            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

        self.data, self.slices = self.collate(data_list)
        torch.save((self._data, self.slices), osp.join(self.processed_dir, f'group_{group_id}.pt'))

    def process_json_group(self, group_id: int):
        r""" Process a single group of files, save processed data to disk.

        Args:
            group_id (int): Group id.
        """

        group_json_files = glob(osp.join(self.tmp_dir, f'group_{group_id}', '*.json'))
        #
        if len(group_json_files) < 15000:
            extract_tar(osp.join(self.raw_dir, self.raw_file_names[0]), self.raw_dir)
            group_json_files = glob(osp.join(self.tmp_dir, f'group_{group_id}', '*.json'))

        data_list = Parallel(n_jobs=self.n_jobs, backend="threading")(
            delayed(process_json_file)(fn) for fn in tqdm.tqdm(group_json_files, desc=f"Group {group_id}"))

        self._post_process_and_save(data_list, group_id)

    def combine_datasets(self, file_paths: List[str]) -> List[HeteroData]:
        r""" Combine datasets from multiple files.

        Notes:
          - deprecated, kept for reference, please use `OPFMultiDataset` instead.
        """
        warnings.warn(
            "combine_datasets is deprecated and will be removed in a future version. "
            "Please use OPFMultiDataset instead for combining multiple datasets.",
            DeprecationWarning,
            stacklevel=2
        )

        combined_data = []
        for file_path in file_paths:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            combined_data.extend(data)
        return combined_data

    def merge_group_files(self) -> None:
        r"""Merge group files into single train/val/test based on groups options.

        Notes:
          - deprecated, kept for reference, please use `OPFMultiDataset` instead.
        """
        warnings.warn(
            "merge_group_files is deprecated and will be removed in a future version. "
            "Please use OPFMultiDataset instead for combining multiple datasets.",
            DeprecationWarning,
            stacklevel=2
        )

        data_files = [osp.join(self.processed_dir, f'group_{self.group_id}.pkl')]
        combined_data = self.combine_datasets(data_files)
        self.data, self.slices = self.collate(combined_data)

    def metadata(self):
        r""" Returns the metadata of the dataset. """
        # return (
        #     ['bus', 'generator', 'load', 'shunt'],
        #     [
        #         ('bus', 'ac_line', 'bus'),
        #         ('bus', 'transformer', 'bus'),
        #         ('generator', 'generator_link', 'bus'),
        #         ('bus', 'generator_link', 'generator'),
        #         ('load', 'load_link', 'bus'),
        #         ('bus', 'load_link', 'load'),
        #         ('shunt', 'shunt_link', 'bus'),
        #         ('bus', 'shunt_link', 'shunt')
        #     ]
        # )

        return {
            "nodes": {"bus": self._data['bus'].x.size(1),
                      "generator": self._data['generator'].x.size(1),
                      "load": self._data['load'].x.size(1),
                      "shunt": self._data['shunt'].x.size(1)},
            "edges": {
                ('bus', 'ac_line', 'bus'): self._data['bus', 'ac_line', 'bus'].edge_attr.size(1),
                ('bus', 'transformer', 'bus'): self._data['bus', 'transformer', 'bus'].edge_attr.size(1),
                ('generator', 'generator_link', 'bus'): 0,
                ('bus', 'generator_link', 'generator'): 0,
                ('load', 'load_link', 'bus'): 0,
                ('bus', 'load_link', 'load'): 0,
                ('shunt', 'shunt_link', 'bus'): 0,
                ('bus', 'shunt_link', 'shunt'): 0
            }
        }

    def process_hdf5_group(self, group_id: int):
        r""" Process a single group of HDF5 files, save processed data to disk.

        Args:
            group_id (int): Group id.
        """
        raw_paths = self.raw_paths
        h5_files = [f for f in raw_paths if f.endswith('.h5')]

        if not h5_files:
            print(f"No HDF5 files found in {self.raw_dir}")
            return

        tasks = []
        for h5_file in h5_files:
            with h5py.File(h5_file, 'r') as f:
                for scenario_key in f.keys():
                    tasks.append((h5_file, scenario_key))

        data_list = Parallel(n_jobs=self.n_jobs, backend="threading")(
            delayed(_process_hdf5_scenario_from_path)(fn, key)
            for fn, key in tqdm.tqdm(tasks, desc=f"Group {group_id} HDF5")
        )
        
        self._post_process_and_save(data_list, group_id)
    def __repr__(self) -> str:
        r""" Returns the string representation of the dataset. """
        return (f'{self.__class__.__name__}({len(self)}, '
                f'case_name={self.case_name}, '
                f'topological_perturbations={self.topological_perturbations})')


class OPFHomogeneousDataset(OPFDataset):
    r"""OPFDataset variant that stores homogeneous graphs during preprocessing."""

    def __init__(
        self,
        *args,
        add_node_type: bool = True,
        add_edge_type: bool = True,
        attach_full_edge_attr: bool = False,
        sanitize_targets: bool = True,
        log_bad_targets: bool = True,
        max_bad_target_logs: int = 1,
        processed_suffix: str = "homo",
        **kwargs,
    ) -> None:
        self._homo_wrapper = OPFHomoWrapper(
            add_node_type=add_node_type,
            add_edge_type=add_edge_type,
            attach_full_edge_attr=attach_full_edge_attr,
        )
        self._sanitize_targets = bool(sanitize_targets)
        self._log_bad_targets = bool(log_bad_targets)
        self._max_bad_target_logs = int(max_bad_target_logs)
        self._bad_target_logs = 0
        self._processed_suffix = processed_suffix or "homo"

        user_pre_transform = kwargs.pop("pre_transform", None)

        def pre_transform(data):
            homo_data = self._homo_wrapper.convert(data)
            self._sanitize_homo_targets(homo_data)
            if user_pre_transform is not None:
                homo_data = user_pre_transform(homo_data)
            return homo_data

        super().__init__(*args, pre_transform=pre_transform, **kwargs)

    @property
    def processed_dir(self) -> str:
        if self._processed_suffix:
            release = f"{self._release}_{self._processed_suffix}"
        else:
            release = self._release
        return osp.join(self._processed_root, release, self.case_name)

    def _sanitize_homo_targets(self, homo_data):
        y = getattr(homo_data, "y", None)
        if not torch.is_tensor(y):
            return

        finite_mask = torch.isfinite(y)
        if finite_mask.ndim > 1:
            row_mask = finite_mask.all(dim=-1)
        else:
            row_mask = finite_mask

        if bool(row_mask.all().item()):
            return

        if self._sanitize_targets:
            if y.ndim == 0:
                y = torch.zeros_like(y)
            else:
                y = y.clone()
                y[~row_mask] = 0
            homo_data.y = y

        homo_data.y_mask = row_mask.to(dtype=torch.bool)

        if self._should_log_bad_targets():
            bad_count = int((~row_mask).sum().item())
            total = int(row_mask.numel())
            action = "sanitized" if self._sanitize_targets else "left as-is"
            print(
                f"[OPFHomogeneousDataset] Non-finite targets: "
                f"{bad_count}/{total} rows {action}; stored y_mask."
            )
            self._bad_target_logs += 1

    def _should_log_bad_targets(self):
        if not self._log_bad_targets:
            return False
        if self._bad_target_logs >= self._max_bad_target_logs:
            return False
        try:
            rank = int(os.environ.get("RANK", "0"))
        except ValueError:
            rank = 0
        return rank == 0


class OPFMultiDataset(ConcatDataset):
    r"""Multi-group OPF dataset that combines multiple OPFDataset instances using ConcatDataset.

    This class allows combining multiple groups from the same case or different cases
    to create larger datasets for training.

    Args:
        datasets (List[OPFDataset]): List of OPFDataset instances to combine.

    Examples:
        >>> # For different cases
        >>> d1 = OPFDataset(root, case_name="pglib_opf_case14_ieee", group_id=0)
        >>> d2 = OPFDataset(root, case_name="pglib_opf_case30_ieee", group_id=0)
        >>> multi_dataset = OPFMultiDataset([d1, d2])

        >>> # For same case, multiple groups
        >>> datasets = []
        >>> for group_id in range(5):
        ...     ds = OPFDataset(root, case_name="pglib_opf_case14_ieee", group_id=group_id)
        ...     datasets.append(ds)
        >>> multi_dataset = OPFMultiDataset(datasets)

        >>> # Mixed: multiple groups from multiple cases
        >>> case_mapping = {
        ...     "pglib_opf_case14_ieee": [0, 1, 2],
        ...     "pglib_opf_case30_ieee": [0, 1]
        ... }
        >>> mixed_dataset = OPFMultiDataset.from_mixed_cases(root, case_mapping)
    """

    def __init__(self, datasets):
        super().__init__(datasets)
        self.datasets = datasets

    @classmethod
    def from_case_groups(
        cls,
        root: str,
        case_name: str,
        group_ids: List[int],
        dataset_cls=OPFDataset,
        **kwargs,
    ):
        r"""Create OPFMultiDataset from multiple groups of the same case.

        Args:
            root (str): Root directory where the dataset should be saved.
            case_name (str): The name of the original pglib-opf case.
            group_ids (List[int]): List of group IDs to load.
            dataset_cls: Dataset class to instantiate for each group.
            **kwargs: Additional arguments passed to OPFDataset constructor.

        Returns:
            OPFMultiDataset: Combined dataset from multiple groups.
        """
        datasets = []
        for group_id in group_ids:
            ds = dataset_cls(root=root, case_name=case_name, group_id=group_id, **kwargs)
            datasets.append(ds)
        return cls(datasets)

    @classmethod
    def from_multiple_cases(
        cls,
        root: str,
        case_configs: List[Dict],
        dataset_cls=OPFDataset,
        **kwargs,
    ):
        r"""Create OPFMultiDataset from multiple cases with their respective group IDs.

        Args:
            root (str): Root directory where the dataset should be saved.
            case_configs (List[Dict]): List of dictionaries, each containing 'case_name'
                and 'group_id' keys.
            dataset_cls: Dataset class to instantiate for each case.
            **kwargs: Additional arguments passed to OPFDataset constructor.

        Returns:
            OPFMultiDataset: Combined dataset from multiple cases.

        Example:
            >>> configs = [
            ...     {"case_name": "pglib_opf_case14_ieee", "group_id": 0},
            ...     {"case_name": "pglib_opf_case30_ieee", "group_id": 1},
            ... ]
            >>> multi_dataset = OPFMultiDataset.from_multiple_cases(root, configs)
        """
        datasets = []
        for config in case_configs:
            case_name = config.pop('case_name')
            group_id = config.pop('group_id')
            # Merge with additional kwargs
            dataset_kwargs = {**kwargs, **config}
            ds = dataset_cls(root=root, case_name=case_name, group_id=group_id, **dataset_kwargs)
            datasets.append(ds)
        return cls(datasets)

    @classmethod
    def from_mixed_cases(
        cls,
        root: str,
        case_group_mapping: Dict[str, List[int]],
        dataset_cls=OPFDataset,
        **kwargs,
    ):
        r"""Create OPFMultiDataset from multiple groups across different cases.

        This method allows loading multiple groups from multiple cases in a single call,
        which is useful when you want to combine several groups from different cases.

        Args:
            root (str): Root directory where the dataset should be saved.
            case_group_mapping (Dict[str, List[int]]): Dictionary mapping case names to
                lists of group IDs to load for each case.
            dataset_cls: Dataset class to instantiate for each group.
            **kwargs: Additional arguments passed to OPFDataset constructor.

        Returns:
            OPFMultiDataset: Combined dataset from multiple groups across multiple cases.

        Example:
            >>> # Load 3 groups from case14 and 2 groups from case30
            >>> case_mapping = {
            ...     "pglib_opf_case14_ieee": [0, 1, 2],
            ...     "pglib_opf_case30_ieee": [0, 1]
            ... }
            >>> multi_dataset = OPFMultiDataset.from_mixed_cases(root, case_mapping)
        """
        datasets = []
        for case_name, group_ids in case_group_mapping.items():
            for group_id in group_ids:
                ds = dataset_cls(root=root, case_name=case_name, group_id=group_id, **kwargs)
                datasets.append(ds)
        return cls(datasets)

    def metadata(self):
        r"""Returns the metadata of the first dataset (assuming all have same structure)."""
        return self.datasets[0].metadata()

    def __repr__(self) -> str:
        case_info = []
        for ds in self.datasets:
            case_info.append(f"{ds.case_name}[{ds.group_id}]")
        cases_str = ", ".join(case_info)
        return f'{self.__class__.__name__}({len(self)}, cases=[{cases_str}])'



def process_json_file(json_file):
    r"""Process a single json file.

    Args:
        json_file (str): Path to the json file.

    Returns:
        data (HeteroData): Processed single data object.
    """
    with open(json_file) as f:
        try:
            obj = json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file: {json_file}")
            return None

    grid = obj['grid']
    solution = obj['solution']
    metadata = obj['metadata']

    # Graph-level properties:
    hdata = HeteroData()
    hdata.baseMVA = torch.tensor(grid['context']).view(-1).item()
    hdata.objective = torch.tensor(metadata['objective'])

    # ! bus (only some have a target):
    # x: `base_kv, bus_type, vmin, vmax`
    bus_x = np.array(grid['nodes']['bus'])
    # bus_type (index 1)
    bus_type = bus_x[:, 1].astype(int)
    # One-hot encode bus_type (4 types: 1,2,3,4)
    # 1: pg, 2: pv, 3: ref, 4: isolated
    bus_type_onehot = np.eye(4)[bus_type - 1]  # bus_type assumed to be 1-based
    # Remove the original bus_type column and concatenate one-hot
    bus_x_wo_type = np.delete(bus_x, 1, axis=1)
    # x: `base_kv, vmin, vmax, pg, pv, ref, isolated`
    bus_x_final = np.concatenate([bus_x_wo_type, bus_type_onehot], axis=1)
    hdata['bus'].x = torch.tensor(bus_x_final)
    # y: `va, vm`
    hdata['bus'].y = torch.tensor(solution['nodes']['bus'])

    # ! generator (only some have a target):
    # x: `mbase, pg, pmin, pmax, qg, qmin, qmax, vg, cost_squared, cost_linear, cost_offset`
    hdata['generator'].x = torch.tensor(grid['nodes']['generator'])
    # y: `pg, qg`
    hdata['generator'].y = torch.tensor(solution['nodes']['generator'])

    # ! load (only some have a target):
    # x: `pd, qd`
    hdata['load'].x = torch.tensor(grid['nodes']['load'])

    # ! shunt (only some have a target):
    # x: `bs, gs`
    hdata['shunt'].x = torch.tensor(grid['nodes']['shunt'])

    # ! ac_line (only ac lines and transformers have features):
    hdata['bus', 'ac_line', 'bus'].edge_index = extract_edge_index(obj, 'ac_line')
    # edge_attr: `angmin, angmax, b_fr, b_to, br_r, br_x, rate_a, rate_b, rate_c`
    hdata['bus', 'ac_line', 'bus'].edge_attr = torch.tensor(grid['edges']['ac_line']['features'])
    # edge_label: `pt, qt, pf, qf`
    hdata['bus', 'ac_line', 'bus'].edge_label = torch.tensor(solution['edges']['ac_line']['features'])

    # ! transformer (only ac lines and transformers have features):
    hdata['bus', 'transformer', 'bus'].edge_index = extract_edge_index(obj, 'transformer')
    # edge_attr: `angmin, angmax, br_r, br_x, rate_a, rate_b, rate_c, tap, shift, b_fr, b_to`
    hdata['bus', 'transformer', 'bus'].edge_attr = torch.tensor(grid['edges']['transformer']['features'])
    # edge_label: `pt, qt, pf, qf`
    hdata['bus', 'transformer', 'bus'].edge_label = torch.tensor(solution['edges']['transformer']['features'])

    # ! virtual links:
    # bus-generator
    hdata['generator', 'generator_link', 'bus'].edge_index = extract_edge_index(obj, 'generator_link')
    hdata['bus', 'generator_link', 'generator'].edge_index = extract_edge_index_rev(obj, 'generator_link')
    # bus-load
    hdata['load', 'load_link', 'bus'].edge_index = extract_edge_index(obj, 'load_link')
    hdata['bus', 'load_link', 'load'].edge_index = extract_edge_index_rev(obj, 'load_link')
    # bus-shunt
    hdata['shunt', 'shunt_link', 'bus'].edge_index = extract_edge_index(obj, 'shunt_link')
    hdata['bus', 'shunt_link', 'shunt'].edge_index = extract_edge_index_rev(obj, 'shunt_link')

    return hdata


def process_hdf5_scenario(scenario, scenario_key: str) -> Union[Optional[HeteroData], List[HeteroData]]:
    """Process a single scenario from HDF5 format to HeteroData."""
    try:
        # contingency scenario or acopf scenario
        if 'base_solution' in scenario:
            return process_contingency_scenario(scenario, scenario_key)

        grid = scenario['grid']
        solution = scenario['solution']
        metadata = scenario['metadata']

        hdata = HeteroData()
        _process_nodes_hdf5(hdata, grid, solution)
        _process_edges_hdf5(hdata, grid, solution)

        hdata.baseMVA = torch.tensor(grid['context']['baseMVA'][()], dtype=torch.float32).view(-1)
        hdata.objective = torch.tensor(metadata.attrs['objective'], dtype=torch.float32)
        hdata.scenario_id = scenario_key

        return hdata
    except Exception as e:
        print(f"Error in process_hdf5_scenario for {scenario_key}: {e}")
        return None


def process_contingency_scenario(scenario, scenario_key: str) -> List[HeteroData]:
    """Process a contingency scenario from HDF5 format."""
    try:
        # formats are slightly different between single scenarios, grouped scenarios, and contingency (see schema.py)
        if 'grid' in scenario:
            grid = scenario['grid']
        elif 'grid' in scenario.file:
            grid = scenario.file['grid']
        elif 'grid' in scenario.parent:
            grid = scenario.parent['grid']
        else:
            grid = scenario['base_solution']['grid']

        try:
            base_mva = torch.tensor(grid['context']['baseMVA'][()], dtype=torch.float32).view(-1)
        except Exception:
            base_mva = torch.tensor([100.0], dtype=torch.float32)

        metadata = scenario['metadata']
        n_contingencies = int(metadata.attrs.get('n_contingencies', 0))

        # Read contingency definition arrays once (before the loop)
        cont_ids = cont_types = cont_names = None
        if 'contingencies' in scenario:
            cg = scenario['contingencies']
            cont_ids = cg['ids'][()]
            cont_types = cg['types'][()]
            cont_names = cg['names'][()]
        else:
            logging.warning("No 'contingencies' group in scenario '%s'; topology unmodified", scenario_key)

        # Load branch mapping JSON (must exist alongside the HDF5 file)
        branch_mapping = None
        if cont_ids is not None:
            from pathlib import Path
            h5_dir = Path(scenario.file.filename).parent
            n_grid_branches = (
                len(grid['edges']['ac_line']['senders'][()])
                + len(grid['edges']['transformer']['senders'][()])
            )
            # Search for branch_mapping*.json files and pick the one
            # whose entry count matches this grid's total branch count.
            branch_mapping = None
            for candidate_path in sorted(h5_dir.glob('branch_mapping*.json')):
                with open(candidate_path, "r") as fp:
                    candidate = json.load(fp)
                if len(candidate) == n_grid_branches:
                    branch_mapping = candidate
                    break
            if branch_mapping is None:
                raise FileNotFoundError(
                    f"No branch mapping file in {h5_dir} with {n_grid_branches} branches. "
                    f"Generate one with: python example/opf/build_contingency_index.py "
                    f"--matpower <path_to_.m_file> --output {h5_dir / 'branch_mapping.json'}"
                )

        results = []

        # iterate over post-contingency solutions and load each as a separate sample in heterodata
        if 'post_contingency' in scenario:
            pc_group = scenario['post_contingency']
            for cont_name in pc_group.keys():
                cont_group = pc_group[cont_name]

                if 'opf' not in cont_group:
                    continue

                solution = cont_group['opf']
                if solution.attrs.get('opf_converged', 1) != 1:
                    continue

                hdata = HeteroData()
                _process_nodes_hdf5(hdata, grid, solution)
                _process_edges_hdf5(hdata, grid, solution)

                hdata.baseMVA = base_mva
                hdata.objective = torch.tensor(solution.attrs['objective'], dtype=torch.float32)
                hdata.scenario_id = f"{scenario_key}_{cont_name}"
                hdata.n_contingencies = n_contingencies

                # Apply contingency topology modifications before adding slacks
                if cont_ids is not None:
                    cont_idx = int(cont_name.split('_')[1]) - 1  # contingency_000001 → 0, etc.
                    parsed = parse_contingency(
                        cont_idx, cont_ids, cont_types, cont_names,
                        branch_mapping,
                    )
                    apply_contingency(hdata, parsed)

                add_slack_generators(hdata)

                results.append(hdata)

        return results
    except Exception as e:
        print(f"Error in process_contingency_scenario for {scenario_key}: {e}")
        return []


def _process_hdf5_scenario_from_path(h5_file_path, scenario_key):
    with h5py.File(h5_file_path, 'r') as f:
        scenario = f[scenario_key]
        return process_hdf5_scenario(scenario, scenario_key)


def _align_features(data: np.ndarray, src_schema, dst_schema) -> np.ndarray:
    """feature indices are different between hdf5 acopf, json acopf, and hdf5 contingency and must be aligned."""
    mapping = src_schema.get_alignment_map(dst_schema)
    aligned = np.zeros((data.shape[0], len(dst_schema.get_feature_names())))
    for src_idx, dst_idx in mapping.items():
        aligned[:, dst_idx] = data[:, src_idx]
    return aligned


def _process_nodes_hdf5(hdata: HeteroData, grid, solution):
    """Process node data from HDF5 format aligned with JSON schema."""
    nodes = grid['nodes']
    sol_nodes = solution['nodes']

    required_fields = ['bus', 'load', 'generator']
    missing_fields = [f for f in required_fields if f not in nodes]

    if missing_fields:
        raise Exception(f"Invalid HDF5 file: missing the following required fields:{missing_fields}")


    bus_data = nodes['bus'][()]
    if bus_data.shape[0] == 5:
        bus_data = bus_data.T
    
    if bus_data.shape[1] >= 5:
        # HDF5 bus: vmin, vmax, zone, area, bus_type
        # JSON bus: base_kv, bus_type, vmin, vmax
        aligned_bus = _align_features(bus_data, H5Bus, JSONBus)
        
        # augment hdf5 data with base_kv if missing
        json_indices = JSONBus.get_field_indices()
        if 'base_kv' in json_indices and np.all(aligned_bus[:, json_indices['base_kv']] == 0):
            aligned_bus[:, json_indices['base_kv']] = 1.0

        # JSON loader does one-hot encoding for bus_type whereas hdf5 uses a category label so we add the one hot:
        bus_type = aligned_bus[:, json_indices['bus_type']].astype(int)
        bus_type_onehot = np.eye(4)[bus_type - 1]
        bus_x_wo_type = np.delete(aligned_bus, json_indices['bus_type'], axis=1)
        bus_x_final = np.concatenate([bus_x_wo_type, bus_type_onehot], axis=1)
    else:
        bus_x_final = bus_data

    hdata['bus'].x = torch.tensor(bus_x_final, dtype=torch.float32)

    # remaining fields are all aligned automatically (see schema.py for index mappings)

    # generator
    gen_data = nodes['generator'][()]
    if gen_data.shape[0] == 10: # H5 format from PowerModels.jl is column-major
        gen_data = gen_data.T
    
    if gen_data.shape[1] >= 10:
        gen_x_final = _align_features(gen_data, H5Generator, JSONGenerator)
    else:
        gen_x_final = gen_data
    hdata['generator'].x = torch.tensor(gen_x_final, dtype=torch.float32)

    # load
    load_data = nodes['load'][()]
    if load_data.shape[0] == 4 or load_data.shape[0] == 2:
        load_data = load_data.T

    if load_data.shape[1] == 4:
        load_x_final = _align_features(load_data, ContingencyH5Load, JSONLoad)
    elif load_data.shape[1] == 2:
        load_x_final = _align_features(load_data, H5Load, JSONLoad)
    else:
        load_x_final = load_data

    hdata['load'].x = torch.tensor(load_x_final, dtype=torch.float32)

    # shunt
    if 'shunt' in nodes:
        shunt_data = nodes['shunt'][()]
        if shunt_data.shape[0] == 2:
            shunt_data = shunt_data.T
            
        if shunt_data.shape[1] == 2:
            shunt_x_final = _align_features(shunt_data, H5Shunt, JSONShunt)
        else:
            shunt_x_final = shunt_data
        hdata['shunt'].x = torch.tensor(shunt_x_final, dtype=torch.float32)


    # solution data
    if 'bus' in sol_nodes:
        bus_sol = sol_nodes['bus'][()]
        if bus_sol.shape[0] == 2:
            bus_sol = bus_sol.T
        hdata['bus'].y = torch.tensor(_align_features(bus_sol, H5BusSolution, JSONBusSolution), dtype=torch.float32)

    if 'generator' in sol_nodes:
        gen_sol = sol_nodes['generator'][()]
        if gen_sol.shape[0] == 2:
            gen_sol = gen_sol.T
        hdata['generator'].y = torch.tensor(_align_features(gen_sol, H5GeneratorSolution, JSONGeneratorSolution), dtype=torch.float32)

    if 'load' in sol_nodes:
        load_sol = sol_nodes['load'][()]
        if load_sol.shape[0] == 2:
            load_sol = load_sol.T
        if load_sol.shape[1] == 2:
            hdata['load'].y = torch.tensor(_align_features(load_sol, ContingencyH5LoadSolution, JSONLoad), dtype=torch.float32)


def _process_edges_hdf5(hdata: HeteroData, grid, solution):
    """Process edge data from HDF5 format."""
    if 'edges' not in grid:
        return
    
    edges = grid['edges']

    # solution might not have edges in some structures (e.g. contingency)
    sol_edges = solution.get('edges', {})

    if 'ac_line' in edges:
        ac_edge = edges['ac_line']
        senders = ac_edge['senders'][()]
        receivers = ac_edge['receivers'][()]
        hdata['bus', 'ac_line', 'bus'].edge_index = torch.stack([
            torch.tensor(senders, dtype=torch.long),
            torch.tensor(receivers, dtype=torch.long)
        ], dim=0)

        ac_features = ac_edge['features'][()]
        if ac_features.shape[0] == 9 or ac_features.shape[0] == 10:
            ac_features = ac_features.T
            
        if ac_features.shape[0] > 0 and ac_features.shape[1] >= 9:
            ac_x_final = _align_features(ac_features, H5ACLine, JSONACLine)
        elif ac_features.shape[0] > 0:
            ac_x_final = ac_features
        else:
            ac_x_final = np.zeros((0, len(JSONACLine.get_feature_names())))
            
        hdata['bus', 'ac_line', 'bus'].edge_attr = torch.tensor(ac_x_final, dtype=torch.float32)

        # group-based and single scenario solution edges
        if 'ac_line' in sol_edges:
            ac_sol_obj = sol_edges['ac_line']
            if isinstance(ac_sol_obj, h5py.Dataset):
                ac_sol = ac_sol_obj[()]
            elif isinstance(ac_sol_obj, h5py.Group) and 'features' in ac_sol_obj:
                ac_sol = ac_sol_obj['features'][()]
            else:
                ac_sol = None
            
            if ac_sol is not None:
                if ac_sol.shape[0] == 4:
                    ac_sol = ac_sol.T
                if ac_sol.shape[0] > 0:
                    hdata['bus', 'ac_line', 'bus'].edge_label = torch.tensor(_align_features(ac_sol, H5EdgeSolution, JSONEdgeSolution),
                                                                            dtype=torch.float32)
                else:
                    hdata['bus', 'ac_line', 'bus'].edge_label = torch.zeros((0, len(JSONEdgeSolution.get_feature_names())), dtype=torch.float32)

    if 'transformer' in edges:
        trans_edge = edges['transformer']
        senders = trans_edge['senders'][()]
        receivers = trans_edge['receivers'][()]
        hdata['bus', 'transformer', 'bus'].edge_index = torch.stack([
            torch.tensor(senders, dtype=torch.long),
            torch.tensor(receivers, dtype=torch.long)
        ], dim=0)

        trans_features = trans_edge['features'][()]
        if trans_features.shape[0] == 11 or trans_features.shape[0] == 12:
            trans_features = trans_features.T

        if trans_features.shape[0] > 0 and trans_features.shape[1] >= 11:
            trans_x_final = _align_features(trans_features, H5Transformer, JSONTransformer)
        elif trans_features.shape[0] > 0:
            trans_x_final = trans_features
        else:
            trans_x_final = np.zeros((0, len(JSONTransformer.get_feature_names())))
            
        hdata['bus', 'transformer', 'bus'].edge_attr = torch.tensor(trans_x_final, dtype=torch.float32)

        if 'transformer' in sol_edges:
            tr_sol_obj = sol_edges['transformer']
            if isinstance(tr_sol_obj, h5py.Dataset):
                trans_sol = tr_sol_obj[()]
            elif isinstance(tr_sol_obj, h5py.Group) and 'features' in tr_sol_obj:
                trans_sol = tr_sol_obj['features'][()]
            else:
                trans_sol = None

            if trans_sol is not None:
                if trans_sol.shape[0] == 4:
                    trans_sol = trans_sol.T
                if trans_sol.shape[0] > 0:
                    hdata['bus', 'transformer', 'bus'].edge_label = torch.tensor(_align_features(trans_sol, H5EdgeSolution, JSONEdgeSolution),
                                                                                 dtype=torch.float32)
                else:
                    hdata['bus', 'transformer', 'bus'].edge_label = torch.zeros((0, len(JSONEdgeSolution.get_feature_names())), dtype=torch.float32)

    _process_virtual_links_hdf5(hdata, edges)


def _process_virtual_links_hdf5(hdata: HeteroData, edges):
    """Process virtual links from HDF5 format."""
    if 'generator_link' in edges:
        gen_link = edges['generator_link']
        senders = gen_link['senders'][()]
        receivers = gen_link['receivers'][()]
        hdata['generator', 'generator_link', 'bus'].edge_index = torch.stack([
            torch.tensor(senders, dtype=torch.long),
            torch.tensor(receivers, dtype=torch.long)
        ], dim=0)
        hdata['bus', 'generator_link', 'generator'].edge_index = torch.stack([
            torch.tensor(receivers, dtype=torch.long),
            torch.tensor(senders, dtype=torch.long)
        ], dim=0)

    if 'load_link' in edges:
        load_link = edges['load_link']
        senders = load_link['senders'][()]
        receivers = load_link['receivers'][()]
        hdata['load', 'load_link', 'bus'].edge_index = torch.stack([
            torch.tensor(senders, dtype=torch.long),
            torch.tensor(receivers, dtype=torch.long)
        ], dim=0)
        hdata['bus', 'load_link', 'load'].edge_index = torch.stack([
            torch.tensor(receivers, dtype=torch.long),
            torch.tensor(senders, dtype=torch.long)
        ], dim=0)

    if 'shunt_link' in edges:
        shunt_link = edges['shunt_link']
        senders = shunt_link['senders'][()]
        receivers = shunt_link['receivers'][()]
        if len(senders) > 0:
            hdata['shunt', 'shunt_link', 'bus'].edge_index = torch.stack([
                torch.tensor(senders, dtype=torch.long),
                torch.tensor(receivers, dtype=torch.long)
            ], dim=0)
            hdata['bus', 'shunt_link', 'shunt'].edge_index = torch.stack([
                torch.tensor(receivers, dtype=torch.long),
                torch.tensor(senders, dtype=torch.long)
            ], dim=0)
        else:
            hdata['shunt', 'shunt_link', 'bus'].edge_index = torch.empty((2, 0), dtype=torch.long)
            hdata['bus', 'shunt_link', 'shunt'].edge_index = torch.empty((2, 0), dtype=torch.long)


def process_hdf5_file(h5_file, n_jobs=1):
    """Process a single HDF5 file.

    Args:
        h5_file (str): Path to the HDF5 file.
        n_jobs (int): Number of jobs for parallel processing. (default: 1)

    Returns:
        List[HeteroData]: List of processed data objects.
    """
    with h5py.File(h5_file, 'r') as f:
        scenario_keys = list(f.keys())

    if n_jobs == 1:
        data_list = []
        with h5py.File(h5_file, 'r') as f:
            for scenario_key in scenario_keys:
                try:
                    scenario = f[scenario_key]
                    hdata = process_hdf5_scenario(scenario, scenario_key)
                    if hdata is not None:
                        if isinstance(hdata, list):
                            data_list.extend(hdata)
                        else:
                            data_list.append(hdata)
                except Exception as e:
                    print(f"Error processing scenario {scenario_key}: {e}")
                    continue
        return data_list
    else:
        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(_process_hdf5_scenario_from_path)(h5_file, key)
            for key in scenario_keys
        )
        
        flattened = []
        for r in results:
            if r is None:
                continue
            if isinstance(r, list):
                flattened.extend(r)
            else:
                flattened.append(r)
        return flattened
