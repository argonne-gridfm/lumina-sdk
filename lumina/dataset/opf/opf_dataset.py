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
import warnings
from glob import glob
from typing import Callable, Dict, List, Literal, Optional, Union

import numpy as np
import torch
import tqdm
from joblib import Parallel, delayed, parallel_backend
from torch.utils.data import ConcatDataset
from torch_geometric.data import (HeteroData, InMemoryDataset, download_url,
                                  extract_tar)

from .utils import extract_edge_index, extract_edge_index_rev


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
        >>> from fm4g import OPFDataset
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

    @property
    def raw_dir(self) -> str:
        r""" Raw data folder. """
        return osp.join(self.root, self._release, self.case_name, 'raw')

    @property
    def processed_dir(self) -> str:
        r""" Processed data folder. """
        return osp.join(self.root, self._release, self.case_name, "processed")

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

    def process_json_group(self, group_id: int):
        r""" Process a single group of files, save processed data to disk.

        Args:
            group_id (int): Group id.
        """

        group_json_files = glob(osp.join(self.tmp_dir, f'group_{group_id}', '*.json'))
        #
        if len(group_json_files) < 15000:
            extract_tar(osp.join(self.raw_dir, self.raw_file_names[group_id]), self.raw_dir)
            group_json_files = glob(osp.join(self.tmp_dir, f'group_{group_id}', '*.json'))

        data_list = Parallel(n_jobs=self.n_jobs, backend="threading")(
            delayed(process_json_file)(fn) for fn in tqdm.tqdm(group_json_files, desc=f"Group {group_id}"))

        if self.pre_filter is not None or self.pre_transform is not None:
            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]
            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

        self.data, self.slices = self.collate(data_list)

        torch.save((self._data, self.slices), osp.join(self.processed_dir, f'group_{group_id}.pt'))

        # NOTE: remove `save_pkl` - data_list to pickle file format.
        # if self.save_pkl:
        #     with open(osp.join(self.processed_dir, f'group_{group_id}.pkl'), 'wb') as f:
        #         pickle.dump(data_list, f)

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
            "nodes": {"bus": self.data['bus'].x.size(1),
                      "generator": self.data['generator'].x.size(1),
                      "load": self.data['load'].x.size(1),
                      "shunt": self.data['shunt'].x.size(1)},
            "edges": {
                ('bus', 'ac_line', 'bus'): self.data['bus', 'ac_line', 'bus'].edge_attr.size(1),
                ('bus', 'transformer', 'bus'): self.data['bus', 'transformer', 'bus'].edge_attr.size(1),
                ('generator', 'generator_link', 'bus'): 0,
                ('bus', 'generator_link', 'generator'): 0,
                ('load', 'load_link', 'bus'): 0,
                ('bus', 'load_link', 'load'): 0,
                ('shunt', 'shunt_link', 'bus'): 0,
                ('bus', 'shunt_link', 'shunt'): 0
            }
        }

    def __repr__(self) -> str:
        r""" Returns the string representation of the dataset. """
        return (f'{self.__class__.__name__}({len(self)}, '
                f'case_name={self.case_name}, '
                f'topological_perturbations={self.topological_perturbations})')


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
    def from_case_groups(cls, root: str, case_name: str, group_ids: List[int], **kwargs):
        r"""Create OPFMultiDataset from multiple groups of the same case.

        Args:
            root (str): Root directory where the dataset should be saved.
            case_name (str): The name of the original pglib-opf case.
            group_ids (List[int]): List of group IDs to load.
            **kwargs: Additional arguments passed to OPFDataset constructor.

        Returns:
            OPFMultiDataset: Combined dataset from multiple groups.
        """
        datasets = []
        for group_id in group_ids:
            ds = OPFDataset(root=root, case_name=case_name, group_id=group_id, **kwargs)
            datasets.append(ds)
        return cls(datasets)

    @classmethod
    def from_multiple_cases(cls, root: str, case_configs: List[Dict], **kwargs):
        r"""Create OPFMultiDataset from multiple cases with their respective group IDs.

        Args:
            root (str): Root directory where the dataset should be saved.
            case_configs (List[Dict]): List of dictionaries, each containing 'case_name'
                and 'group_id' keys.
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
            ds = OPFDataset(root=root, case_name=case_name, group_id=group_id, **dataset_kwargs)
            datasets.append(ds)
        return cls(datasets)

    @classmethod
    def from_mixed_cases(cls, root: str, case_group_mapping: Dict[str, List[int]], **kwargs):
        r"""Create OPFMultiDataset from multiple groups across different cases.

        This method allows loading multiple groups from multiple cases in a single call,
        which is useful when you want to combine several groups from different cases.

        Args:
            root (str): Root directory where the dataset should be saved.
            case_group_mapping (Dict[str, List[int]]): Dictionary mapping case names to
                lists of group IDs to load for each case.
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
                ds = OPFDataset(root=root, case_name=case_name, group_id=group_id, **kwargs)
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

    def process_hdf5_group(self, group_id: int):
        r""" Process a single group of HDF5 files, save processed data to disk.
        TODO: with new data format support in .h5

        Args:
            group_id (int): Group id.
        """
        raise NotImplementedError("HDF5 processing not implemented yet.")


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


def process_hdf5_file(h5_file):
    """Process a single HDF5 file.
    TODO: with new data format support in .h5

    Args:
        h5_file (str): Path to the HDF5 file.

    Returns:
        data (HeteroData): Processed single data object.
    """
    raise NotImplementedError("HDF5 processing not implemented yet.")
