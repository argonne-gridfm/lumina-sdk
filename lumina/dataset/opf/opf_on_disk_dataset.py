"""On-disk OPF dataset backed by SQLite/RocksDB."""

import os
import os.path as osp
import shutil
import stat
import warnings
from glob import glob
from typing import Callable, List, Literal, Optional

import torch
import tqdm
from torch_geometric.data import Data, OnDiskDataset, RocksDatabase, SQLiteDatabase, download_url, extract_tar
from torch_geometric.data.dataset import Dataset

from lumina.dataset.opf.opf_dataset import process_json_file
from lumina.utils.graph_utils import OPFHomoWrapper


class OPFSQLiteDatabase(SQLiteDatabase):
    def __init__(
        self,
        path: str,
        name: str,
        schema: object = object,
        timeout_sec: float = 600.0,
        busy_timeout_ms: Optional[int] = None,
        journal_mode: Optional[str] = "WAL",
        synchronous: Optional[str] = "NORMAL",
    ) -> None:
        self._timeout_sec = float(timeout_sec) if timeout_sec is not None else None
        if busy_timeout_ms is None and self._timeout_sec is not None:
            busy_timeout_ms = int(self._timeout_sec * 1000)
        self._busy_timeout_ms = int(busy_timeout_ms) if busy_timeout_ms is not None else None
        self._journal_mode = journal_mode
        self._synchronous = synchronous
        super().__init__(path=path, name=name, schema=schema)

    def connect(self) -> None:
        import sqlite3

        timeout = self._timeout_sec if self._timeout_sec is not None else 5.0
        self._connection = sqlite3.connect(self.path, timeout=timeout)
        self._cursor = self._connection.cursor()
        if self._busy_timeout_ms is not None:
            self._connection.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")
        if self._journal_mode:
            self._connection.execute(f"PRAGMA journal_mode={self._journal_mode}")
        if self._synchronous:
            self._connection.execute(f"PRAGMA synchronous={self._synchronous}")


class OPFOnDiskDataset(OnDiskDataset):
    r"""On-disk OPF dataset backed by SQLite/RocksDB.

    Stores individual HeteroData samples in a database to avoid loading the
    full dataset into CPU memory.
    """

    url = "https://storage.googleapis.com/gridopt-dataset"

    def __init__(
        self,
        root: str,
        case_name: Literal[
            "pglib_opf_case14_ieee",
            "pglib_opf_case30_ieee",
            "pglib_opf_case57_ieee",
            "pglib_opf_case118_ieee",
            "pglib_opf_case500_goc",
            "pglib_opf_case2000_goc",
            "pglib_opf_case4661_sdet",
            "pglib_opf_case6470_rte",
            "pglib_opf_case10000_goc",
            "pglib_opf_case13659_pegase",
        ] = "pglib_opf_case14_ieee",
        group_id: int = 0,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
        keep_temp: bool = False,
        n_jobs: int = -1,
        local_raw_folder: str = None,
        backend: str = "sqlite",
        schema: object = object,
        log: bool = True,
        write_batch_size: int = 128,
        sqlite_timeout_sec: float = 600.0,
        sqlite_busy_timeout_ms: Optional[int] = None,
        sqlite_journal_mode: Optional[str] = "WAL",
        sqlite_synchronous: Optional[str] = "NORMAL",
    ) -> None:
        if backend not in OnDiskDataset.BACKENDS:
            raise ValueError(
                f"Database backend must be one of {set(OnDiskDataset.BACKENDS.keys())} "
                f"(got '{backend}')"
            )

        self.backend = backend
        self.schema = schema
        self._db = None
        self._numel = None

        self.case_name = case_name
        self.group_id = int(group_id)

        self._raw_root = osp.join(root, "OPFData/raw")
        self._processed_root = osp.join(root, "OPFData/on_disk")
        self._release = "dataset_release_1"
        self.n_jobs = n_jobs
        self.keep_temp = keep_temp
        self.local_raw_folder = local_raw_folder
        self.write_batch_size = max(1, int(write_batch_size))
        self.sqlite_timeout_sec = float(sqlite_timeout_sec)
        self.sqlite_busy_timeout_ms = (
            int(sqlite_busy_timeout_ms) if sqlite_busy_timeout_ms is not None else None
        )
        self.sqlite_journal_mode = sqlite_journal_mode
        self.sqlite_synchronous = sqlite_synchronous

        Dataset.__init__(
            self,
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            log=log,
            force_reload=force_reload,
        )

        if osp.exists(self.processed_dir):
            try:
                current_mode = os.stat(self.processed_dir).st_mode
                os.chmod(self.processed_dir, current_mode | stat.S_IWGRP)
            except OSError as exc:
                warnings.warn(
                    f"Failed to set group write permission on {self.processed_dir}: {exc}"
                )

    @property
    def raw_dir(self) -> str:
        return osp.join(self._raw_root, self._release)

    @property
    def processed_dir(self) -> str:
        return osp.join(self._processed_root, self._release, self.case_name)

    @property
    def tmp_dir(self) -> str:
        return osp.join(self.raw_dir, "gridopt-dataset-tmp", self._release, self.case_name)

    @property
    def raw_file_names(self) -> List[str]:
        return [f"{self.case_name}_{self.group_id}.tar.gz"]

    @property
    def processed_file_names(self) -> List[str]:
        return [self._db_filename()]

    def _db_filename(self) -> str:
        if self.backend == "rocksdb":
            return f"group_{self.group_id}.rocksdb"
        return f"group_{self.group_id}.{self.backend}.db"

    @property
    def db(self):
        if self._db is not None:
            return self._db

        os.makedirs(self.processed_dir, exist_ok=True)
        path = self.processed_paths[0]

        if self.backend == "sqlite":
            self._db = OPFSQLiteDatabase(
                path=path,
                name=self.__class__.__name__,
                schema=self.schema,
                timeout_sec=self.sqlite_timeout_sec,
                busy_timeout_ms=self.sqlite_busy_timeout_ms,
                journal_mode=self.sqlite_journal_mode,
                synchronous=self.sqlite_synchronous,
            )
        else:
            self._db = RocksDatabase(path=path, schema=self.schema)

        self._numel = len(self._db)
        return self._db

    def download(self) -> None:
        self.download_and_extract(self.raw_file_names[0])

    def download_and_extract(self, name: str) -> None:
        url = f"{self.url}/{self._release}/{name}"
        path = download_url(url, self.raw_dir)
        extract_tar(path, self.raw_dir)

    def _clear_processed(self) -> None:
        path = self.processed_paths[0]
        if osp.isdir(path):
            shutil.rmtree(path)
            return
        if osp.exists(path):
            os.remove(path)
            for suffix in ("-wal", "-shm", "-journal"):
                sidecar = f"{path}{suffix}"
                if osp.exists(sidecar):
                    os.remove(sidecar)

    def process(self) -> None:
        if osp.exists(self.processed_paths[0]):
            self._clear_processed()

        if not osp.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        try:
            self.process_json_group(self.group_id)
        except Exception as exc:
            print(f"Error processing group {self.group_id}: {exc}")
            raise exc

        if not self.keep_temp:
            shutil.rmtree(osp.join(self.raw_dir, "gridopt-dataset-tmp"))

    def process_json_group(self, group_id: int) -> None:
        group_json_files = glob(osp.join(self.tmp_dir, f"group_{group_id}", "*.json"))
        if len(group_json_files) < 15000:
            extract_tar(osp.join(self.raw_dir, self.raw_file_names[0]), self.raw_dir)
            group_json_files = glob(osp.join(self.tmp_dir, f"group_{group_id}", "*.json"))

        batch = []
        for json_file in tqdm.tqdm(group_json_files, desc=f"Group {group_id}"):
            data = process_json_file(json_file)
            if data is None:
                continue
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            batch.append(data)
            if len(batch) >= self.write_batch_size:
                self.extend(batch)
                batch = []

        if batch:
            self.extend(batch)

    def metadata(self):
        sample = self.get(0)

        def node_dim(node_type: str) -> int:
            if node_type not in getattr(sample, "node_types", []):
                return 0
            store = sample[node_type]
            x = getattr(store, "x", None)
            if torch.is_tensor(x):
                return int(x.size(1)) if x.ndim > 1 else 1
            return 0

        def edge_dim(edge_type) -> int:
            if edge_type not in getattr(sample, "edge_types", []):
                return 0
            store = sample[edge_type]
            edge_attr = getattr(store, "edge_attr", None)
            if torch.is_tensor(edge_attr):
                return int(edge_attr.size(1)) if edge_attr.ndim > 1 else 1
            return 0

        return {
            "nodes": {
                "bus": node_dim("bus"),
                "generator": node_dim("generator"),
                "load": node_dim("load"),
                "shunt": node_dim("shunt"),
            },
            "edges": {
                ("bus", "ac_line", "bus"): edge_dim(("bus", "ac_line", "bus")),
                ("bus", "transformer", "bus"): edge_dim(("bus", "transformer", "bus")),
                ("generator", "generator_link", "bus"): 0,
                ("bus", "generator_link", "generator"): 0,
                ("load", "load_link", "bus"): 0,
                ("bus", "load_link", "load"): 0,
                ("shunt", "shunt_link", "bus"): 0,
                ("bus", "shunt_link", "shunt"): 0,
            },
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({len(self)}, "
            f"case_name={self.case_name})"
        )


class OPFOnDiskHomogeneousDataset(OPFOnDiskDataset):
    r"""On-disk OPF dataset that precomputes homogeneous graphs."""

    def __init__(
        self,
        *args,
        add_node_type: bool = True,
        add_edge_type: bool = True,
        sanitize_targets: bool = True,
        log_bad_targets: bool = True,
        max_bad_target_logs: int = 1,
        processed_suffix: str = "homo",
        attach_full_edge_attr: bool = False,
        prune_homo: bool = True,
        storage_dtype: Optional[str] = "float16",
        restore_fp32: bool = True,
        **kwargs,
    ) -> None:
        self._processed_suffix = processed_suffix or "homo"
        self._homo_wrapper = OPFHomoWrapper(
            add_node_type=add_node_type,
            add_edge_type=add_edge_type,
            attach_full_edge_attr=attach_full_edge_attr,
        )
        self._sanitize_targets = bool(sanitize_targets)
        self._log_bad_targets = bool(log_bad_targets)
        self._max_bad_target_logs = int(max_bad_target_logs)
        self._bad_target_logs = 0
        self._prune_homo = bool(prune_homo)
        self._storage_dtype = self._resolve_dtype(storage_dtype)
        self._restore_fp32 = bool(restore_fp32)

        user_pre_transform = kwargs.pop("pre_transform", None)
        user_transform = kwargs.pop("transform", None)

        def pre_transform(data):
            homo_data = self._homo_wrapper.convert(data)
            self._copy_graph_attrs(data, homo_data)
            self._sanitize_homo_targets(homo_data)
            if user_pre_transform is not None:
                homo_data = user_pre_transform(homo_data)
            if self._prune_homo:
                homo_data = self._prune_homo_data(homo_data)
            if self._storage_dtype is not None:
                homo_data = self._cast_homo_data(homo_data, self._storage_dtype)
            return homo_data

        def transform(data):
            if (
                self._restore_fp32
                and self._storage_dtype is not None
                and self._storage_dtype != torch.float32
            ):
                data = self._cast_homo_data(data, torch.float32)
            if user_transform is not None:
                data = user_transform(data)
            return data

        super().__init__(*args, pre_transform=pre_transform, transform=transform, **kwargs)

    @property
    def processed_dir(self) -> str:
        release = self._release
        if self._processed_suffix:
            release = f"{release}_{self._processed_suffix}"
        return osp.join(self._processed_root, release, self.case_name)

    def _resolve_dtype(self, dtype):
        if dtype is None:
            return None
        if isinstance(dtype, torch.dtype):
            return dtype
        if isinstance(dtype, str):
            key = dtype.strip().lower()
            if key in {"none", "null", ""}:
                return None
            if key in {"fp16", "float16"}:
                return torch.float16
            if key in {"bf16", "bfloat16"}:
                return torch.bfloat16
            if key in {"fp32", "float32"}:
                return torch.float32
        raise ValueError(f"Unsupported storage_dtype: {dtype}")

    def _cast_homo_data(self, data: Data, dtype: torch.dtype) -> Data:
        for key in ("x", "edge_attr", "edge_attr_full", "y"):
            value = getattr(data, key, None)
            if torch.is_tensor(value) and value.is_floating_point():
                setattr(data, key, value.to(dtype))
        return data

    def _prune_homo_data(self, data: Data) -> Data:
        keep = {}
        for key in ("x", "edge_index", "edge_attr", "edge_attr_full", "y", "y_mask", "node_type", "edge_type"):
            value = getattr(data, key, None)
            if value is not None:
                keep[key] = value
        pruned = Data(**keep)
        for key in ("node_type_names", "edge_type_names", "baseMVA", "base_mva"):
            if hasattr(data, key):
                setattr(pruned, key, getattr(data, key))
        return pruned

    def _copy_graph_attrs(self, hetero_data, homo_data):
        if hasattr(hetero_data, "baseMVA"):
            homo_data.baseMVA = hetero_data.baseMVA
        elif hasattr(hetero_data, "base_mva"):
            homo_data.baseMVA = hetero_data.base_mva

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
                f"[OPFOnDiskHomogeneousDataset] Non-finite targets: "
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
