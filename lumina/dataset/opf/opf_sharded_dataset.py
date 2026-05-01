"""Sharded OPF dataset utilities and iterable dataset."""

import json
import os
import os.path as osp
import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import torch
from torch.utils.data import IterableDataset, get_worker_info

try:
    import torch.distributed as dist
except Exception:  # pragma: no cover - distributed optional
    dist = None

from torch_geometric.data import InMemoryDataset


@dataclass(frozen=True)
class ShardInfo:
    """Metadata descriptor for a single dataset shard file.

    Args:
        path (str): Absolute or relative path to the ``.pt`` shard file.
        num_samples (int): Number of samples contained in this shard.
        group_id (Optional[int]): OPF group identifier, used for filtering.
        name (Optional[str]): Human-readable shard name.
    """

    path: str
    num_samples: int
    group_id: Optional[int] = None
    name: Optional[str] = None


def _num_samples_from_slices(slices) -> int:
    if torch.is_tensor(slices):
        return int(slices.numel()) - 1
    if isinstance(slices, dict):
        for value in slices.values():
            return _num_samples_from_slices(value)
    raise ValueError("Unable to infer sample count from slices.")


def _infer_num_samples(path: str) -> int:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, tuple) and len(obj) == 2:
        _, slices = obj
        return _num_samples_from_slices(slices)
    if isinstance(obj, list):
        return len(obj)
    raise ValueError(f"Unsupported shard payload in {path}.")


def load_shard_manifest(path: str) -> Dict:
    """Load a shard manifest JSON file and annotate it with its source path.

    Args:
        path (str): Path to the manifest JSON file.

    Returns:
        Dict: Parsed manifest dictionary with an added ``_manifest_path`` key.
    """
    with open(path, "r") as f:
        manifest = json.load(f)
    manifest["_manifest_path"] = path
    return manifest


def _manifest_base_dir(manifest: Dict) -> str:
    manifest_path = manifest.get("_manifest_path") or manifest.get("manifest_path")
    manifest_dir = osp.dirname(manifest_path) if manifest_path else os.getcwd()
    base_dir = manifest.get("base_dir") or manifest_dir
    if not osp.isabs(base_dir):
        base_dir = osp.join(manifest_dir, base_dir)
    return base_dir


def build_shard_infos(manifest: Dict) -> List[ShardInfo]:
    """Build a list of ShardInfo objects from a parsed manifest dictionary.

    Resolves relative shard paths against the manifest's base directory and
    infers ``num_samples`` from the shard file when not specified.

    Args:
        manifest (Dict): Parsed shard manifest containing a ``"shards"`` list.

    Returns:
        List[ShardInfo]: Ordered list of shard descriptors.

    Raises:
        KeyError: If the manifest is missing the ``"shards"`` key or a shard
            entry is missing the ``"path"`` key.
    """
    if "shards" not in manifest:
        raise KeyError("Shard manifest missing 'shards' list.")
    base_dir = _manifest_base_dir(manifest)
    shard_infos: List[ShardInfo] = []
    for entry in manifest["shards"]:
        if "path" not in entry:
            raise KeyError("Shard entry missing 'path'.")
        path = entry["path"]
        if not osp.isabs(path):
            path = osp.join(base_dir, path)
        name = entry.get("name") or osp.basename(path)
        num_samples = entry.get("num_samples")
        if num_samples is None:
            num_samples = _infer_num_samples(path)
        group_id = entry.get("group_id")
        group_id = int(group_id) if group_id is not None else None
        shard_infos.append(
            ShardInfo(
                path=path,
                num_samples=int(num_samples),
                group_id=group_id,
                name=name,
            )
        )
    return shard_infos


def filter_shards_by_group(shards: Iterable[ShardInfo], group_ids: Optional[Iterable[int]]):
    """Filter shards to keep only those matching the given group identifiers.

    Args:
        shards (Iterable[ShardInfo]): Shard descriptors to filter.
        group_ids (Optional[Iterable[int]]): Group IDs to retain. If ``None``
            or empty, all shards are returned unfiltered.

    Returns:
        List[ShardInfo]: Filtered list of shard descriptors.

    Raises:
        ValueError: If filtering is requested but any shard lacks a
            ``group_id``.
    """
    shards = list(shards)
    if not group_ids:
        return shards
    missing = [shard for shard in shards if shard.group_id is None]
    if missing:
        raise ValueError("group_ids filtering requested but shard entries lack group_id metadata.")
    group_set = {int(group_id) for group_id in group_ids}
    return [shard for shard in shards if shard.group_id in group_set]


def resolve_split_shards(manifest: Dict, shards: List[ShardInfo], split: str) -> List[ShardInfo]:
    """Select shards belonging to a named split defined in the manifest.

    The manifest ``"splits"`` section may reference shards by integer index,
    shard name, path, or basename.

    Args:
        manifest (Dict): Parsed shard manifest containing a ``"splits"`` section.
        shards (List[ShardInfo]): Full ordered list of shard descriptors.
        split (str): Name of the split (e.g. ``"train"``, ``"val"``, ``"test"``).

    Returns:
        List[ShardInfo]: Shards belonging to the requested split.

    Raises:
        KeyError: If the split name is not found in the manifest or a shard
            reference cannot be resolved.
    """
    splits = manifest.get("splits") or {}
    if split not in splits:
        raise KeyError(f"Split '{split}' not found in shard manifest.")
    spec = splits[split]
    if not spec:
        return []
    if all(isinstance(item, int) for item in spec):
        return [shards[int(idx)] for idx in spec]
    lookup = {}
    for shard in shards:
        if shard.name:
            lookup[shard.name] = shard
        lookup[shard.path] = shard
        lookup[osp.basename(shard.path)] = shard
    selected = []
    for item in spec:
        if item not in lookup:
            raise KeyError(f"Split entry '{item}' not found among shard names.")
        selected.append(lookup[item])
    return selected


def split_shards_by_ratio(
    shards: Iterable[ShardInfo],
    train_ratio: float,
    val_ratio: float,
    seed: int = 42,
    shuffle: bool = True,
):
    """Partition shards into train/val/test splits by sample-count ratio.

    Shards are optionally shuffled and then assigned to splits greedily until
    each split's cumulative sample count reaches the target ratio.

    Args:
        shards (Iterable[ShardInfo]): Shard descriptors to partition.
        train_ratio (float): Fraction of total samples for training.
        val_ratio (float): Fraction of total samples for validation.
        seed (int): Random seed for shuffling. (default: :obj:`42`)
        shuffle (bool): Shuffle shards before splitting. (default: :obj:`True`)

    Returns:
        dict: Dictionary with ``"train"``, ``"val"``, and ``"test"`` keys,
            each mapping to a list of :class:`ShardInfo`.
    """
    shards = list(shards)
    splits = {"train": [], "val": [], "test": []}
    if not shards:
        return splits
    if shuffle:
        rng = random.Random(int(seed))
        rng.shuffle(shards)
    total_samples = sum(shard.num_samples for shard in shards)
    train_target = int(total_samples * train_ratio)
    val_target = int(total_samples * val_ratio)
    seen = 0
    for shard in shards:
        if seen < train_target:
            splits["train"].append(shard)
        elif seen < train_target + val_target:
            splits["val"].append(shard)
        else:
            splits["test"].append(shard)
        seen += shard.num_samples
    if not splits["train"]:
        splits["train"].append(shards[0])
        splits["test"] = shards[1:]
    return splits


class _InMemoryShardAccessor:
    def __init__(self, data, slices):
        self._data = data
        self.slices = slices

    @property
    def data(self):
        return self._data

    def get(self, idx: int):
        return InMemoryDataset.get(self, idx)

    def len(self) -> int:
        return _num_samples_from_slices(self.slices)

    def __len__(self) -> int:
        return self.len()


class OPFShardedIterableDataset(IterableDataset):
    """Iterable dataset that streams OPF samples from sharded ``.pt`` files.

    Shards are distributed across DDP ranks and DataLoader workers so that
    each global worker processes a disjoint subset. Shard order can be
    shuffled per epoch.

    Args:
        shards (Iterable[ShardInfo]): Shard descriptors to iterate over.
        shuffle_shards (bool): Shuffle shard order each epoch.
            (default: :obj:`False`)
        seed (int): Base random seed for shard shuffling.
            (default: :obj:`0`)
        transform (Optional[Callable]): Per-sample transform applied on
            iteration. (default: :obj:`None`)
    """

    def __init__(
        self,
        shards: Iterable[ShardInfo],
        shuffle_shards: bool = False,
        seed: int = 0,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.shards = list(shards)
        self.shuffle_shards = bool(shuffle_shards)
        self.seed = int(seed)
        self.epoch = 0
        self.transform = transform
        self._num_samples = self._compute_num_samples()

    def _compute_num_samples(self) -> Optional[int]:
        if not self.shards:
            return 0
        return sum(int(shard.num_samples) for shard in self.shards)

    def __len__(self) -> int:
        if self._num_samples is None:
            raise TypeError("Sharded dataset length is unknown.")
        return int(self._num_samples)

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch for deterministic shard shuffling.

        Args:
            epoch (int): Current training epoch number.
        """
        self.epoch = int(epoch)

    def _dist_info(self):
        if dist is not None and dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        return rank, world_size

    def _iter_inmemory(self, data, slices, num_samples: int):
        accessor = _InMemoryShardAccessor(data, slices)
        for idx in range(num_samples):
            sample = accessor.get(idx)
            if self.transform is not None:
                sample = self.transform(sample)
            yield sample

    def _iter_shard(self, shard: ShardInfo):
        obj = torch.load(shard.path, map_location="cpu", weights_only=False)
        try:
            if isinstance(obj, tuple) and len(obj) == 2:
                data, slices = obj
                yield from self._iter_inmemory(data, slices, shard.num_samples)
                return
            if isinstance(obj, list):
                for sample in obj:
                    if self.transform is not None:
                        sample = self.transform(sample)
                    yield sample
                return
            if isinstance(obj, dict) and "data" in obj and "slices" in obj:
                yield from self._iter_inmemory(obj["data"], obj["slices"], shard.num_samples)
                return
            raise ValueError(f"Unsupported shard payload type in {shard.path}.")
        finally:
            del obj

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        rank, world_size = self._dist_info()
        total_workers = max(1, num_workers * world_size)
        global_worker_id = rank * num_workers + worker_id

        shard_indices = list(range(len(self.shards)))
        if self.shuffle_shards:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(shard_indices)

        for local_idx, shard_idx in enumerate(shard_indices):
            if (local_idx % total_workers) != global_worker_id:
                continue
            shard = self.shards[shard_idx]
            yield from self._iter_shard(shard)

    def peek(self):
        """Return the first sample from the first shard without iterating all data.

        Returns:
            The first data sample from the dataset.

        Raises:
            RuntimeError: If no shards are available.
        """
        if not self.shards:
            raise RuntimeError("No shards available to sample from.")
        shard = self.shards[0]
        return next(self._iter_shard(shard))

    def metadata(self):
        """Return node and edge feature dimensionality metadata from the first sample.

        Returns:
            dict: Dictionary with ``"nodes"`` and ``"edges"`` keys mapping
                node types and edge types to their feature dimensions.
        """
        sample = self.peek()

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
        return f"{self.__class__.__name__}(shards={len(self.shards)}, samples={len(self)})"
