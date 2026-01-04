"""Utilities for tagging OPF samples with a case identifier."""

import os
from typing import Iterable

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

try:
    import torch.distributed as dist
except Exception:  # pragma: no cover - distributed optional
    dist = None


def attach_case_id(sample, case_id):
    """Attach a case_id tensor to a PyG Data/HeteroData sample."""
    if sample is None:
        return sample
    try:
        sample.case_id = torch.tensor(int(case_id), dtype=torch.long)
    except (TypeError, ValueError):
        return sample
    return sample


class CaseTaggedDataset(Dataset):
    """Dataset wrapper that stamps each sample with a case_id."""

    def __init__(self, dataset, case_id):
        self.dataset = dataset
        self.case_id = int(case_id)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return attach_case_id(sample, self.case_id)

    def __getattr__(self, name):
        if name in {"dataset", "case_id"}:
            raise AttributeError(name)
        return getattr(self.dataset, name)


class CaseTaggedIterableDataset(IterableDataset):
    """Iterable dataset wrapper that stamps each sample with a case_id."""

    def __init__(self, dataset: Iterable, case_id: int):
        super().__init__()
        self.dataset = dataset
        self.case_id = int(case_id)

    def __iter__(self):
        for sample in self.dataset:
            yield attach_case_id(sample, self.case_id)

    def __len__(self):
        if hasattr(self.dataset, "__len__"):
            return len(self.dataset)
        raise TypeError("Wrapped dataset length is unknown.")

    def __getattr__(self, name):
        if name in {"dataset", "case_id"}:
            raise AttributeError(name)
        return getattr(self.dataset, name)


class LimitedIterableDataset(IterableDataset):
    """Iterable dataset wrapper that yields at most max_samples across workers."""

    def __init__(self, dataset: Iterable, max_samples: int):
        super().__init__()
        self.dataset = dataset
        self.max_samples = int(max_samples)

    def _dist_info(self):
        if dist is not None and dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        return rank, world_size

    def __iter__(self):
        if self.max_samples <= 0:
            yield from self.dataset
            return
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        rank, world_size = self._dist_info()
        total_workers = max(1, num_workers * world_size)
        global_worker_id = rank * num_workers + worker_id
        base = self.max_samples // total_workers
        remainder = self.max_samples % total_workers
        local_limit = base + (1 if global_worker_id < remainder else 0)
        if local_limit <= 0:
            return
        count = 0
        for sample in self.dataset:
            if count >= local_limit:
                break
            yield sample
            count += 1

    def __len__(self):
        if self.max_samples <= 0:
            return len(self.dataset)
        try:
            total = len(self.dataset)
        except TypeError:
            return self.max_samples
        return min(int(total), int(self.max_samples))

    def __getattr__(self, name):
        if name in {"dataset", "max_samples"}:
            raise AttributeError(name)
        return getattr(self.dataset, name)
