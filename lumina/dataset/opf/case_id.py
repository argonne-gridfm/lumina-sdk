"""Utilities for tagging OPF samples with a case identifier."""

from typing import Iterable

import torch
from torch.utils.data import Dataset, IterableDataset


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
