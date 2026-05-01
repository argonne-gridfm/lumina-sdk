"""Reusable PyG-style transforms for :class:`OPFDataset`."""

import torch
from torch_geometric.data import HeteroData


def to_float32(data: HeteroData) -> HeteroData:
    """Cast all ``float64`` tensors in a ``HeteroData`` sample to ``float32``.

    OPFDataset stores features as ``float64`` (numpy default), but PyTorch
    models are typically built with ``float32`` weights. Pass this as the
    ``transform=`` argument to :class:`OPFDataset` to avoid casting in every
    training/eval loop:

        >>> from lumina.dataset.opf.opf_dataset import OPFDataset
        >>> from lumina.dataset.opf.transforms import to_float32
        >>> ds = OPFDataset(root='./opf_data', transform=to_float32)
    """
    for store in data.stores:
        for key, val in list(store.items()):
            if torch.is_tensor(val) and val.dtype == torch.float64:
                store[key] = val.float()
    return data
