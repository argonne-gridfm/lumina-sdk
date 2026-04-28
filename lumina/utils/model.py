"""
Copyright (c) 2025 by Argonne National Laboratory.
All rights reserved.
"""

import random
import numpy as np
import torch


def set_seed(seed):
    """Set random seeds for reproducibility across all backends.

    Configures Python ``random``, NumPy, PyTorch CPU, and (if available)
    PyTorch CUDA random number generators. Also sets cuDNN to deterministic
    mode.

    Args:
        seed (int): Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def dict_agg(stats, key, value, op='concat'):
    """Aggregate a value into a dictionary entry by summing or concatenating.

    If *key* already exists in *stats*, the value is combined with the
    existing entry using the specified operation.  Otherwise, the value is
    stored directly.

    Args:
        stats (dict): Dictionary to update in place.
        key (str): Key in the dictionary.
        value (numpy.ndarray): Value to add or concatenate.
        op (str): Operation type -- ``'sum'`` for element-wise addition,
            ``'concat'`` for ``numpy.concatenate`` along axis 0.

    Raises:
        NotImplementedError: If *op* is not ``'sum'`` or ``'concat'``.
    """
    # Modifies stats in place
    if key in stats.keys():
        if op == 'sum':
            stats[key] += value
        elif op == 'concat':
            stats[key] = np.concatenate((stats[key], value), axis=0)
        else:
            raise NotImplementedError
    else:
        stats[key] = value
