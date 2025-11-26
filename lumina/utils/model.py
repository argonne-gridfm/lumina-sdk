"""
Copyright (c) 2025 by Argonne National Laboratory.
All rights reserved.
"""

import random
import numpy as np
import torch


def set_seed(seed):
    """ Set random seeds for reproducibility.

    Args:
        seed (int): Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def LagM_generate(vio_lag, device):
    """ Lagrangian multipliers for multi-topology

    Args:
        vil_lag (dict):
        device (torch.device):

    Returns:
        (tuple): inequality line, equality active power, equality reactive power Lagrangian multipliers
    """
    ineq_LagM_line = torch.ones(1, 2 * vio_lag.nl).to(device=device)
    eq_LagM_active = torch.ones(1, vio_lag.nbus).to(device=device)
    eq_LagM_reactive = torch.ones(1, vio_lag.nbus).to(device=device)
    return ineq_LagM_line, eq_LagM_active, eq_LagM_reactive


def dict_agg(stats, key, value, op='concat'):
    """ Aggregate or update a dictionary entry by summing or concatenating values in place

    Args:
        stats (dict): Dictionary to be updated.
        key (str): Key in the dictionary.
        value (np.ndarray): Value to be added or concatenated.
        op (str): Operation type, either 'sum' or 'concat'.
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
