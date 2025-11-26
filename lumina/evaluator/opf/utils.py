"""
Utility functions for working with physics-informed losses in ACOPF problems.

This module provides helper functions to extract network parameters from OPF datasets
and create data structures compatible with the physics loss functions.

Copyright (c) 2025, Argonne National Laboratory
All rights reserved.
"""

import warnings
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch


def extract_network_parameters_from_batch(batch, device: torch.device = None) -> Dict:
    """
    Extract network parameters from a batch of OPF data.

    Args:
        batch: Batch from OPFDataset containing heterogeneous graph data
        device: Target device for tensors

    Returns:
        Dictionary containing extracted network parameters
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    extracted_data = {}

    try:
        # Extract load data (pd, qd)
        if 'load' in batch.x_dict:
            load_data = batch['load'].x  # Shape: [n_loads, features]
            extracted_data['pd'] = load_data[:, 0].to(device)  # Active power demand
            extracted_data['qd'] = load_data[:, 1].to(device)  # Reactive power demand

            # Get load bus indices from edge connections
            if ('load', 'load_link', 'bus') in batch.edge_index_dict:
                # Load to bus connections - get bus indices
                load_bus_edges = batch[('load', 'load_link', 'bus')].edge_index
                extracted_data['load_bus_indices'] = load_bus_edges[1, :].to(device)  # Bus indices
            elif ('bus', 'load_link', 'load') in batch.edge_index_dict:
                # Bus to load connections - get bus indices
                bus_load_edges = batch[('bus', 'load_link', 'load')].edge_index
                extracted_data['load_bus_indices'] = bus_load_edges[0, :].to(device)  # Bus indices

        # Extract generator bus indices
        if ('generator', 'generator_link', 'bus') in batch.edge_index_dict:
            gen_bus_edges = batch[('generator', 'generator_link', 'bus')].edge_index
            extracted_data['gen_bus_indices'] = gen_bus_edges[1, :].to(device)  # Bus indices
        elif ('bus', 'generator_link', 'generator') in batch.edge_index_dict:
            bus_gen_edges = batch[('bus', 'generator_link', 'generator')].edge_index
            extracted_data['gen_bus_indices'] = bus_gen_edges[0, :].to(device)  # Bus indices

        # Extract line edge indices for thermal limits
        if ('bus', 'ac_line', 'bus') in batch.edge_index_dict:
            line_edges = batch[('bus', 'ac_line', 'bus')].edge_index
            extracted_data['line_edge_index'] = line_edges.to(device)

            # Extract line limits from edge attributes if available
            if hasattr(batch[('bus', 'ac_line', 'bus')], 'edge_attr'):
                line_attr = batch[('bus', 'ac_line', 'bus')].edge_attr
                if line_attr.size(1) > 6:  # Assuming thermal limit is 7th column (index 6)
                    extracted_data['line_limits'] = line_attr[:, 6].to(device)

    except Exception as e:
        warnings.warn(f"Error extracting network parameters from batch: {e}")

    return extracted_data


def extract_voltage_and_generation_limits_from_batch(batch, device: torch.device = None) -> Tuple[Dict, Dict]:
    """
    Extract voltage and generation limits from batch data.

    Args:
        batch: Batch from OPFDataset
        device: Target device for tensors

    Returns:
        Tuple of (voltage_limits dict, generation_limits dict)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    voltage_limits = {}
    generation_limits = {}

    try:
        # Extract voltage limits from bus data
        if 'bus' in batch.x_dict:
            bus_data = batch['bus'].x
            # Assuming bus features include [base_kv, vmin, vmax, bus_type_onehot...]
            if bus_data.size(1) >= 3:
                voltage_limits['vmin'] = bus_data[:, 1].to(device)
                voltage_limits['vmax'] = bus_data[:, 2].to(device)

        # Extract generation limits from generator data
        if 'generator' in batch.x_dict:
            gen_data = batch['generator'].x
            # Assuming generator features include [mbase, pg, pmin, pmax, qg, qmin, qmax, vg, costs...]
            if gen_data.size(1) >= 7:
                generation_limits['pmin'] = gen_data[:, 2].to(device)
                generation_limits['pmax'] = gen_data[:, 3].to(device)
                generation_limits['qmin'] = gen_data[:, 5].to(device)
                generation_limits['qmax'] = gen_data[:, 6].to(device)

    except Exception as e:
        warnings.warn(f"Error extracting limits from batch: {e}")

    return voltage_limits, generation_limits


def extract_generation_costs_from_batch(batch, device: torch.device = None) -> Optional[torch.Tensor]:
    """
    Extract generation cost coefficients from batch data.

    Args:
        batch: Batch from OPFDataset
        device: Target device for tensors

    Returns:
        Tensor of generation cost coefficients or None
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        if 'generator' in batch.x_dict:
            gen_data = batch['generator'].x
            # Assuming cost coefficients are the last 3 features
            if gen_data.size(1) >= 3:
                return gen_data[:, -3:].to(device)

    except Exception as e:
        warnings.warn(f"Error extracting generation costs from batch: {e}")

    return None


def denormalize_predictions(
    predictions: Dict[str, torch.Tensor],
    batch,
    voltage_range: Tuple[float, float] = (0.95, 1.05),
    angle_range: Tuple[float, float] = (-180, 180),
    power_range: Tuple[float, float] = (0, 100)  # Will be replaced by actual limits
) -> Dict[str, torch.Tensor]:
    """
    Denormalize model predictions from [0,1] range to physical units.

    Args:
        predictions: Normalized predictions from model
        batch: Batch containing limit information
        voltage_range: Default voltage magnitude range (per unit)
        angle_range: Default voltage angle range (degrees)
        power_range: Default power range (MW/MVAr)

    Returns:
        Denormalized predictions in physical units
    """
    denorm_predictions = {}

    # Denormalize bus predictions (voltage magnitude and angle)
    if 'bus' in predictions:
        bus_pred = predictions['bus'].clone()

        # Extract actual limits from batch if available
        if 'bus' in batch.x_dict and batch['bus'].x.size(1) >= 3:
            vmin = batch['bus'].x[:, 1]
            vmax = batch['bus'].x[:, 2]

            # Denormalize voltage magnitude
            vm_denorm = bus_pred[..., 0] * (vmax - vmin) + vmin

        else:
            # Use default range
            vm_denorm = bus_pred[..., 0] * (voltage_range[1] - voltage_range[0]) + voltage_range[0]

        # Denormalize voltage angle
        va_denorm = bus_pred[..., 1] * (angle_range[1] - angle_range[0]) + angle_range[0]

        denorm_predictions['bus'] = torch.stack([vm_denorm, va_denorm], dim=-1)

    # Denormalize generator predictions (active and reactive power)
    if 'generator' in predictions:
        gen_pred = predictions['generator'].clone()

        # Extract actual limits from batch if available
        if 'generator' in batch.x_dict and batch['generator'].x.size(1) >= 7:
            pmin = batch['generator'].x[:, 2]
            pmax = batch['generator'].x[:, 3]
            qmin = batch['generator'].x[:, 5]
            qmax = batch['generator'].x[:, 6]

            # Denormalize active power
            pg_denorm = gen_pred[..., 0] * (pmax - pmin) + pmin

            # Denormalize reactive power
            qg_denorm = gen_pred[..., 1] * (qmax - qmin) + qmin

        else:
            # Use default range
            pg_denorm = gen_pred[..., 0] * (power_range[1] - power_range[0]) + power_range[0]
            qg_denorm = gen_pred[..., 1] * (power_range[1] - power_range[0]) + power_range[0]

        denorm_predictions['generator'] = torch.stack([pg_denorm, qg_denorm], dim=-1)

    return denorm_predictions
