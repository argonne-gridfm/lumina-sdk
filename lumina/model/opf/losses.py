"""
Flexible Loss Function Classes for ACOPF Training

This module provides a comprehensive set of loss functions that can be used
for training neural networks on ACOPF problems. Supports various loss types
including MSE, RMSE, MAPE, SmoothL1Loss, and combinations thereof.

Copyright (c) 2025, Argonne National Laboratory
All rights reserved.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSELoss(nn.Module):
    """Root Mean Squared Error Loss.

    .. math::
        RMSE = \\sqrt{\frac{1}{N} \\sum_{i=1}^{N} (y_i - \\hat{y}_i)^2}

    Args:
        reduction (str): Reduction method to apply to the output: 'mean', 'sum',
            or 'none'. Default: 'mean'
        epsilon (float): Small value to avoid sqrt(0). Default: 1e-8
    """

    def __init__(self, reduction: str = 'mean', epsilon: float = 1e-8):
        super(RMSELoss, self).__init__()
        self.reduction = reduction
        self.epsilon = epsilon
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mse = self.mse_loss(predictions, targets)
        rmse = torch.sqrt(mse + self.epsilon)

        if self.reduction == 'mean':
            return rmse.mean()
        elif self.reduction == 'sum':
            return rmse.sum()
        elif self.reduction == 'none':
            return rmse
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")


class MAPELoss(nn.Module):
    """Mean Absolute Percentage Error Loss.

    .. math::
        MAPE = \\frac{1}{N} \\sum_{i=1}^{N} \\left| \\frac{y_i - \\hat{y}_i}{y_i} \\right|

    Args:
        reduction (str): Reduction method to apply to the output: 'mean', 'sum',
            or 'none'. Default: 'mean'
        epsilon (float): Small value to avoid division by zero. Default: 1e-8
    """

    def __init__(self, reduction: str = 'mean', epsilon: float = 1e-8):
        super(MAPELoss, self).__init__()
        self.reduction = reduction
        self.epsilon = epsilon

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        abs_error = torch.abs(predictions - targets)
        abs_target = torch.abs(targets) + self.epsilon  # Add epsilon to avoid division by zero
        mape = abs_error / abs_target

        if self.reduction == 'mean':
            return mape.mean()
        elif self.reduction == 'sum':
            return mape.sum()
        elif self.reduction == 'none':
            return mape
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")


class ACOPFLossFunction(nn.Module):
    """
    Simplified loss function class for ACOPF training using PyTorch built-in losses.

    Supports:
    - MSE (Mean Squared Error) - torch.nn.MSELoss
    - RMSE (Root Mean Squared Error) - sqrt of MSE
    - MAE (Mean Absolute Error) - torch.nn.L1Loss
    - MAPE (Mean Absolute Percentage Error) - custom implementation
    - SmoothL1Loss (Huber Loss) - torch.nn.SmoothL1Loss

    Args:
        loss_type (str): Type of loss function to use:
            - 'mse': Mean Squared Error
            - 'rmse': Root Mean Squared Error
            - 'mae': Mean Absolute Error
            - 'mape': Mean Absolute Percentage Error
            - 'smooth_l1': Smooth L1 Loss (Huber Loss)
        node_weights (dict, optional): Weights for different node types.
            Default: {'bus': 1.0, 'generator': 1.0}
        reduction (str): Reduction method ('mean', 'sum', 'none'). Default: 'mean'
        epsilon (float): Small value to avoid division by zero in MAPE. Default: 1e-8
        beta (float): Beta parameter for SmoothL1Loss. Default: 1.0
    """

    def __init__(
        self,
        loss_type: str = 'mse',
        node_weights: Optional[Dict[str, float]] = None,
        reduction: str = 'mean',
        epsilon: float = 1e-8,
        beta: float = 1.0
    ):
        super(ACOPFLossFunction, self).__init__()

        self.loss_type = loss_type
        self.reduction = reduction
        self.epsilon = epsilon
        self.beta = beta

        # Default node weights (bus and generator equally important)
        self.node_weights = node_weights or {'bus': 1.0, 'generator': 1.0}

        # Validate loss type
        valid_types = ['mse', 'rmse', 'mae', 'mape', 'smooth_l1']
        if loss_type not in valid_types:
            raise ValueError(f"Invalid loss_type '{loss_type}'. Must be one of {valid_types}")

        # Initialize PyTorch loss functions
        self._init_loss_functions()

    def _init_loss_functions(self):
        """Initialize PyTorch loss functions based on loss_type."""
        if self.loss_type == 'mse':
            self.criterion = nn.MSELoss(reduction=self.reduction)
        elif self.loss_type == 'mae':
            self.criterion = nn.L1Loss(reduction=self.reduction)
        elif self.loss_type == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss(reduction=self.reduction, beta=self.beta)
        elif self.loss_type == 'rmse':
            # RMSE is just sqrt of MSE
            self.criterion = nn.MSELoss(reduction='none')
        elif self.loss_type == 'mape':
            # MAPE needs custom implementation but we'll use it with L1Loss base
            self.criterion = nn.L1Loss(reduction='none')

    def _compute_single_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss using the configured PyTorch loss function."""
        if self.loss_type == 'mse':
            return self.criterion(predictions, targets)
        elif self.loss_type == 'mae':
            return self.criterion(predictions, targets)
        elif self.loss_type == 'smooth_l1':
            return self.criterion(predictions, targets)
        elif self.loss_type == 'rmse':
            mse = self.criterion(predictions, targets)
            rmse = torch.sqrt(mse + self.epsilon)
            return self._reduce_loss(rmse)
        elif self.loss_type == 'mape':
            # Mean Absolute Percentage Error - custom implementation
            abs_error = torch.abs(predictions - targets)
            abs_target = torch.abs(targets) + self.epsilon  # Add epsilon to avoid division by zero
            mape = abs_error / abs_target
            return self._reduce_loss(mape)
        else:
            raise ValueError(f"Unknown loss function: {self.loss_type}")

    def _reduce_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Apply reduction to loss tensor."""
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute loss for ACOPF predictions.

        Args:
            predictions (dict): Model predictions with keys like 'bus', 'generator'
            targets (dict): Ground truth targets with same keys as predictions

        Returns:
            dict: Dictionary containing:
                - 'total_loss': Overall weighted loss
                - '{node_type}_loss': Loss for each node type
                - 'loss_components': Individual loss components (if using combined loss)
        """
        total_loss = 0.0
        node_losses = {}
        loss_components = {}

        # Compute loss for each node type
        for node_type in predictions.keys():
            if node_type not in targets:
                continue

            pred = predictions[node_type]
            target = targets[node_type]

            # Compute loss using PyTorch built-in functions
            node_loss = self._compute_single_loss(pred, target)

            # Weight by node importance
            node_weight = self.node_weights.get(node_type, 1.0)

            node_losses[f"{node_type}_loss"] = node_loss
            total_loss += node_weight * node_loss

        # Prepare results
        results = {
            'total_loss': total_loss,
            **node_losses
        }

        if loss_components:
            results['loss_components'] = loss_components

        return results

    def compute_loss(self, predictions: Dict[str, torch.Tensor], batch) -> Dict[str, torch.Tensor]:
        """
        Compute loss given predictions and batch data.

        This method extracts targets from the batch and calls the forward method.

        Args:
            predictions (dict): Model predictions with keys like 'bus', 'generator'
            batch: Batch data containing targets for each node type

        Returns:
            dict: Dictionary containing loss values as returned by forward()
        """
        # Extract targets from batch
        targets = {}
        for node_type in predictions.keys():
            if hasattr(batch[node_type], 'y') and batch[node_type].y is not None:
                targets[node_type] = batch[node_type].y

        # Call the forward method
        return self.forward(predictions, targets)

    def get_loss_info(self) -> Dict:
        """Get information about the configured loss function."""
        return {
            'loss_type': self.loss_type,
            'node_weights': self.node_weights,
            'reduction': self.reduction,
            'epsilon': self.epsilon,
            'beta': self.beta
        }


class PhysicsInformedLoss(ACOPFLossFunction):
    """
    Physics-informed loss function that combines standard ML losses with physics constraints.

    This extends the basic ACOPFLossFunction to include penalty terms for constraint violations.

    Args:
        base_loss_config (dict): Configuration for base ML loss
        physics_weight (float): Weight for physics constraint penalty. Default: 1.0
        constraint_types (list): Types of constraints to include.
            Options: ['power_flow', 'line_limits', 'voltage_limits', 'generation_limits']
        penalty_method (str): How to penalize constraint violations.
            Options: 'quadratic', 'absolute', 'log_barrier'
    """

    def __init__(
        self,
        base_loss_config: Dict = None,
        physics_weight: float = 1.0,
        constraint_types: List[str] = None,
        penalty_method: str = 'quadratic',
        **kwargs
    ):
        # Initialize base loss function
        base_config = base_loss_config or {'loss_type': 'mse'}
        super().__init__(**base_config, **kwargs)

        self.physics_weight = physics_weight
        self.constraint_types = constraint_types or ['power_flow', 'line_limits']
        self.penalty_method = penalty_method

        # Constraint violation computer (to be set externally)
        self.constraint_computer = None

    def set_constraint_computer(self, constraint_computer):
        """Set the constraint violation computer."""
        self.constraint_computer = constraint_computer

    def _compute_physics_penalty(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute physics constraint penalty."""
        if self.constraint_computer is None:
            return torch.tensor(0.0, device=next(iter(predictions.values())).device)

        # Compute constraint violations
        violations = self.constraint_computer.compute_violations(predictions)

        total_penalty = 0.0
        for constraint_type in self.constraint_types:
            if constraint_type in violations:
                violation = violations[constraint_type]

                if self.penalty_method == 'quadratic':
                    penalty = torch.mean(violation ** 2)
                elif self.penalty_method == 'absolute':
                    penalty = torch.mean(torch.abs(violation))
                elif self.penalty_method == 'log_barrier':
                    # Log barrier for inequality constraints (assumes violation > 0 means infeasible)
                    penalty = -torch.mean(torch.log(torch.clamp(-violation, min=self.epsilon)))
                else:
                    raise ValueError(f"Unknown penalty method: {self.penalty_method}")

                total_penalty += penalty

        return total_penalty

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute physics-informed loss.

        Returns loss dictionary with additional 'physics_penalty' key.
        """
        # Compute base ML loss
        results = super().forward(predictions, targets)

        # Compute physics penalty
        physics_penalty = self._compute_physics_penalty(predictions)

        # Add physics penalty to total loss
        total_loss = results['total_loss'] + self.physics_weight * physics_penalty

        # Update results
        results.update({
            'total_loss': total_loss,
            'ml_loss': results['total_loss'],  # Store original ML loss
            'physics_penalty': physics_penalty
        })

        return results


class OPFLossManager(nn.Module):
    """
    Unified loss manager that supports switching between different loss types:
    - Standard ML losses (MSE, MAE, etc.)
    - Augmented Lagrangian
    - Violated (Violation-based) Lagrangian

    This provides a single interface for training with different loss formulations.

    Args:
        loss_type (str): Type of loss to use. Options:
            - 'mse', 'rmse', 'mae', 'mape', 'smooth_l1': Standard ML losses
            - 'augmented_lagrangian': Augmented Lagrangian method
            - 'violated_lagrangian': Violation-based Lagrangian method
        device (torch.device, optional): Device for computations
        lagrangian_config (dict, optional): Configuration for Lagrangian methods
        **kwargs: Additional arguments passed to the loss function
    """

    def __init__(
        self,
        loss_type: str = 'mse',
        device: Optional[torch.device] = None,
        lagrangian_config: Optional[Dict] = None,
        **kwargs
    ):
        super().__init__()

        self.loss_type = loss_type
        self.device = device or torch.device('cpu')
        lag_config = lagrangian_config or {}
        self._last_lagrangian_loss = None
        self._iters_since_lagrangian_update = 0

        # Initialize the appropriate loss function
        if loss_type == 'augmented_lagrangian':
            from .augmented_lagrangian import AugmentedLagrangianACOPF

            lagrangian_kwargs = dict(lag_config)
            self.lagrangian = AugmentedLagrangianACOPF(**lagrangian_kwargs)
            self.base_loss = ACOPFLossFunction(loss_type='mse', **kwargs)

        elif loss_type == 'violated_lagrangian':
            from .violated_lagrangian import ViolatedLagrangianACOPF

            lagrangian_kwargs = dict(lag_config)
            self.lagrangian = ViolatedLagrangianACOPF(**lagrangian_kwargs)
            self.base_loss = ACOPFLossFunction(loss_type='mse', **kwargs)

        else:
            # Standard ML loss
            self.base_loss = ACOPFLossFunction(loss_type=loss_type, **kwargs)
            self.lagrangian = None

        # Track whether Lagrangian network parameters have been initialized
        self._lagrangian_initialized = False

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        batch,
        return_info: bool = True,
        constraint_data: Optional[Dict] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Compute loss based on the configured loss type.

        Args:
            predictions: Model predictions
            batch: Batch data containing targets and inputs
            return_info: Whether to return additional loss information

        Returns:
            If return_info=False: loss tensor
            If return_info=True: (loss tensor, info dict)
        """
        # Extract targets from batch
        targets = {}
        for node_type in predictions.keys():
            if hasattr(batch[node_type], 'y') and batch[node_type].y is not None:
                targets[node_type] = batch[node_type].y

        if self.loss_type in ['augmented_lagrangian', 'violated_lagrangian']:
            # Compute base MSE loss
            base_results = self.base_loss(predictions, targets)
            mse_loss = base_results['total_loss']

            # Compute Lagrangian loss using shared constraint pipeline
            current_n_bus = batch['bus'].x.size(0)
            stored_Y = getattr(self.lagrangian, 'Y_real', None)
            need_init = (
                not self._lagrangian_initialized
                or stored_Y is None
                or stored_Y.size(0) != current_n_bus
            )
            if need_init:
                self._ensure_network_parameters(batch, predictions['bus'].device)
            constraint_batch = constraint_data or self._create_constraint_batch(batch, predictions)
            lag_loss, info = self.lagrangian(mse_loss, predictions, constraint_batch)

            if return_info:
                info.update(base_results)
                # TODO: should we use task_loss?
                info['mse_loss'] = mse_loss.item()
                return lag_loss, info
            else:
                return lag_loss
        else:
            # Standard ML loss
            results = self.base_loss(predictions, targets)
            loss = results['total_loss']

            if return_info:
                return loss, results
            else:
                return loss

    def update_lagrangian(
        self,
        constraint_violation: Optional[float] = None,
        constraints: Optional[torch.Tensor] = None,
        update_penalty: bool = True,
        is_training: bool = True,
        force: bool = False,
    ):
        """
        Update Lagrangian multipliers or penalty parameters.

        Args:
            model: Neural network model (used for violated Lagrangian updates)
            dataloader: Training dataloader (used for violated Lagrangian updates)
            constraint_violation: Current constraint violation (for augmented Lagrangian)
            constraints: Constraint vector (for updating multipliers in augmented Lagrangian)
            update_penalty: Whether to update the penalty parameter μ
            is_training: Whether updates should run (skip during eval)
            force: Bypass loss-based trigger and update immediately
        """
        if self.lagrangian is None or not is_training:
            return

        # After warmup, always update multipliers using the EMA constraints.
        constraint_tensor = constraints if constraints is not None else None
        self.lagrangian.update_multipliers(constraint_tensor)

        if self.loss_type == 'augmented_lagrangian':
            # Penalty updates are handled on epoch boundaries via step_epoch; allow
            # a manual/forced update if explicitly requested.
            if force and update_penalty and constraint_violation is not None:
                self.lagrangian.update_penalty_parameter(constraint_violation)

    def step_epoch(self):
        """Call at the end of each epoch for Lagrangian methods."""
        if self.lagrangian is not None and hasattr(self.lagrangian, 'step_epoch'):
            self.lagrangian.step_epoch()

    def _create_constraint_batch(self, batch, predictions):
        """Create constraint data using actual OPF batch fields (no random stubs)."""
        device = predictions['bus'].device
        n_bus = batch['bus'].x.size(0)

        base_mva = getattr(batch, 'baseMVA', None)
        if base_mva is None and hasattr(batch, 'base_mva'):
            base_mva = getattr(batch, 'base_mva')
        if torch.is_tensor(base_mva):
            # baseMVA comes as a length-1 tensor; extract safely
            base_mva = base_mva.view(-1)[0].item()
        base_mva = float(base_mva) if base_mva is not None else 100.0

        # Aggregate load (pd, qd) onto buses using bus->load links
        pd = torch.zeros(n_bus, device=device)
        qd = torch.zeros(n_bus, device=device)
        if ('bus', 'load_link', 'load') in getattr(batch, 'edge_types', []) and ('load' in getattr(batch, 'node_types', [])):
            load_edge = batch['bus', 'load_link', 'load'].edge_index.to(device)
            load_x = batch['load'].x.to(device)
            load_pd = load_x[:, 0]
            load_qd = load_x[:, 1] if load_x.size(1) > 1 else torch.zeros_like(load_pd)

            bus_idx = load_edge[0].clamp(max=n_bus - 1)
            load_idx = load_edge[1].clamp(max=load_pd.size(0) - 1)
            pd.index_add_(0, bus_idx, load_pd[load_idx])
            qd.index_add_(0, bus_idx, load_qd[load_idx])
        elif 'load' in getattr(batch, 'node_types', []):
            load_x = batch['load'].x.to(device)
            n_load = load_x.size(0)
            take = min(n_load, n_bus)
            pd[:take] += load_x[:take, 0]
            if load_x.size(1) > 1:
                qd[:take] += load_x[:take, 1]

        # Map generators to buses using bus->generator links
        gen_pred = predictions.get('generator', None)
        n_generators = gen_pred.size(0) if gen_pred is not None else batch['generator'].x.size(0)
        gen_bus_indices = torch.zeros(n_generators, device=device, dtype=torch.long)
        if ('bus', 'generator_link', 'generator') in getattr(batch, 'edge_types', []):
            gen_edge = batch['bus', 'generator_link', 'generator'].edge_index.to(device)
            bus_idx = gen_edge[0].clamp(max=n_bus - 1)
            gen_idx = gen_edge[1].clamp(max=n_generators - 1)
            gen_bus_indices[gen_idx] = bus_idx

        # Collect load bus indices (unique) for convenience
        load_bus_indices = None
        if ('bus', 'load_link', 'load') in getattr(batch, 'edge_types', []):
            load_bus_indices = batch['bus', 'load_link', 'load'].edge_index[0].to(device)

        # Line topology and limits
        line_edge_index = None
        line_limits = None
        if ('bus', 'ac_line', 'bus') in getattr(batch, 'edge_types', []):
            line_edge_index = batch['bus', 'ac_line', 'bus'].edge_index.to(device)
            if batch['bus', 'ac_line', 'bus'].edge_attr is not None:
                edge_attr = batch['bus', 'ac_line', 'bus'].edge_attr.to(device)
                # rate_a if available; otherwise no limits
                if edge_attr.size(1) >= 7:
                    line_limits = edge_attr[:, 6].abs()

        # Build constraint batch wrapper
        class ConstraintBatch:
            def __init__(self):
                self.baseMVA = base_mva
                self.n_bus = n_bus

            def get(self, key, default=None):
                if key == 'pd':
                    return pd
                if key == 'qd':
                    return qd
                if key == 'gen_bus_indices':
                    return gen_bus_indices
                if key == 'load_bus_indices':
                    return load_bus_indices
                if key == 'line_edge_index':
                    return line_edge_index
                if key == 'line_limits':
                    return line_limits
                return default

        return ConstraintBatch()

    def _build_admittance_from_batch(self, batch, device):
        """Approximate Y-bus (real/imag) from line r/x; ignores shunts/transformers for now."""
        if ('bus', 'ac_line', 'bus') not in batch.edge_index_dict:
            return None, None

        edge_index = batch['bus', 'ac_line', 'bus'].edge_index.to(device)
        edge_attr = batch['bus', 'ac_line', 'bus'].edge_attr
        if edge_attr is None or edge_attr.size(0) != edge_index.size(1) or edge_attr.size(1) < 6:
            return None, None

        edge_attr = edge_attr.to(device)
        r = edge_attr[:, 4]
        x = edge_attr[:, 5]
        denom = (r ** 2 + x ** 2).clamp_min(1e-6)
        y_real = r / denom
        y_imag = -x / denom

        n_bus = batch['bus'].x.size(0)
        Y_real = torch.zeros(n_bus, n_bus, device=device)
        Y_imag = torch.zeros(n_bus, n_bus, device=device)

        i = edge_index[0].clamp(max=n_bus - 1)
        j = edge_index[1].clamp(max=n_bus - 1)

        # Off-diagonal
        Y_real.index_put_((i, j), -y_real, accumulate=True)
        Y_real.index_put_((j, i), -y_real, accumulate=True)
        Y_imag.index_put_((i, j), -y_imag, accumulate=True)
        Y_imag.index_put_((j, i), -y_imag, accumulate=True)

        # Diagonal
        Y_real.index_put_((i, i), y_real, accumulate=True)
        Y_real.index_put_((j, j), y_real, accumulate=True)
        Y_imag.index_put_((i, i), y_imag, accumulate=True)
        Y_imag.index_put_((j, j), y_imag, accumulate=True)

        return Y_real, Y_imag

    def _ensure_network_parameters(self, batch, device):
        """Initialize AugLag network parameters from batch data when needed."""
        if self.lagrangian is None:
            return

        n_bus = batch['bus'].x.size(0)
        need_init = (
            getattr(self.lagrangian, 'Y_real', None) is None
            or self.lagrangian.Y_real.size(0) != n_bus
        )

        if not need_init:
            return

        Y_real, Y_imag = self._build_admittance_from_batch(batch, device)
        if Y_real is None or Y_imag is None:
            return

        line_limits = None
        if ('bus', 'ac_line', 'bus') in getattr(batch, 'edge_types', []):
            edge_attr = batch['bus', 'ac_line', 'bus'].edge_attr
            if edge_attr is not None and edge_attr.size(1) >= 7:
                line_limits = edge_attr[:, 6].abs().to(device)

        base_mva = getattr(batch, 'baseMVA', None)
        if base_mva is None and hasattr(batch, 'base_mva'):
            base_mva = getattr(batch, 'base_mva')
        if torch.is_tensor(base_mva):
            base_mva = base_mva.view(-1)[0].item()
        base_mva = float(base_mva) if base_mva is not None else 100.0

        self.lagrangian.set_network_parameters(
            Y_real=Y_real,
            Y_imag=Y_imag,
            line_limits=line_limits,
            base_mva=base_mva
        )

        # Avoid re-initializing network parameters on subsequent calls
        self._lagrangian_initialized = True
        self._lagrangian_bus_count = n_bus

    def _extract_inputs(self, batch):
        """Extract input data dictionary for violated Lagrangian."""
        inputs = {}

        # Extract load data
        if hasattr(batch, 'load') and hasattr(batch['load'], 'x'):
            inputs['load'] = batch['load'].x

        return inputs

    def get_loss_info(self) -> Dict:
        """Get information about the configured loss."""
        info = {
            'loss_type': self.loss_type,
        }

        if self.lagrangian is not None:
            if hasattr(self.lagrangian, 'mu_k'):
                info['penalty_parameter'] = self.lagrangian.mu_k
            if hasattr(self.lagrangian, 'current_epoch'):
                info['lagrangian_epoch'] = self.lagrangian.current_epoch
            if hasattr(self.lagrangian, 'warmup_epochs'):
                info['lagrangian_warmup_epochs'] = self.lagrangian.warmup_epochs
            if hasattr(self.lagrangian, 'ema_beta'):
                info['lagrangian_ema_beta'] = self.lagrangian.ema_beta

        if hasattr(self.base_loss, 'get_loss_info'):
            info.update(self.base_loss.get_loss_info())

        return info


# Convenience functions for common loss configurations
def create_mse_loss(**kwargs) -> ACOPFLossFunction:
    """Create MSE loss function."""
    return ACOPFLossFunction(loss_type='mse', **kwargs)


def create_rmse_loss(**kwargs) -> ACOPFLossFunction:
    """Create RMSE loss function."""
    return ACOPFLossFunction(loss_type='rmse', **kwargs)


def create_mape_loss(**kwargs) -> ACOPFLossFunction:
    """Create MAPE loss function."""
    return ACOPFLossFunction(loss_type='mape', **kwargs)


def create_combined_loss(weights: Dict[str, float], **kwargs) -> ACOPFLossFunction:
    """Create combined loss function with specified weights."""
    return ACOPFLossFunction(loss_type=weights, **kwargs)


def create_robust_loss(**kwargs) -> ACOPFLossFunction:
    """Create robust loss combining MSE and SmoothL1."""
    return ACOPFLossFunction(
        loss_type={'mse': 0.7, 'smooth_l1': 0.3},
        **kwargs
    )


# Example usage and testing
if __name__ == "__main__":
    # Test the loss functions
    batch_size = 32
    n_bus = 14
    n_gen = 5

    # Create dummy predictions and targets
    predictions = {
        'bus': torch.randn(batch_size, 2),  # VM, VA
        'generator': torch.randn(batch_size, 2)  # PG, QG
    }

    targets = {
        'bus': torch.randn(batch_size, 2),
        'generator': torch.randn(batch_size, 2)
    }

    # Test different loss functions
    print("Testing ACOPF Loss Functions")
    print("=" * 40)

    # MSE Loss
    mse_loss = create_mse_loss()
    result = mse_loss(predictions, targets)
    print(f"MSE Loss: {result['total_loss']:.4f}")
    print(f"  Bus Loss: {result['bus_loss']:.4f}")
    print(f"  Generator Loss: {result['generator_loss']:.4f}")

    # RMSE Loss
    rmse_loss = create_rmse_loss()
    result = rmse_loss(predictions, targets)
    print(f"RMSE Loss: {result['total_loss']:.4f}")

    # Combined Loss
    combined_loss = create_combined_loss({'mse': 0.6, 'mape': 0.4})
    result = combined_loss(predictions, targets)
    print(f"Combined Loss: {result['total_loss']:.4f}")
    if 'loss_components' in result:
        for component, value in result['loss_components'].items():
            print(f"  {component}: {value:.4f}")

    # Weighted node importance
    weighted_loss = ACOPFLossFunction(
        loss_type='mse',
        node_weights={'bus': 2.0, 'generator': 1.0}  # Bus losses are more important
    )
    result = weighted_loss(predictions, targets)
    print(f"Weighted Loss (Bus=2x): {result['total_loss']:.4f}")

    print("\nLoss function classes created successfully!")
