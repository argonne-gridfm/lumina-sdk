"""
Flexible Loss Function Classes for ACOPF Training

This module provides a comprehensive set of loss functions that can be used
for training neural networks on ACOPF problems. Supports various loss types
including MSE, RMSE, MAPE, SmoothL1Loss, and combinations thereof.

Copyright (c) 2025, Argonne National Laboratory
All rights reserved.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import math
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CaseYCacheEntry:
    case_id: int
    n_bus: int
    y_real_sparse: torch.Tensor
    y_imag_sparse: torch.Tensor
    y_diag_real: torch.Tensor
    y_diag_imag: torch.Tensor
    line_y_real: Optional[torch.Tensor]
    line_y_imag: Optional[torch.Tensor]
    line_limits: Optional[torch.Tensor]
    base_mva: float
    device: torch.device


class CaseYCache:
    def __init__(self):
        self._cache: Dict[int, CaseYCacheEntry] = {}

    def get(self, case_id: int) -> Optional[CaseYCacheEntry]:
        return self._cache.get(int(case_id))

    def set(self, entry: CaseYCacheEntry) -> None:
        self._cache[int(entry.case_id)] = entry


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
        total_loss = None
        node_losses = {}
        loss_components = {}

        # Compute loss for each node type
        for node_type in predictions.keys():
            if node_type not in targets:
                continue

            pred = predictions[node_type]
            target = targets[node_type]

            if torch.is_tensor(target):
                finite_mask = torch.isfinite(target)
                if finite_mask.ndim > 1:
                    finite_mask = finite_mask.all(dim=-1)
                if finite_mask.ndim == 0:
                    if not bool(finite_mask.item()):
                        continue
                else:
                    if not finite_mask.any():
                        continue
                    pred = pred[finite_mask]
                    target = target[finite_mask]

            # Compute loss using PyTorch built-in functions
            node_loss = self._compute_single_loss(pred, target)

            # Weight by node importance
            node_weight = self.node_weights.get(node_type, 1.0)

            node_losses[f"{node_type}_loss"] = node_loss
            weighted_loss = node_weight * node_loss
            total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss

        # Prepare results
        if total_loss is None:
            device = next(iter(predictions.values())).device
            total_loss = torch.tensor(0.0, device=device)

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
        targets = self._extract_targets(predictions, batch)
        if self._is_homo_batch(batch) and self.loss_type in ['augmented_lagrangian', 'violated_lagrangian']:
            raise ValueError(
                "Augmented/violated Lagrangian losses require heterogeneous batches. "
                "Disable HomoOPFDataset or use a hetero model."
            )
        masked_predictions = predictions
        masked_targets = targets
        if self._is_homo_batch(batch) and hasattr(batch, "y_mask"):
            masked_predictions, masked_targets = self._apply_homo_target_mask(
                predictions,
                targets,
                batch,
            )

        # Call the forward method
        return self.forward(predictions, targets)

    @staticmethod
    def _is_homo_batch(batch) -> bool:
        return hasattr(batch, 'node_type') and not hasattr(batch, 'node_types')

    def _extract_targets(self, predictions: Dict[str, torch.Tensor], batch) -> Dict[str, torch.Tensor]:
        if self._is_homo_batch(batch):
            targets = {}
            y = getattr(batch, 'y', None)
            node_type = getattr(batch, 'node_type', None)
            if y is None or node_type is None:
                return targets
            node_types = ["bus", "generator", "load", "shunt"]
            for idx, name in enumerate(node_types):
                if name not in predictions:
                    continue
                mask = node_type == idx
                if mask.any():
                    targets[name] = y[mask]
            return targets

        targets = {}
        for node_type in predictions.keys():
            if hasattr(batch[node_type], 'y') and batch[node_type].y is not None:
                targets[node_type] = batch[node_type].y
        return targets

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
        log_normalized_violation: bool = False,
        **kwargs
    ):
        super().__init__()

        self.loss_type = loss_type
        self.device = device or torch.device('cpu')
        self.case_y_cache = CaseYCache()
        lag_config = lagrangian_config or {}
        self._last_lagrangian_loss = None
        self._iters_since_lagrangian_update = 0
        self.constraint_monitor = None
        self.log_normalized_violation = bool(log_normalized_violation)
        self.constraint_timing_enabled = bool(lag_config.get("constraint_timing", False))
        self.constraint_eval_stats = {"count": 0, "total_ms": 0.0, "last_ms": 0.0}

        # Initialize the appropriate loss function
        if loss_type == 'augmented_lagrangian':
            from .augmented_lagrangian import AugmentedLagrangianACOPF

            lagrangian_kwargs = dict(lag_config)
            self.lagrangian = AugmentedLagrangianACOPF(**lagrangian_kwargs)
            base_loss_type = lag_config.get('base_loss_type', 'mse')
            self.base_loss = ACOPFLossFunction(loss_type=base_loss_type, **kwargs)

        elif loss_type == 'violated_lagrangian':
            from .violated_lagrangian import ViolatedLagrangianACOPF

            lagrangian_kwargs = dict(lag_config)
            self.lagrangian = ViolatedLagrangianACOPF(**lagrangian_kwargs)
            base_loss_type = lag_config.get('base_loss_type', 'mse')
            self.base_loss = ACOPFLossFunction(loss_type=base_loss_type, **kwargs)

        else:
            # Standard ML loss
            self.base_loss = ACOPFLossFunction(loss_type=loss_type, **kwargs)
            self.lagrangian = None
            from .augmented_lagrangian import AugmentedLagrangianACOPF

            monitor_kwargs = dict(lag_config)
            monitor_kwargs.setdefault("verbose", False)
            self.constraint_monitor = AugmentedLagrangianACOPF(**monitor_kwargs)

        # Track whether Lagrangian network parameters have been initialized
        self._lagrangian_initialized = False

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        batch,
        return_info: bool = True,
        constraint_data: Optional[Dict] = None,
        collect_constraints: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Compute loss based on the configured loss type.

        Args:
            predictions: Model predictions
            batch: Batch data containing targets and inputs
            return_info: Whether to return additional loss information
            collect_constraints: Whether to compute constraint metrics (standard losses only)

        Returns:
            If return_info=False: loss tensor
            If return_info=True: (loss tensor, info dict)
        """
        # Extract targets from batch
        targets = self._extract_targets(predictions, batch)
        if self._is_homo_batch(batch) and self.loss_type in ['augmented_lagrangian', 'violated_lagrangian']:
            raise ValueError(
                "Augmented/violated Lagrangian losses require heterogeneous batches. "
                "Disable HomoOPFDataset or use a hetero model."
            )

        masked_predictions = predictions
        masked_targets = targets
        if self._is_homo_batch(batch) and hasattr(batch, "y_mask"):
            masked_predictions, masked_targets = self._apply_homo_target_mask(
                predictions,
                targets,
                batch,
            )

        if self.loss_type in ['augmented_lagrangian', 'violated_lagrangian']:
            # Compute base MSE loss
            base_results = self.base_loss(predictions, targets)
            mse_loss = base_results['total_loss']

            # Compute Lagrangian loss using shared constraint pipeline
            current_n_bus = batch['bus'].x.size(0)
            stored_ybus = getattr(self.lagrangian, 'Y_real_sparse', None)
            need_init = (
                not self._lagrangian_initialized
                or stored_ybus is None
                or stored_ybus.size(0) != current_n_bus
            )
            if need_init:
                self._ensure_network_parameters(batch, predictions['bus'].device, target=self.lagrangian)
            constraint_batch = constraint_data or self._create_constraint_batch(batch, predictions)
            lag_loss, info = self.lagrangian(mse_loss, predictions, constraint_batch)

            if return_info:
                if self.log_normalized_violation:
                    self._add_normalized_violation_metrics(info)
                info.update(base_results)
                info['objective'] = mse_loss.item()
                return lag_loss, info
            else:
                return lag_loss
        else:
            # Standard ML loss
            results = self.base_loss(masked_predictions, masked_targets)
            loss = results['total_loss']
            results.setdefault('objective', loss)

            if return_info:
                if collect_constraints:
                    results.update(self._collect_constraint_metrics(predictions, batch, constraint_data))
                return loss, results
            else:
                return loss

    def _collect_constraint_metrics(self, predictions, batch, constraint_data):
        monitor = self.constraint_monitor
        if monitor is None:
            return {}
        if self._is_homo_batch(batch):
            return self._collect_constraint_metrics_homo(predictions, batch, constraint_data)
        if 'bus' not in predictions or 'generator' not in predictions:
            return {}

        timing_start = self._start_constraint_timing()
        try:
            self._ensure_network_parameters(batch, predictions['bus'].device, target=monitor)
        except Exception:
            self._stop_constraint_timing(timing_start)
            return {}

        if (
            getattr(monitor, 'Y_real_sparse', None) is None
            or getattr(monitor, 'Y_imag_sparse', None) is None
        ):
            self._stop_constraint_timing(timing_start)
            return {}

        try:
            constraint_batch = constraint_data or self._create_constraint_batch(batch, predictions)
            constraints = monitor.compute_constraints(predictions, constraint_batch)
        except Exception:
            self._stop_constraint_timing(timing_start)
            return {}

        if constraints is None or constraints.numel() == 0:
            raw_violation = 0.0
            ema_violation = 0.0
        else:
            if torch.is_grad_enabled():
                monitor._constraint_signal_for_loss(constraints)
                raw_violation = monitor._raw_violation
                ema_violation = monitor._ema_violation
                if raw_violation is None:
                    raw_violation = monitor._compute_violation_norm(constraints)
                if ema_violation is None:
                    ema_violation = raw_violation
            else:
                raw_violation = monitor._compute_violation_norm(constraints)
                ema_violation = raw_violation

        constraint_violation = ema_violation if ema_violation is not None else raw_violation
        info = {
            'constraint_violation': constraint_violation,
            'raw_constraint_violation': raw_violation,
            'ema_constraint_violation': ema_violation,
            'p_balance_rmse': monitor._last_p_balance_rms,
            'q_balance_rmse': monitor._last_q_balance_rms,
            'line_limit_rmse': monitor._last_line_limit_rms,
        }
        if self.log_normalized_violation and constraints is not None:
            info["n_constraints"] = int(constraints.numel())
            self._add_normalized_violation_metrics(info)
        self._stop_constraint_timing(timing_start, info)
        return info

    def _collect_constraint_metrics_homo(self, predictions, batch, constraint_data):
        monitor = self.constraint_monitor
        if monitor is None:
            return {}
        if 'bus' not in predictions or 'generator' not in predictions:
            return {}

        timing_start = self._start_constraint_timing()
        try:
            self._ensure_network_parameters(batch, predictions['bus'].device, target=monitor)
        except Exception:
            self._stop_constraint_timing(timing_start)
            return {}

        if (
            getattr(monitor, 'Y_real_sparse', None) is None
            or getattr(monitor, 'Y_imag_sparse', None) is None
        ):
            self._stop_constraint_timing(timing_start)
            return {}

        try:
            constraint_batch = constraint_data or self._create_constraint_batch_homo(batch, predictions)
            if constraint_batch is None:
                self._stop_constraint_timing(timing_start)
                return {}
            constraints = monitor.compute_constraints(predictions, constraint_batch)
        except Exception:
            self._stop_constraint_timing(timing_start)
            return {}

        if constraints is None or constraints.numel() == 0:
            raw_violation = 0.0
            ema_violation = 0.0
        else:
            if torch.is_grad_enabled():
                monitor._constraint_signal_for_loss(constraints)
                raw_violation = monitor._raw_violation
                ema_violation = monitor._ema_violation
                if raw_violation is None:
                    raw_violation = monitor._compute_violation_norm(constraints)
                if ema_violation is None:
                    ema_violation = raw_violation
            else:
                raw_violation = monitor._compute_violation_norm(constraints)
                ema_violation = raw_violation

        constraint_violation = ema_violation if ema_violation is not None else raw_violation
        info = {
            'constraint_violation': constraint_violation,
            'raw_constraint_violation': raw_violation,
            'ema_constraint_violation': ema_violation,
            'p_balance_rmse': monitor._last_p_balance_rms,
            'q_balance_rmse': monitor._last_q_balance_rms,
            'line_limit_rmse': monitor._last_line_limit_rms,
        }
        if self.log_normalized_violation and constraints is not None:
            info["n_constraints"] = int(constraints.numel())
            self._add_normalized_violation_metrics(info)
        self._stop_constraint_timing(timing_start, info)
        return info

    def _start_constraint_timing(self):
        if not self.constraint_timing_enabled:
            return None
        return time.perf_counter()

    def _stop_constraint_timing(self, start_time, info=None):
        if start_time is None:
            return
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        stats = self.constraint_eval_stats
        stats["count"] += 1
        stats["total_ms"] += elapsed_ms
        stats["last_ms"] = elapsed_ms
        if info is not None:
            info["constraint_eval_ms"] = elapsed_ms
            info["constraint_eval_avg_ms"] = stats["total_ms"] / stats["count"]

    @staticmethod
    def _is_homo_batch(batch) -> bool:
        return hasattr(batch, 'node_type') and not hasattr(batch, 'node_types')

    def _extract_targets(self, predictions: Dict[str, torch.Tensor], batch) -> Dict[str, torch.Tensor]:
        if self._is_homo_batch(batch):
            targets = {}
            y = getattr(batch, 'y', None)
            node_type = getattr(batch, 'node_type', None)
            if y is None or node_type is None:
                return targets
            node_types = ["bus", "generator", "load", "shunt"]
            for idx, name in enumerate(node_types):
                if name not in predictions:
                    continue
                mask = node_type == idx
                if mask.any():
                    targets[name] = y[mask]
            return targets

        targets = {}
        for node_type in predictions.keys():
            if hasattr(batch[node_type], 'y') and batch[node_type].y is not None:
                targets[node_type] = batch[node_type].y
        return targets

    def _add_normalized_violation_metrics(self, info):
        if not info:
            return
        n_constraints = info.get("n_constraints")
        if n_constraints is None:
            constraints = info.get("constraints")
            if torch.is_tensor(constraints):
                n_constraints = int(constraints.numel())
        try:
            n_constraints = int(n_constraints)
        except (TypeError, ValueError):
            n_constraints = 0
        if n_constraints <= 0:
            return
        scale = math.sqrt(n_constraints)

        def _norm_value(value):
            if value is None:
                return None
            if torch.is_tensor(value):
                return value / scale
            try:
                return float(value) / scale
            except (TypeError, ValueError):
                return None

        raw = info.get("raw_constraint_violation")
        ema = info.get("ema_constraint_violation")
        if raw is not None:
            info["raw_constraint_violation_norm"] = _norm_value(raw)
        if ema is not None:
            info["ema_constraint_violation_norm"] = _norm_value(ema)

    def _apply_homo_target_mask(self, predictions, targets, batch):
        y_mask = getattr(batch, "y_mask", None)
        node_type = getattr(batch, "node_type", None)
        if y_mask is None or node_type is None or not torch.is_tensor(y_mask):
            return predictions, targets

        if y_mask.ndim > 1:
            y_mask = y_mask.all(dim=-1)
        if y_mask.numel() != node_type.numel():
            return predictions, targets

        node_types = ["bus", "generator", "load", "shunt"]
        masked_predictions = {}
        masked_targets = {}
        for idx, name in enumerate(node_types):
            if name not in predictions or name not in targets:
                continue
            type_mask = node_type == idx
            if not bool(type_mask.any().item()):
                continue
            valid_mask = y_mask[type_mask]
            if not bool(valid_mask.any().item()):
                continue
            masked_predictions[name] = predictions[name][valid_mask]
            masked_targets[name] = targets[name][valid_mask]

        return masked_predictions, masked_targets

    def update_lagrangian(
        self,
        constraint_violation: Optional[float] = None,
        constraints: Optional[torch.Tensor] = None,
        update_penalty: bool = True,
        is_training: bool = True,
        force: bool = False,
        sample_count: Optional[int] = None,
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
            sample_count: Number of samples processed since last update (for sample-based scheduling)
        """
        if self.lagrangian is None or not is_training:
            return

        if sample_count is not None and hasattr(self.lagrangian, "step_samples"):
            self.lagrangian.step_samples(sample_count)

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

    def _get_homo_type_names(self, batch):
        node_type_names = getattr(batch, "node_type_names", None)
        edge_type_names = getattr(batch, "edge_type_names", None)
        if not node_type_names:
            node_type_names = getattr(batch, "_node_type_names", None)
        if not edge_type_names:
            edge_type_names = getattr(batch, "_edge_type_names", None)

        if isinstance(node_type_names, (list, tuple)) and node_type_names:
            if isinstance(node_type_names[0], (list, tuple)):
                node_type_names = node_type_names[0]
        if isinstance(edge_type_names, (list, tuple)) and edge_type_names:
            if isinstance(edge_type_names[0], (list, tuple)):
                edge_type_names = edge_type_names[0]

        node_type_names = list(node_type_names) if node_type_names else None
        edge_type_names = list(edge_type_names) if edge_type_names else None
        return node_type_names, edge_type_names

    def _homo_type_id(self, node_type_names, name, fallback):
        if node_type_names and name in node_type_names:
            return int(node_type_names.index(name))
        return int(fallback)

    def _edge_type_matches(self, edge_name, src, rel, dst):
        if isinstance(edge_name, (list, tuple)) and len(edge_name) == 3:
            return edge_name[0] == src and edge_name[1] == rel and edge_name[2] == dst
        if isinstance(edge_name, str):
            parts = edge_name.split("::")
            if len(parts) == 3:
                return parts[0] == src and parts[1] == rel and parts[2] == dst
        return False

    def _homo_edge_type_id(self, edge_type_names, src, rel, dst):
        if not edge_type_names:
            return None
        for idx, edge_name in enumerate(edge_type_names):
            if self._edge_type_matches(edge_name, src, rel, dst):
                return int(idx)
        return None

    def _create_constraint_batch_homo(self, batch, predictions):
        device = predictions['bus'].device
        node_type = getattr(batch, 'node_type', None)
        if node_type is None:
            return None
        node_type = node_type.to(device)

        node_type_names, edge_type_names = self._get_homo_type_names(batch)
        if not node_type_names:
            node_type_names = ["bus", "generator", "load", "shunt"]

        bus_type_id = self._homo_type_id(node_type_names, "bus", 0)
        gen_type_id = self._homo_type_id(node_type_names, "generator", 1)
        load_type_id = self._homo_type_id(node_type_names, "load", 2)

        node_batch = getattr(batch, "batch", None)
        if node_batch is not None:
            node_batch = node_batch.to(device)
            graph0_mask = node_batch == 0
            num_graphs = int(node_batch.max().item()) + 1
        else:
            graph0_mask = None
            num_graphs = 1

        bus_mask = node_type == bus_type_id
        gen_mask = node_type == gen_type_id
        load_mask = node_type == load_type_id

        bus_mask_g0 = bus_mask if graph0_mask is None else bus_mask & graph0_mask
        gen_mask_g0 = gen_mask if graph0_mask is None else gen_mask & graph0_mask
        load_mask_g0 = load_mask if graph0_mask is None else load_mask & graph0_mask

        bus_nodes_g0 = bus_mask_g0.nonzero(as_tuple=False).view(-1)
        gen_nodes_g0 = gen_mask_g0.nonzero(as_tuple=False).view(-1)
        load_nodes_g0 = load_mask_g0.nonzero(as_tuple=False).view(-1)

        n_bus = int(bus_nodes_g0.numel())
        if n_bus == 0:
            return None
        n_gen = int(gen_nodes_g0.numel())
        n_load = int(load_nodes_g0.numel())

        num_nodes = batch.x.size(0)
        bus_index_map = torch.full((num_nodes,), -1, device=device, dtype=torch.long)
        bus_index_map[bus_nodes_g0] = torch.arange(n_bus, device=device)
        gen_index_map = torch.full((num_nodes,), -1, device=device, dtype=torch.long)
        gen_index_map[gen_nodes_g0] = torch.arange(n_gen, device=device)
        load_index_map = torch.full((num_nodes,), -1, device=device, dtype=torch.long)
        load_index_map[load_nodes_g0] = torch.arange(n_load, device=device)

        base_mva = getattr(batch, 'baseMVA', None)
        if base_mva is None and hasattr(batch, 'base_mva'):
            base_mva = getattr(batch, 'base_mva')
        if torch.is_tensor(base_mva):
            base_mva = base_mva.view(-1)[0].item()
        base_mva = float(base_mva) if base_mva is not None else 100.0

        load_pd = None
        load_qd = None
        if load_mask.any():
            load_x = batch.x[load_mask].to(device)
            pd_flat = load_x[:, 0] if load_x.size(1) > 0 else torch.zeros(load_x.size(0), device=device)
            if load_x.size(1) > 1:
                qd_flat = load_x[:, 1]
            else:
                qd_flat = torch.zeros_like(pd_flat)

            if n_load > 0 and pd_flat.numel() == num_graphs * n_load:
                load_pd = pd_flat.view(num_graphs, n_load)
                load_qd = qd_flat.view(num_graphs, n_load)
            elif num_graphs == 1:
                load_pd = pd_flat.view(1, -1)
                load_qd = qd_flat.view(1, -1)
            else:
                return None

        edge_index = getattr(batch, 'edge_index', None)
        edge_type = getattr(batch, 'edge_type', None)
        edge_attr = getattr(batch, 'edge_attr_full', None)
        if edge_attr is None:
            edge_attr = getattr(batch, 'edge_attr', None)
        if edge_index is None:
            return None
        edge_index = edge_index.to(device)
        if edge_type is not None:
            edge_type = edge_type.to(device)
        if edge_attr is not None:
            edge_attr = edge_attr.to(device)

        if node_batch is not None:
            edge_graph_mask = (node_batch[edge_index[0]] == 0) & (node_batch[edge_index[1]] == 0)
        else:
            edge_graph_mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=device)
        src_type = node_type[edge_index[0]]
        dst_type = node_type[edge_index[1]]

        gen_bus_indices = torch.zeros(n_gen, device=device, dtype=torch.long)
        gen_edge_id = self._homo_edge_type_id(edge_type_names, "bus", "generator_link", "generator")
        if edge_type is not None and gen_edge_id is not None:
            gen_edge_mask = (edge_type == gen_edge_id) & edge_graph_mask
        else:
            gen_edge_mask = (src_type == bus_type_id) & (dst_type == gen_type_id) & edge_graph_mask
        gen_edges = edge_index[:, gen_edge_mask]
        if gen_edges.numel() > 0:
            bus_idx = bus_index_map[gen_edges[0]]
            gen_idx = gen_index_map[gen_edges[1]]
            valid = (bus_idx >= 0) & (gen_idx >= 0)
            gen_bus_indices[gen_idx[valid]] = bus_idx[valid]

        load_bus_indices = None
        load_edge_id = self._homo_edge_type_id(edge_type_names, "bus", "load_link", "load")
        if n_load > 0:
            if edge_type is not None and load_edge_id is not None:
                load_edge_mask = (edge_type == load_edge_id) & edge_graph_mask
            else:
                load_edge_mask = (src_type == bus_type_id) & (dst_type == load_type_id) & edge_graph_mask
            load_edges = edge_index[:, load_edge_mask]
            if load_edges.numel() > 0:
                bus_idx = bus_index_map[load_edges[0]]
                load_idx = load_index_map[load_edges[1]]
                valid = (bus_idx >= 0) & (load_idx >= 0)
                load_bus_indices = torch.zeros(n_load, device=device, dtype=torch.long)
                load_bus_indices[load_idx[valid]] = bus_idx[valid]

        line_edge_index = None
        line_limits = None
        line_edge_id = self._homo_edge_type_id(edge_type_names, "bus", "ac_line", "bus")
        if edge_type is not None and line_edge_id is not None:
            line_edge_mask = (edge_type == line_edge_id) & edge_graph_mask
        else:
            line_edge_mask = (src_type == bus_type_id) & (dst_type == bus_type_id) & edge_graph_mask
        line_edges = edge_index[:, line_edge_mask]
        if line_edges.numel() > 0:
            bus_i = bus_index_map[line_edges[0]]
            bus_j = bus_index_map[line_edges[1]]
            valid = (bus_i >= 0) & (bus_j >= 0)
            if bool(valid.any().item()):
                line_edge_index = torch.stack([bus_i[valid], bus_j[valid]], dim=0)
                if edge_attr is not None:
                    line_attr = edge_attr[line_edge_mask][valid]
                    if line_attr.size(1) >= 7:
                        line_limits = line_attr[:, 6].abs()

        class ConstraintBatch:
            def __init__(self):
                self.baseMVA = base_mva
                self.n_bus = n_bus

            def get(self, key, default=None):
                if key == 'pd':
                    return load_pd
                if key == 'qd':
                    return load_qd
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
            return None, None, None, None, None, None

        edge_index = batch['bus', 'ac_line', 'bus'].edge_index.to(device)
        edge_attr = batch['bus', 'ac_line', 'bus'].edge_attr
        if edge_attr is None or edge_attr.size(0) != edge_index.size(1) or edge_attr.size(1) < 6:
            return None, None, None, None, None, None

        edge_attr = edge_attr.to(device)
        r = edge_attr[:, 4]
        x = edge_attr[:, 5]
        denom = (r ** 2 + x ** 2).clamp_min(1e-6)
        y_real = r / denom
        y_imag = -x / denom
        line_y_real = -y_real
        line_y_imag = -y_imag

        n_bus = batch['bus'].x.size(0)

        i = edge_index[0].clamp(max=n_bus - 1)
        j = edge_index[1].clamp(max=n_bus - 1)

        diag_real = torch.zeros(n_bus, device=device)
        diag_imag = torch.zeros(n_bus, device=device)
        diag_real.index_add_(0, i, y_real)
        diag_real.index_add_(0, j, y_real)
        diag_imag.index_add_(0, i, y_imag)
        diag_imag.index_add_(0, j, y_imag)

        Y_real_sparse, Y_imag_sparse = self._build_sparse_ybus(i, j, y_real, y_imag, n_bus, device)
        return Y_real_sparse, Y_imag_sparse, diag_real, diag_imag, line_y_real, line_y_imag

    def _build_admittance_from_homo(self, batch, device):
        node_type = getattr(batch, 'node_type', None)
        edge_type = getattr(batch, 'edge_type', None)
        edge_index = getattr(batch, 'edge_index', None)
        edge_attr = getattr(batch, 'edge_attr_full', None)
        if edge_attr is None:
            edge_attr = getattr(batch, 'edge_attr', None)
        if node_type is None or edge_index is None or edge_attr is None:
            return None, None, None, None, None, None, None

        node_type = node_type.to(device)
        if edge_type is not None:
            edge_type = edge_type.to(device)
        edge_index = edge_index.to(device)

        node_type_names, edge_type_names = self._get_homo_type_names(batch)
        if not node_type_names:
            node_type_names = ["bus", "generator", "load", "shunt"]

        bus_type_id = self._homo_type_id(node_type_names, "bus", 0)
        bus_mask = node_type == bus_type_id

        node_batch = getattr(batch, "batch", None)
        if node_batch is not None:
            node_batch = node_batch.to(device)
            bus_mask = bus_mask & (node_batch == 0)

        bus_nodes = bus_mask.nonzero(as_tuple=False).view(-1)
        n_bus = int(bus_nodes.numel())
        if n_bus == 0:
            return None, None, None, None, None, None, None

        bus_index_map = torch.full((batch.x.size(0),), -1, device=device, dtype=torch.long)
        bus_index_map[bus_nodes] = torch.arange(n_bus, device=device)

        line_edge_id = self._homo_edge_type_id(edge_type_names, "bus", "ac_line", "bus")

        if node_batch is not None:
            edge_graph_mask = (node_batch[edge_index[0]] == 0) & (node_batch[edge_index[1]] == 0)
        else:
            edge_graph_mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=device)

        if edge_type is not None and line_edge_id is not None:
            line_mask = (edge_type == line_edge_id) & edge_graph_mask
        else:
            src_type = node_type[edge_index[0]]
            dst_type = node_type[edge_index[1]]
            line_mask = (src_type == bus_type_id) & (dst_type == bus_type_id) & edge_graph_mask
        if not bool(line_mask.any().item()):
            return None, None, None, None, None, None, None

        if edge_attr.size(0) != edge_index.size(1) or edge_attr.size(1) < 6:
            return None, None, None, None, None, None, None

        edge_attr = edge_attr.to(device)
        line_attr = edge_attr[line_mask]
        line_edges = edge_index[:, line_mask]

        bus_i = bus_index_map[line_edges[0]]
        bus_j = bus_index_map[line_edges[1]]
        valid = (bus_i >= 0) & (bus_j >= 0)
        if not bool(valid.any().item()):
            return None, None, None, None, None, None, None

        r = line_attr[:, 4]
        x = line_attr[:, 5]
        r = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        denom = (r ** 2 + x ** 2).clamp_min(1e-6)
        y_real = r / denom
        y_imag = -x / denom
        line_y_real = -y_real
        line_y_imag = -y_imag

        bus_i = bus_i[valid]
        bus_j = bus_j[valid]
        y_real = y_real[valid]
        y_imag = y_imag[valid]
        line_y_real = line_y_real[valid]
        line_y_imag = line_y_imag[valid]

        diag_real = torch.zeros(n_bus, device=device)
        diag_imag = torch.zeros(n_bus, device=device)
        diag_real.index_add_(0, bus_i, y_real)
        diag_real.index_add_(0, bus_j, y_real)
        diag_imag.index_add_(0, bus_i, y_imag)
        diag_imag.index_add_(0, bus_j, y_imag)

        line_limits = None
        if line_attr.size(1) >= 7:
            line_limits = line_attr[:, 6].abs()[valid]

        Y_real_sparse, Y_imag_sparse = self._build_sparse_ybus(bus_i, bus_j, y_real, y_imag, n_bus, device)
        return (
            Y_real_sparse,
            Y_imag_sparse,
            diag_real,
            diag_imag,
            line_y_real,
            line_y_imag,
            line_limits,
        )

    def _build_sparse_ybus(self, bus_i, bus_j, y_real, y_imag, n_bus, device):
        if bus_i is None or bus_j is None or y_real is None or y_imag is None:
            return None, None
        if bus_i.numel() == 0 or n_bus <= 0:
            empty_idx = torch.empty((2, 0), dtype=torch.long, device=device)
            empty_val = torch.empty((0,), device=device)
            y_real_sparse = torch.sparse_coo_tensor(empty_idx, empty_val, (n_bus, n_bus))
            y_imag_sparse = torch.sparse_coo_tensor(empty_idx, empty_val, (n_bus, n_bus))
            return y_real_sparse, y_imag_sparse

        diag_real = torch.zeros(n_bus, device=device)
        diag_imag = torch.zeros(n_bus, device=device)
        diag_real.index_add_(0, bus_i, y_real)
        diag_real.index_add_(0, bus_j, y_real)
        diag_imag.index_add_(0, bus_i, y_imag)
        diag_imag.index_add_(0, bus_j, y_imag)

        diag_idx = torch.arange(n_bus, device=device)
        row = torch.cat([bus_i, bus_j, diag_idx], dim=0)
        col = torch.cat([bus_j, bus_i, diag_idx], dim=0)
        values_real = torch.cat([-y_real, -y_real, diag_real], dim=0)
        values_imag = torch.cat([-y_imag, -y_imag, diag_imag], dim=0)
        indices = torch.stack([row, col], dim=0)
        y_real_sparse = torch.sparse_coo_tensor(indices, values_real, (n_bus, n_bus)).coalesce()
        y_imag_sparse = torch.sparse_coo_tensor(indices, values_imag, (n_bus, n_bus)).coalesce()
        return y_real_sparse, y_imag_sparse

    def _resolve_case_id(self, batch):
        case_id = getattr(batch, "case_id", None)
        if case_id is None:
            return None
        if torch.is_tensor(case_id):
            if case_id.numel() == 0:
                return None
            if case_id.dim() > 0:
                unique = torch.unique(case_id)
                if unique.numel() != 1:
                    return None
                case_id = unique[0]
            return int(case_id.item())
        try:
            return int(case_id)
        except (TypeError, ValueError):
            return None

    def _maybe_apply_cached_y(self, case_id, n_bus, device, target):
        if case_id is None:
            return False
        entry = self.case_y_cache.get(case_id)
        if entry is None:
            return False
        if entry.n_bus != n_bus:
            return False
        if entry.device != device:
            return False
        target.set_network_parameters(
            Y_real_sparse=entry.y_real_sparse,
            Y_imag_sparse=entry.y_imag_sparse,
            Y_diag_real=entry.y_diag_real,
            Y_diag_imag=entry.y_diag_imag,
            line_y_real=entry.line_y_real,
            line_y_imag=entry.line_y_imag,
            line_limits=entry.line_limits,
            base_mva=entry.base_mva,
        )
        if target is self.lagrangian:
            self._lagrangian_initialized = True
            self._lagrangian_bus_count = n_bus
        return True

    def _cache_case_y(
        self,
        case_id,
        n_bus,
        device,
        y_real_sparse,
        y_imag_sparse,
        y_diag_real,
        y_diag_imag,
        line_y_real,
        line_y_imag,
        line_limits,
        base_mva,
    ):
        if case_id is None:
            return
        entry = CaseYCacheEntry(
            case_id=case_id,
            n_bus=n_bus,
            y_real_sparse=y_real_sparse,
            y_imag_sparse=y_imag_sparse,
            y_diag_real=y_diag_real,
            y_diag_imag=y_diag_imag,
            line_y_real=line_y_real,
            line_y_imag=line_y_imag,
            line_limits=line_limits,
            base_mva=base_mva,
            device=device,
        )
        self.case_y_cache.set(entry)

    def _ensure_network_parameters(self, batch, device, target=None):
        """Initialize AugLag network parameters from batch data when needed."""
        target = target or self.lagrangian
        if target is None:
            return

        if self._is_homo_batch(batch):
            node_type = getattr(batch, 'node_type', None)
            if node_type is None:
                return
            node_type_names, _ = self._get_homo_type_names(batch)
            if not node_type_names:
                node_type_names = ["bus", "generator", "load", "shunt"]
            bus_type_id = self._homo_type_id(node_type_names, "bus", 0)
            bus_mask = node_type == bus_type_id
            node_batch = getattr(batch, "batch", None)
            if node_batch is not None:
                bus_mask = bus_mask & (node_batch == 0)
            n_bus = int(bus_mask.sum().item())
            if n_bus <= 0:
                return
            case_id = self._resolve_case_id(batch)
            stored_sparse = getattr(target, 'Y_real_sparse', None)
            need_init = stored_sparse is None or stored_sparse.size(0) != n_bus

            if not need_init:
                return

            if self._maybe_apply_cached_y(case_id, n_bus, device, target):
                return

            (
                Y_real_sparse,
                Y_imag_sparse,
                diag_real,
                diag_imag,
                line_y_real,
                line_y_imag,
                line_limits,
            ) = self._build_admittance_from_homo(batch, device)
            if Y_real_sparse is None or Y_imag_sparse is None:
                return

            base_mva = getattr(batch, 'baseMVA', None)
            if base_mva is None and hasattr(batch, 'base_mva'):
                base_mva = getattr(batch, 'base_mva')
            if torch.is_tensor(base_mva):
                base_mva = base_mva.view(-1)[0].item()
            base_mva = float(base_mva) if base_mva is not None else 100.0

            target.set_network_parameters(
                Y_real_sparse=Y_real_sparse,
                Y_imag_sparse=Y_imag_sparse,
                Y_diag_real=diag_real,
                Y_diag_imag=diag_imag,
                line_y_real=line_y_real,
                line_y_imag=line_y_imag,
                line_limits=line_limits,
                base_mva=base_mva,
            )

            if target is self.lagrangian:
                self._lagrangian_initialized = True
                self._lagrangian_bus_count = n_bus
            self._cache_case_y(
                case_id,
                n_bus,
                device,
                Y_real_sparse,
                Y_imag_sparse,
                diag_real,
                diag_imag,
                line_y_real,
                line_y_imag,
                line_limits,
                base_mva,
            )
            return

        n_bus = batch['bus'].x.size(0)
        case_id = self._resolve_case_id(batch)
        stored_sparse = getattr(target, 'Y_real_sparse', None)
        need_init = stored_sparse is None or stored_sparse.size(0) != n_bus

        if not need_init:
            return

        if self._maybe_apply_cached_y(case_id, n_bus, device, target):
            return

        (
            Y_real_sparse,
            Y_imag_sparse,
            diag_real,
            diag_imag,
            line_y_real,
            line_y_imag,
        ) = self._build_admittance_from_batch(batch, device)
        if Y_real_sparse is None or Y_imag_sparse is None:
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

        target.set_network_parameters(
            Y_real_sparse=Y_real_sparse,
            Y_imag_sparse=Y_imag_sparse,
            Y_diag_real=diag_real,
            Y_diag_imag=diag_imag,
            line_y_real=line_y_real,
            line_y_imag=line_y_imag,
            line_limits=line_limits,
            base_mva=base_mva,
        )

        # Avoid re-initializing network parameters on subsequent calls
        if target is self.lagrangian:
            self._lagrangian_initialized = True
            self._lagrangian_bus_count = n_bus
        self._cache_case_y(
            case_id,
            n_bus,
            device,
            Y_real_sparse,
            Y_imag_sparse,
            diag_real,
            diag_imag,
            line_y_real,
            line_y_imag,
            line_limits,
            base_mva,
        )

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
