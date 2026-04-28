"""
Flexible Loss Function Classes for ACOPF Training

This module provides a comprehensive set of loss functions that can be used
for training neural networks on ACOPF problems. Supports various loss types
including MSE, RMSE, MAPE, SmoothL1Loss, and combinations thereof.

Copyright (c) 2025, Argonne National Laboratory
All rights reserved.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

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
        abs_target = torch.abs(targets) + self.epsilon
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

        self.node_weights = node_weights or {'bus': 1.0, 'generator': 1.0}

        valid_types = ['mse', 'rmse', 'mae', 'mape', 'smooth_l1']
        if loss_type not in valid_types:
            raise ValueError(f"Invalid loss_type '{loss_type}'. Must be one of {valid_types}")

        self._init_loss_functions()

    def _init_loss_functions(self):
        if self.loss_type == 'mse':
            self.criterion = nn.MSELoss(reduction=self.reduction)
        elif self.loss_type == 'mae':
            self.criterion = nn.L1Loss(reduction=self.reduction)
        elif self.loss_type == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss(reduction=self.reduction, beta=self.beta)
        elif self.loss_type == 'rmse':
            self.criterion = nn.MSELoss(reduction='none')
        elif self.loss_type == 'mape':
            self.criterion = nn.L1Loss(reduction='none')

    def _compute_single_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
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
            abs_error = torch.abs(predictions - targets)
            abs_target = torch.abs(targets) + self.epsilon
            mape = abs_error / abs_target
            return self._reduce_loss(mape)
        else:
            raise ValueError(f"Unknown loss function: {self.loss_type}")

    def _reduce_loss(self, loss: torch.Tensor) -> torch.Tensor:
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
        total_loss = None
        node_losses = {}
        loss_components = {}

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

            node_loss = self._compute_single_loss(pred, target)
            node_weight = self.node_weights.get(node_type, 1.0)

            node_losses[f"{node_type}_loss"] = node_loss
            weighted_loss = node_weight * node_loss
            total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss

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
        targets = self._extract_targets(predictions, batch)
        masked_predictions = predictions
        masked_targets = targets
        if self._is_homo_batch(batch) and hasattr(batch, "y_mask"):
            masked_predictions, masked_targets = self._apply_homo_target_mask(
                predictions,
                targets,
                batch,
            )
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

    Args:
        base_loss_config (dict): Configuration for base ML loss
        physics_weight (float): Weight for physics constraint penalty. Default: 1.0
        constraint_types (list): Types of constraints to include.
        penalty_method (str): How to penalize constraint violations.
    """

    def __init__(
        self,
        base_loss_config: Dict = None,
        physics_weight: float = 1.0,
        constraint_types: List[str] = None,
        penalty_method: str = 'quadratic',
        **kwargs
    ):
        base_config = base_loss_config or {'loss_type': 'mse'}
        super().__init__(**base_config, **kwargs)

        self.physics_weight = physics_weight
        self.constraint_types = constraint_types or ['power_flow', 'line_limits']
        self.penalty_method = penalty_method
        self.constraint_computer = None

    def set_constraint_computer(self, constraint_computer):
        self.constraint_computer = constraint_computer

    def _compute_physics_penalty(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.constraint_computer is None:
            return torch.tensor(0.0, device=next(iter(predictions.values())).device)

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
                    penalty = -torch.mean(torch.log(torch.clamp(-violation, min=self.epsilon)))
                else:
                    raise ValueError(f"Unknown penalty method: {self.penalty_method}")

                total_penalty += penalty

        return total_penalty

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        results = super().forward(predictions, targets)
        physics_penalty = self._compute_physics_penalty(predictions)
        total_loss = results['total_loss'] + self.physics_weight * physics_penalty
        results.update({
            'total_loss': total_loss,
            'ml_loss': results['total_loss'],
            'physics_penalty': physics_penalty
        })
        return results


class OPFLossManager(nn.Module):
    """
    Unified loss manager for standard ML loss types.

    Args:
        loss_type (str): Type of loss to use. Options:
            - 'mse', 'rmse', 'mae', 'mape', 'smooth_l1': Standard ML losses
        device (torch.device, optional): Device for computations
        **kwargs: Additional arguments passed to the loss function
    """

    def __init__(
        self,
        loss_type: str = 'mse',
        device: Optional[torch.device] = None,
        **kwargs
    ):
        super().__init__()

        self.loss_type = loss_type
        self.device = device or torch.device('cpu')
        self.base_loss = ACOPFLossFunction(loss_type=loss_type, **kwargs)

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        batch,
        return_info: bool = True,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        targets = self._extract_targets(predictions, batch)

        masked_predictions = predictions
        masked_targets = targets
        if self._is_homo_batch(batch) and hasattr(batch, "y_mask"):
            masked_predictions, masked_targets = self._apply_homo_target_mask(
                predictions,
                targets,
                batch,
            )

        results = self.base_loss(masked_predictions, masked_targets)
        loss = results['total_loss']
        results.setdefault('objective', loss)

        if return_info:
            return loss, results
        else:
            return loss

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

    def get_loss_info(self) -> Dict:
        info = {
            'loss_type': self.loss_type,
        }
        if hasattr(self.base_loss, 'get_loss_info'):
            info.update(self.base_loss.get_loss_info())
        return info


# Convenience functions for common loss configurations
def create_mse_loss(**kwargs) -> ACOPFLossFunction:
    return ACOPFLossFunction(loss_type='mse', **kwargs)


def create_rmse_loss(**kwargs) -> ACOPFLossFunction:
    return ACOPFLossFunction(loss_type='rmse', **kwargs)


def create_mape_loss(**kwargs) -> ACOPFLossFunction:
    return ACOPFLossFunction(loss_type='mape', **kwargs)


def create_combined_loss(weights: Dict[str, float], **kwargs) -> ACOPFLossFunction:
    return ACOPFLossFunction(loss_type=weights, **kwargs)


def create_robust_loss(**kwargs) -> ACOPFLossFunction:
    return ACOPFLossFunction(
        loss_type={'mse': 0.7, 'smooth_l1': 0.3},
        **kwargs
    )
