import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Union, Optional
import numpy as np

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