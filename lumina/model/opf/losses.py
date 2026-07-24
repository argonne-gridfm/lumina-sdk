"""
Flexible Loss Function Classes for ACOPF Training

This module provides a comprehensive set of loss functions that can be used
for training neural networks on ACOPF problems. Supports various loss types
including MSE, RMSE, MAPE, SmoothL1Loss, and combinations thereof.

Copyright (c) 2025, Argonne National Laboratory
All rights reserved.
"""

from typing import Dict, List, Optional, Tuple, Union

import math
import time

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
    """Physics-informed loss combining standard ML loss with physics constraint penalties.

    Extends ``ACOPFLossFunction`` by adding a penalty term computed from
    power system constraint violations (e.g. power flow, line limits).
    The total loss is ``ML_loss + physics_weight * physics_penalty``.

    Args:
        base_loss_config (dict, optional): Configuration dict forwarded to
            ``ACOPFLossFunction`` (must contain at least ``'loss_type'``).
            Defaults to ``{'loss_type': 'mse'}``.
        physics_weight (float): Scalar weight for the physics penalty term.
            Defaults to 1.0.
        constraint_types (list, optional): Which constraint families to
            penalize, e.g. ``['power_flow', 'line_limits']``.
        penalty_method (str): Penalty formulation. One of ``'quadratic'``,
            ``'absolute'``, or ``'log_barrier'``. Defaults to ``'quadratic'``.
        **kwargs: Additional keyword arguments forwarded to
            ``ACOPFLossFunction``.
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
        """Attach a constraint violation computer for physics penalty evaluation.

        Args:
            constraint_computer: Object with a ``compute_violations`` method
                that accepts predictions and returns a dict of violation
                tensors keyed by constraint type.
        """
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
    """Loss manager for existing SDK losses and heterogeneous ACOPF AL training."""

    def __init__(
        self,
        loss_type: str = "mse",
        device: Optional[torch.device] = None,
        lagrangian_config: Optional[Dict] = None,
        log_normalized_violation: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.device = device or torch.device("cpu")
        self.log_normalized_violation = bool(log_normalized_violation)
        config = dict(lagrangian_config or {})
        base_type = config.get("base_loss_type", "mse")
        self.base_loss = ACOPFLossFunction(
            loss_type=base_type if loss_type == "augmented_lagrangian" else loss_type,
            **kwargs,
        )

        from .augmented_lagrangian import AugmentedLagrangianACOPF

        self.lagrangian = (
            AugmentedLagrangianACOPF(**config)
            if loss_type == "augmented_lagrangian"
            else None
        )
        # The monitor supplies feasibility metrics during the initial MSE phase.
        self.constraint_monitor = (
            None if self.lagrangian is not None else AugmentedLagrangianACOPF(**config)
        )
        self._lagrangian_initialized = False
        self._pending_lagrangian_state = None
        self.constraint_timing_enabled = bool(config.get("constraint_timing", False))
        self.constraint_eval_stats = {"count": 0, "total_ms": 0.0, "last_ms": 0.0}

    def training_state_dict(self) -> Dict:
        state = {"loss_type": self.loss_type}
        if self.lagrangian is not None:
            state["lagrangian"] = self.lagrangian.training_state_dict()
        return state

    def load_training_state_dict(self, state: Dict):
        if isinstance(state, dict) and self.lagrangian is not None:
            self._pending_lagrangian_state = state.get("lagrangian")

    def _restore_pending_lagrangian_state(self):
        if self._pending_lagrangian_state is not None:
            self.lagrangian.load_training_state_dict(self._pending_lagrangian_state)
            self._pending_lagrangian_state = None

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        batch,
        return_info: bool = True,
        constraint_data: Optional[Dict] = None,
        collect_constraints: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        targets = self._extract_targets(predictions, batch)
        if self._is_homo_batch(batch):
            if self.lagrangian is not None:
                raise ValueError(
                    "Augmented Lagrangian training supports heterogeneous OPF batches only."
                )
            if hasattr(batch, "y_mask"):
                predictions, targets = self._apply_homo_target_mask(
                    predictions, targets, batch
                )

        base_results = self.base_loss(predictions, targets)
        objective = base_results["total_loss"]
        if self.lagrangian is None:
            base_results.setdefault("objective", objective)
            if collect_constraints and not self._is_homo_batch(batch):
                base_results.update(
                    self._collect_constraint_metrics(
                        predictions, batch, constraint_data
                    )
                )
            return (objective, base_results) if return_info else objective

        start = self._start_constraint_timing()
        self._ensure_network_parameters(batch, predictions["bus"].device)
        self._restore_pending_lagrangian_state()
        constraint_batch = constraint_data or self._create_constraint_batch(
            batch, predictions
        )
        loss, info = self.lagrangian(objective, predictions, constraint_batch)
        info.update(base_results)
        info["objective"] = objective
        if self.log_normalized_violation:
            self._add_normalized_violation_metrics(info)
        self._stop_constraint_timing(start, info)
        return (loss, info) if return_info else loss

    @staticmethod
    def _is_homo_batch(batch) -> bool:
        return hasattr(batch, "node_type") and not hasattr(batch, "node_types")

    def _collect_constraint_metrics(self, predictions, batch, constraint_data):
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
        is_training: bool = True,
        sample_count: Optional[int] = None,
    ):
        """
        Update Lagrangian multipliers or penalty parameters.

        Args:
            constraint_violation: Current constraint violation.
            constraints: Constraint vector (for updating multipliers in augmented Lagrangian)
            is_training: Whether updates should run (skip during eval)
            sample_count: Accepted for the trainer interface; updates are epoch-based.
        """
        if self.lagrangian is None or not is_training:
            return

        # After warmup, always update multipliers using the EMA constraints.
        constraint_tensor = constraints if constraints is not None else None
        self.lagrangian.update_multipliers(constraint_tensor)

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

        # Line topology and limits (ac lines + transformers)
        line_edge_index = None
        line_limits = None
        line_edges = []
        line_limits_list = []
        if ('bus', 'ac_line', 'bus') in getattr(batch, 'edge_types', []):
            edge_index = batch['bus', 'ac_line', 'bus'].edge_index.to(device)
            edge_attr = batch['bus', 'ac_line', 'bus'].edge_attr
            if edge_attr is not None and edge_attr.size(1) >= 7:
                edge_attr = edge_attr.to(device)
                limits = edge_attr[:, 6].abs()
                valid = torch.isfinite(limits) & (limits > 0)
                if bool(valid.any().item()):
                    line_edges.append(edge_index[:, valid])
                    line_limits_list.append(limits[valid])
        if ('bus', 'transformer', 'bus') in getattr(batch, 'edge_types', []):
            edge_index = batch['bus', 'transformer', 'bus'].edge_index.to(device)
            edge_attr = batch['bus', 'transformer', 'bus'].edge_attr
            if edge_attr is not None and edge_attr.size(1) >= 5:
                edge_attr = edge_attr.to(device)
                limits = edge_attr[:, 4].abs()
                valid = torch.isfinite(limits) & (limits > 0)
                if bool(valid.any().item()):
                    line_edges.append(edge_index[:, valid])
                    line_limits_list.append(limits[valid])
        if line_edges:
            line_edge_index = torch.cat(line_edges, dim=1)
            line_limits = torch.cat(line_limits_list, dim=0)

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
        """Build full AC Y-bus and per-branch admittances from ac lines/transformers."""
        n_bus = batch['bus'].x.size(0)

        branch_sets = []
        # AC lines: [angmin, angmax, b_fr, b_to, r, x, rate_a, rate_b, rate_c]
        if ('bus', 'ac_line', 'bus') in batch.edge_index_dict:
            edge_index = batch['bus', 'ac_line', 'bus'].edge_index.to(device)
            edge_attr = batch['bus', 'ac_line', 'bus'].edge_attr
            branch = self._build_branch_admittance(
                edge_index=edge_index,
                edge_attr=edge_attr,
                r_idx=4,
                x_idx=5,
                b_fr_idx=2,
                b_to_idx=3,
                rate_a_idx=6,
                tap_idx=None,
                shift_idx=None,
            )
            if branch is not None:
                branch_sets.append(branch)

        # Transformers: [angmin, angmax, r, x, rate_a, rate_b, rate_c, tap, shift, b_fr, b_to]
        if ('bus', 'transformer', 'bus') in batch.edge_index_dict:
            edge_index = batch['bus', 'transformer', 'bus'].edge_index.to(device)
            edge_attr = batch['bus', 'transformer', 'bus'].edge_attr
            branch = self._build_branch_admittance(
                edge_index=edge_index,
                edge_attr=edge_attr,
                r_idx=2,
                x_idx=3,
                b_fr_idx=9,
                b_to_idx=10,
                rate_a_idx=4,
                tap_idx=7,
                shift_idx=8,
            )
            if branch is not None:
                branch_sets.append(branch)

        if not branch_sets:
            return (
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

        bus_i = torch.cat([b["i"] for b in branch_sets], dim=0)
        bus_j = torch.cat([b["j"] for b in branch_sets], dim=0)
        y_ff = torch.cat([b["y_ff"] for b in branch_sets], dim=0)
        y_ft = torch.cat([b["y_ft"] for b in branch_sets], dim=0)
        y_tf = torch.cat([b["y_tf"] for b in branch_sets], dim=0)
        y_tt = torch.cat([b["y_tt"] for b in branch_sets], dim=0)
        rate_a = torch.cat([b["rate_a"] for b in branch_sets], dim=0)

        valid_bus = (bus_i >= 0) & (bus_i < n_bus) & (bus_j >= 0) & (bus_j < n_bus)
        if not bool(valid_bus.any().item()):
            return (
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        bus_i = bus_i[valid_bus]
        bus_j = bus_j[valid_bus]
        y_ff = y_ff[valid_bus]
        y_ft = y_ft[valid_bus]
        y_tf = y_tf[valid_bus]
        y_tt = y_tt[valid_bus]
        rate_a = rate_a[valid_bus]

        shunt_bus_idx = None
        shunt_y = None
        if 'shunt' in getattr(batch, 'node_types', []):
            shunt_x = batch['shunt'].x.to(device)
            shunt_edge = None
            if ('shunt', 'shunt_link', 'bus') in batch.edge_index_dict:
                shunt_edge = batch['shunt', 'shunt_link', 'bus'].edge_index.to(device)
                shunt_src = shunt_edge[0]
                shunt_dst = shunt_edge[1]
            elif ('bus', 'shunt_link', 'shunt') in batch.edge_index_dict:
                shunt_edge = batch['bus', 'shunt_link', 'shunt'].edge_index.to(device)
                shunt_src = shunt_edge[1]
                shunt_dst = shunt_edge[0]
            if shunt_edge is not None and shunt_x is not None and shunt_x.numel() > 0:
                if shunt_x.size(1) >= 2:
                    bs = shunt_x[shunt_src, 0]
                    gs = shunt_x[shunt_src, 1]
                else:
                    bs = shunt_x[shunt_src, 0]
                    gs = torch.zeros_like(bs)
                shunt_bus_idx = shunt_dst
                shunt_y = torch.complex(gs, bs)
                valid_shunt = (shunt_bus_idx >= 0) & (shunt_bus_idx < n_bus)
                if bool(valid_shunt.any().item()):
                    shunt_bus_idx = shunt_bus_idx[valid_shunt]
                    shunt_y = shunt_y[valid_shunt]
                else:
                    shunt_bus_idx = None
                    shunt_y = None

        (
            Y_real_sparse,
            Y_imag_sparse,
            diag_real,
            diag_imag,
        ) = self._build_sparse_ybus_from_branches(
            bus_i=bus_i,
            bus_j=bus_j,
            y_ff=y_ff,
            y_ft=y_ft,
            y_tf=y_tf,
            y_tt=y_tt,
            n_bus=n_bus,
            device=device,
            shunt_bus_idx=shunt_bus_idx,
            shunt_y=shunt_y,
        )

        valid_limits = torch.isfinite(rate_a) & (rate_a > 0)
        if not bool(valid_limits.any().item()):
            return (
                Y_real_sparse,
                Y_imag_sparse,
                diag_real,
                diag_imag,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

        bus_i = bus_i[valid_limits]
        bus_j = bus_j[valid_limits]
        line_edge_index = torch.stack([bus_i, bus_j], dim=0)
        line_limits = rate_a[valid_limits]
        y_ff = y_ff[valid_limits]
        y_ft = y_ft[valid_limits]
        y_tf = y_tf[valid_limits]
        y_tt = y_tt[valid_limits]

        return (
            Y_real_sparse,
            Y_imag_sparse,
            diag_real,
            diag_imag,
            line_edge_index,
            y_ff.real,
            y_ff.imag,
            y_ft.real,
            y_ft.imag,
            y_tf.real,
            y_tf.imag,
            y_tt.real,
            y_tt.imag,
            line_limits,
        )

    def _build_branch_admittance(
        self,
        edge_index,
        edge_attr,
        *,
        r_idx: int,
        x_idx: int,
        b_fr_idx: int,
        b_to_idx: int,
        rate_a_idx: int,
        tap_idx: Optional[int],
        shift_idx: Optional[int],
    ):
        if edge_index is None or edge_attr is None:
            return None
        if edge_attr.size(0) != edge_index.size(1):
            return None
        max_idx = max(
            r_idx,
            x_idx,
            b_fr_idx,
            b_to_idx,
            rate_a_idx,
            tap_idx if tap_idx is not None else 0,
            shift_idx if shift_idx is not None else 0,
        )
        if edge_attr.size(1) <= max_idx:
            return None

        edge_attr = edge_attr.to(edge_index.device)
        r = edge_attr[:, r_idx]
        x = edge_attr[:, x_idx]
        b_fr = edge_attr[:, b_fr_idx]
        b_to = edge_attr[:, b_to_idx]
        rate_a = edge_attr[:, rate_a_idx]
        tap = edge_attr[:, tap_idx] if tap_idx is not None else torch.ones_like(r)
        shift = edge_attr[:, shift_idx] if shift_idx is not None else torch.zeros_like(r)

        r = torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        b_fr = torch.nan_to_num(b_fr, nan=0.0, posinf=0.0, neginf=0.0)
        b_to = torch.nan_to_num(b_to, nan=0.0, posinf=0.0, neginf=0.0)
        rate_a = torch.nan_to_num(rate_a, nan=0.0, posinf=0.0, neginf=0.0).abs()
        tap = torch.nan_to_num(tap, nan=1.0, posinf=1.0, neginf=1.0)
        tap = torch.where(tap.abs() < 1e-6, torch.ones_like(tap), tap)
        shift = torch.nan_to_num(shift, nan=0.0, posinf=0.0, neginf=0.0)

        denom = (r ** 2 + x ** 2).clamp_min(1e-12)
        y_real = r / denom
        y_imag = -x / denom
        y = torch.complex(y_real, y_imag)

        tap_real = tap * torch.cos(shift)
        tap_imag = tap * torch.sin(shift)
        tap_complex = torch.complex(tap_real, tap_imag)
        tap_mag_sq = tap * tap

        y_ff = (y + torch.complex(torch.zeros_like(b_fr), b_fr)) / tap_mag_sq
        y_tt = y + torch.complex(torch.zeros_like(b_to), b_to)
        y_ft = -y / torch.conj(tap_complex)
        y_tf = -y / tap_complex

        return {
            "i": edge_index[0],
            "j": edge_index[1],
            "y_ff": y_ff,
            "y_ft": y_ft,
            "y_tf": y_tf,
            "y_tt": y_tt,
            "rate_a": rate_a,
        }

    def _build_sparse_ybus_from_branches(
        self,
        *,
        bus_i,
        bus_j,
        y_ff,
        y_ft,
        y_tf,
        y_tt,
        n_bus: int,
        device,
        shunt_bus_idx=None,
        shunt_y=None,
    ):
        if bus_i is None or bus_j is None or y_ff is None:
            return None, None, None, None
        if bus_i.numel() == 0 or n_bus <= 0:
            empty_idx = torch.empty((2, 0), dtype=torch.long, device=device)
            empty_val = torch.empty((0,), device=device)
            y_real_sparse = torch.sparse_coo_tensor(empty_idx, empty_val, (n_bus, n_bus))
            y_imag_sparse = torch.sparse_coo_tensor(empty_idx, empty_val, (n_bus, n_bus))
            diag_real = torch.zeros(n_bus, device=device)
            diag_imag = torch.zeros(n_bus, device=device)
            return y_real_sparse, y_imag_sparse, diag_real, diag_imag

        row = torch.cat([bus_i, bus_i, bus_j, bus_j], dim=0)
        col = torch.cat([bus_i, bus_j, bus_i, bus_j], dim=0)
        values = torch.cat([y_ff, y_ft, y_tf, y_tt], dim=0)

        if shunt_bus_idx is not None and shunt_y is not None and shunt_bus_idx.numel() > 0:
            row = torch.cat([row, shunt_bus_idx], dim=0)
            col = torch.cat([col, shunt_bus_idx], dim=0)
            values = torch.cat([values, shunt_y], dim=0)

        indices = torch.stack([row, col], dim=0)
        y_real_sparse = torch.sparse_coo_tensor(indices, values.real, (n_bus, n_bus)).coalesce()
        y_imag_sparse = torch.sparse_coo_tensor(indices, values.imag, (n_bus, n_bus)).coalesce()

        diag_real = torch.zeros(n_bus, device=device)
        diag_imag = torch.zeros(n_bus, device=device)
        diag_real.index_add_(0, bus_i, y_ff.real)
        diag_real.index_add_(0, bus_j, y_tt.real)
        diag_imag.index_add_(0, bus_i, y_ff.imag)
        diag_imag.index_add_(0, bus_j, y_tt.imag)
        if shunt_bus_idx is not None and shunt_y is not None and shunt_bus_idx.numel() > 0:
            diag_real.index_add_(0, shunt_bus_idx, shunt_y.real)
            diag_imag.index_add_(0, shunt_bus_idx, shunt_y.imag)

        return y_real_sparse, y_imag_sparse, diag_real, diag_imag

    def get_loss_info(self) -> Dict:
        """Get information about the configured loss function."""
        return {
            'loss_type': self.loss_type,
            'node_weights': self.node_weights,
            'reduction': self.reduction,
            'epsilon': self.epsilon,
            'beta': self.beta
        }

    def _ensure_network_parameters(self, batch, device, target=None):
        target = target or self.lagrangian
        if target is None:
            return
        if self._is_homo_batch(batch):
            raise ValueError(
                "Augmented Lagrangian network setup requires heterogeneous OPF batches."
            )

        n_bus = batch["bus"].x.size(0)
        current = getattr(target, "Y_real_sparse", None)
        if current is not None and current.size(0) == n_bus:
            return

        values = self._build_admittance_from_batch(batch, device)
        if values[0] is None or values[1] is None:
            raise ValueError("Unable to construct the AC admittance matrix from this batch.")
        (
            y_real,
            y_imag,
            diag_real,
            diag_imag,
            line_edge_index,
            y_ff_real,
            y_ff_imag,
            y_ft_real,
            y_ft_imag,
            y_tf_real,
            y_tf_imag,
            y_tt_real,
            y_tt_imag,
            line_limits,
        ) = values

        base_mva = getattr(batch, "baseMVA", getattr(batch, "base_mva", 100.0))
        if torch.is_tensor(base_mva):
            base_mva = base_mva.reshape(-1)[0].item()
        target.set_network_parameters(
            Y_real_sparse=y_real,
            Y_imag_sparse=y_imag,
            Y_diag_real=diag_real,
            Y_diag_imag=diag_imag,
            line_edge_index=line_edge_index,
            line_y_ff_real=y_ff_real,
            line_y_ff_imag=y_ff_imag,
            line_y_ft_real=y_ft_real,
            line_y_ft_imag=y_ft_imag,
            line_y_tf_real=y_tf_real,
            line_y_tf_imag=y_tf_imag,
            line_y_tt_real=y_tt_real,
            line_y_tt_imag=y_tt_imag,
            line_limits=line_limits,
            base_mva=float(base_mva),
        )
        if target is self.lagrangian:
            self._lagrangian_initialized = True


# Convenience functions for common loss configurations


def create_mse_loss(**kwargs) -> ACOPFLossFunction:
    """Create an ACOPFLossFunction configured with MSE loss.

    Args:
        **kwargs: Additional arguments forwarded to ``ACOPFLossFunction``
            (e.g. ``node_weights``, ``reduction``).

    Returns:
        ACOPFLossFunction: A loss function instance using MSE.
    """
    return ACOPFLossFunction(loss_type='mse', **kwargs)


def create_rmse_loss(**kwargs) -> ACOPFLossFunction:
    """Create an ACOPFLossFunction configured with RMSE loss.

    Args:
        **kwargs: Additional arguments forwarded to ``ACOPFLossFunction``.

    Returns:
        ACOPFLossFunction: A loss function instance using RMSE.
    """
    return ACOPFLossFunction(loss_type='rmse', **kwargs)


def create_mape_loss(**kwargs) -> ACOPFLossFunction:
    """Create an ACOPFLossFunction configured with MAPE loss.

    Args:
        **kwargs: Additional arguments forwarded to ``ACOPFLossFunction``.

    Returns:
        ACOPFLossFunction: A loss function instance using MAPE.
    """
    return ACOPFLossFunction(loss_type='mape', **kwargs)


def create_combined_loss(weights: Dict[str, float], **kwargs) -> ACOPFLossFunction:
    """Create an ACOPFLossFunction with a weighted combination of loss types.

    Args:
        weights (Dict[str, float]): Mapping of loss type names to their
            weights, e.g. ``{'mse': 0.5, 'mae': 0.5}``.
        **kwargs: Additional arguments forwarded to ``ACOPFLossFunction``.

    Returns:
        ACOPFLossFunction: A loss function instance using the specified
            weighted combination.
    """
    return ACOPFLossFunction(loss_type=weights, **kwargs)


def create_robust_loss(**kwargs) -> ACOPFLossFunction:
    """Create an ACOPFLossFunction with a robust loss blend (70% MSE + 30% SmoothL1).

    Args:
        **kwargs: Additional arguments forwarded to ``ACOPFLossFunction``.

    Returns:
        ACOPFLossFunction: A loss function instance using the robust blend.
    """
    return ACOPFLossFunction(
        loss_type={'mse': 0.7, 'smooth_l1': 0.3},
        **kwargs
    )
