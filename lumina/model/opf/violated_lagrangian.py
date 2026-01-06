"""
Violation-based Lagrangian for ACOPF training.

Uses linear Lagrangian term on constraint violations:
    L = f(x) + sum_i lambda_i * violation_i

Violations are abs(eq) for power balance and relu(ineq) for line limits.
Multipliers remain non-negative and are updated with a single penalty parameter mu_k.
"""

from typing import Dict, Optional, Tuple

import torch

from .augmented_lagrangian import AugmentedLagrangianACOPF


class ViolatedLagrangianACOPF(AugmentedLagrangianACOPF):
    """Violation-based Lagrangian that reuses ACOPF constraint computations."""

    def __init__(
        self,
        mu_0: float = 0.001,
        warmup_epochs: int = 0,
        warmup_iter: Optional[int] = None,
        multiplier_clip: float = 100.0,
        **kwargs
    ):
        if warmup_iter is not None:
            warmup_epochs = warmup_iter
        super().__init__(mu_0=mu_0, warmup_epochs=warmup_epochs, multiplier_clip=multiplier_clip, **kwargs)

    def _build_violation_vector(
        self,
        eq_constraints: torch.Tensor,
        ineq_constraints: torch.Tensor
    ) -> torch.Tensor:
        """Build non-negative violation vector for violated-Lagrangian loss."""
        device = eq_constraints.device if eq_constraints is not None else ineq_constraints.device
        eq_violation = (
            eq_constraints.abs().view(-1)
            if eq_constraints.numel() > 0
            else torch.tensor([], device=device)
        )
        ineq_violation = (
            torch.relu(ineq_constraints).view(-1)
            if ineq_constraints.numel() > 0
            else torch.tensor([], device=device)
        )

        if eq_violation.numel() > 0 and ineq_violation.numel() > 0:
            return torch.cat([eq_violation, ineq_violation])
        if eq_violation.numel() > 0:
            return eq_violation
        if ineq_violation.numel() > 0:
            return ineq_violation
        return torch.tensor([], device=device)

    def compute_violated_lagrangian(
        self,
        mse_loss: torch.Tensor,
        eq_constraints: torch.Tensor,
        ineq_constraints: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute violated Lagrangian loss:
        L_V(x, lambda) = f(x) + sum_i lambda_i * |violation_i|
        """
        violation_vector = self._build_violation_vector(eq_constraints, ineq_constraints)
        if violation_vector.numel() == 0:
            zero = torch.tensor(0.0, device=mse_loss.device)
            self._raw_violation = 0.0
            self._ema_violation = 0.0
            self._latest_constraints = None
            self._latest_constraint_signal = None
            return mse_loss, {
                'objective': mse_loss,
                'lagrange_term': zero,
                'constraint_violation': zero,
                'raw_constraint_violation': zero,
                'ema_constraint_violation': zero,
                'multipliers_active': False,
                'constraints': violation_vector,
            }

        # Get EMA constraint signal if available, otherwise use raw violations
        constraint_signal = self._constraint_signal_for_loss(violation_vector)

        if self.lambda_k is None or self.lambda_k.numel() != constraint_signal.numel():
            self.lambda_k = torch.zeros_like(constraint_signal)
        elif self.lambda_k.device != constraint_signal.device:
            self.lambda_k = self.lambda_k.to(constraint_signal.device)

        lagrange_term = torch.tensor(0.0, device=mse_loss.device)
        if self._should_use_multipliers():
            lagrange_term = torch.dot(self.lambda_k, constraint_signal)

        total_loss = mse_loss + lagrange_term

        violation_value = self._ema_violation if self._ema_violation is not None else self._raw_violation
        if violation_value is None:
            violation_value = self._compute_violation_norm(violation_vector)
        self.constraint_history.append(violation_value)
        device = mse_loss.device
        constraint_violation = torch.tensor(violation_value, device=device)
        raw_violation = torch.tensor(self._raw_violation if self._raw_violation is not None else violation_value,
                                     device=device)
        ema_violation = torch.tensor(self._ema_violation if self._ema_violation is not None else violation_value,
                                     device=device)

        last_multiplier_norm = torch.tensor(
            self._last_multiplier_norm if self._last_multiplier_norm is not None else 0.0,
            device=device
        )
        last_multiplier_violation = torch.tensor(
            self._last_multiplier_violation if self._last_multiplier_violation is not None else violation_value,
            device=device
        )

        components = {
            'objective': mse_loss,
            'lagrange_term': lagrange_term,
            'constraint_violation': constraint_violation,
            'raw_constraint_violation': raw_violation,
            'ema_constraint_violation': ema_violation,
            'p_balance_rmse': self._last_p_balance_rms,
            'q_balance_rmse': self._last_q_balance_rms,
            'line_limit_rmse': self._last_line_limit_rms,
            'multipliers_active': self._should_use_multipliers(),
            'last_multiplier_norm': last_multiplier_norm,
            'last_multiplier_violation': last_multiplier_violation,
            'constraints': violation_vector.detach(),
        }

        return total_loss, components

    def forward(
        self,
        mse_loss: torch.Tensor,
        predictions: Dict[str, torch.Tensor],
        data: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        eq_constraints, ineq_constraints = self.compute_constraint_components(predictions, data)
        lag_loss, components = self.compute_violated_lagrangian(mse_loss, eq_constraints, ineq_constraints)
        constraints = components.get('constraints', torch.tensor([], device=mse_loss.device))

        info = {
            **components,
            'penalty_parameter': self.mu_k,
            'n_constraints': constraints.numel(),
            'constraints': constraints.detach(),
            'multipliers_active': self._should_use_multipliers(),
        }

        return lag_loss, info

    def _apply_multiplier_update(self, constraint_signal: torch.Tensor):
        update = self.mu_k * constraint_signal.detach()
        self.lambda_k = self.lambda_k + update

        if self.multiplier_clip is not None:
            self.lambda_k = torch.clamp(self.lambda_k, 0.0, self.multiplier_clip)
        else:
            self.lambda_k = torch.clamp_min(self.lambda_k, 0.0)

    def _post_multiplier_update(self):
        self.lagrange_history.append(self.lambda_k.clone().cpu().numpy())

    def step_epoch(self):
        """Increment epoch counter; sample-based scheduling uses step_samples."""
        self._schedule.advance_epoch()

    def reset_for_new_problem(self):
        """Reset algorithm state for a new optimization problem."""
        super().reset_for_new_problem()
