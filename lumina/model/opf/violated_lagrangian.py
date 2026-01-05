"""
Violation-based Lagrangian for ACOPF training.

Uses linear Lagrangian term on constraint violations:
    L = f(x) + sum_i lambda_i * violation_i

Violations are abs(eq) for power balance and relu(ineq) for line limits.
Multipliers remain non-negative and are updated with a single penalty parameter mu_k.
"""

import warnings
from typing import Dict, Optional, Tuple

import torch

from .augmented_lagrangian import AugmentedLagrangianACOPF


class ViolatedLagrangianACOPF(AugmentedLagrangianACOPF):
    """Violation-based Lagrangian that reuses ACOPF constraint computations."""

    def __init__(
        self,
        rho: float = 0.001,
        warmup_epochs: int = 0,
        warmup_iter: Optional[int] = None,
        multiplier_clip: float = 100.0,
        **kwargs
    ):
        if warmup_iter is not None:
            warmup_epochs = warmup_iter
        super().__init__(warmup_epochs=warmup_epochs, multiplier_clip=multiplier_clip, **kwargs)
        self.mu_k = rho

    def compute_power_flow_constraints(
        self,
        vm_pred: torch.Tensor,
        va_pred: torch.Tensor,
        pg_pred: torch.Tensor,
        qg_pred: torch.Tensor,
        pd: torch.Tensor,
        qd: torch.Tensor,
        gen_bus_indices: torch.Tensor,
        load_bus_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute power flow equality constraints c_i(x) = 0.

        .. math: :
            P_i ^ {inj} - P_i ^ {calc} = 0  # Active power balance
            Q_i ^ {inj} - Q_i ^ {calc} = 0  # Reactive power balance
        where:
            P_i ^ {inj} = \\sum P_g at bus i - \\sum P_d at bus i
            Q_i^{inj} = \\sum Q_g at bus i - \\sum Q_d at bus i
            P_i^{calc}, Q_i^{calc} computed from admittance matrix (Y) and voltages

        Args:
            vm (torch.Tensor): Voltage magnitudes at buses [batch_size, n_bus]
            va (torch.Tensor): Voltage angles at buses [batch_size, n_bus] in radians
            pg (torch.Tensor): Active power generation at generators [batch_size, n_gen]
            qg (torch.Tensor): Reactive power generation at generators [batch_size, n_gen]
            pd (torch.Tensor): Active power demand at load buses [batch_size, n_load]
            qd (torch.Tensor): Reactive power demand at load buses [batch_size, n_load]
            gen_bus_indices (torch.Tensor): Indices of generator buses [n_gen]
            load_bus_indices (torch.Tensor): Indices of load buses [n_load]

        Returns:
            Constraint violations [c_1(x), c_2(x), ..., c_n(x)]
            where c_i(x) = 0 represents power balance at bus i
        """
        if self.Y_real_sparse is None or self.Y_imag_sparse is None:
            return torch.tensor([], device=vm_pred.device, requires_grad=True)

        # Detect actual batch size and number of buses
        if vm_pred.dim() == 1:
            # If 1D, it could be a single sample or flattened batch
            n_bus = self.Y_real_sparse.shape[0]
            if vm_pred.numel() % n_bus == 0:
                batch_size = vm_pred.numel() // n_bus
                vm_pred = vm_pred.view(batch_size, n_bus)
                va_pred = va_pred.view(batch_size, n_bus)
                if pg_pred is not None:
                    pg_pred = pg_pred.view(batch_size, -1) if pg_pred.numel() > 1 else pg_pred.unsqueeze(0)
                if qg_pred is not None:
                    qg_pred = qg_pred.view(batch_size, -1) if qg_pred.numel() > 1 else qg_pred.unsqueeze(0)
                single_sample = batch_size == 1
            else:
                batch_size = 1
                vm_pred = vm_pred.unsqueeze(0)
                va_pred = va_pred.unsqueeze(0)
                single_sample = True
        else:
            batch_size = vm_pred.size(0)
            single_sample = False

        # Ensure pd and qd have proper batch dimension
        if pd is not None and pd.dim() == 1:
            pd = pd.unsqueeze(0) if batch_size == 1 else pd.view(batch_size, -1)
        if qd is not None and qd.dim() == 1:
            qd = qd.unsqueeze(0) if batch_size == 1 else qd.view(batch_size, -1)

        # OPFData stores angles in radians by default.
        va_rad = torch.deg2rad(va_pred) if self.angles_in_degrees else va_pred

        # Compute voltage phasors
        v_real = vm_pred * torch.cos(va_rad)
        v_imag = vm_pred * torch.sin(va_rad)

        # Initialize power injections at buses
        p_inj = torch.zeros_like(v_real)
        q_inj = torch.zeros_like(v_imag)

        # Guarantee tensors are 2D for scatter_add and keep batch size aligned.
        if p_inj.dim() < 2:
            p_inj = p_inj.view(1, -1)
            q_inj = q_inj.view(1, -1)
        if vm_pred.dim() < 2:
            vm_pred = vm_pred.view(p_inj.shape[0], -1)
            va_pred = va_pred.view(p_inj.shape[0], -1)
        if pg_pred is not None and pg_pred.dim() < 2:
            pg_pred = pg_pred.view(1, -1)
        if qg_pred is not None and qg_pred.dim() < 2:
            qg_pred = qg_pred.view(1, -1)
        batch_size = p_inj.size(0)

        # Add generation at generator buses (already in per-unit, so no division by base_mva)
        if gen_bus_indices is not None and pg_pred is not None:
            # Ensure gen_bus_indices doesn't exceed the number of buses or generators
            max_gen_idx = min(len(gen_bus_indices), pg_pred.shape[1] if pg_pred.dim() > 1 else pg_pred.shape[0])
            gen_indices_limited = gen_bus_indices[:max_gen_idx]

            gen_indices_expanded = gen_indices_limited.unsqueeze(0).expand(batch_size, -1).to(pg_pred.device)

            # Limit pg and qg to match the indices
            pg_limited = pg_pred[:, :max_gen_idx] if pg_pred.dim() > 1 else pg_pred[:max_gen_idx]
            qg_limited = qg_pred[:, :max_gen_idx] if qg_pred.dim() > 1 else qg_pred[:max_gen_idx]

            # Assuming predictions are already in per-unit
            p_inj.scatter_add_(1, gen_indices_expanded, pg_limited)
            q_inj.scatter_add_(1, gen_indices_expanded, qg_limited)

        # Subtract loads at load buses (assuming loads are also in per-unit)
        if pd is not None:
            if qd is None:
                qd = torch.zeros_like(pd)
            if pd.dim() == 1:
                pd = pd.unsqueeze(0) if batch_size == 1 else pd.view(batch_size, -1)
            if qd.dim() == 1:
                qd = qd.unsqueeze(0) if batch_size == 1 else qd.view(batch_size, -1)

            if pd.size(1) == p_inj.size(1):
                # pd already aggregated per bus.
                p_inj = p_inj - pd
                q_inj = q_inj - qd
            elif load_bus_indices is not None and pd.size(1) == load_bus_indices.numel():
                load_indices_expanded = load_bus_indices.unsqueeze(0).expand(batch_size, -1).to(pd.device)
                p_inj.scatter_add_(1, load_indices_expanded, -pd)
                q_inj.scatter_add_(1, load_indices_expanded, -qd)

        # Compute power flows using full AC equations via Y-bus currents
        device = v_real.device

        def _sparse_mm_cpu(v_real_local, v_imag_local):
            if self._Y_real_sparse_cpu is None or self._Y_imag_sparse_cpu is None:
                y_real_cpu = self.Y_real_sparse.coalesce().cpu()
                y_imag_cpu = self.Y_imag_sparse.coalesce().cpu()
                if y_real_cpu.dtype in (torch.float16, torch.bfloat16):
                    y_real_cpu = y_real_cpu.float()
                    y_imag_cpu = y_imag_cpu.float()
                self._Y_real_sparse_cpu = y_real_cpu
                self._Y_imag_sparse_cpu = y_imag_cpu
            v_real_cpu = v_real_local if v_real_local.device.type == "cpu" else v_real_local.to("cpu")
            v_imag_cpu = v_imag_local if v_imag_local.device.type == "cpu" else v_imag_local.to("cpu")
            target_dtype = self._Y_real_sparse_cpu.dtype
            if v_real_cpu.dtype != target_dtype:
                v_real_cpu = v_real_cpu.to(target_dtype)
                v_imag_cpu = v_imag_cpu.to(target_dtype)
            v_real_t_cpu = v_real_cpu.transpose(0, 1)
            v_imag_t_cpu = v_imag_cpu.transpose(0, 1)
            i_real_t_cpu = torch.sparse.mm(self._Y_real_sparse_cpu, v_real_t_cpu) - torch.sparse.mm(
                self._Y_imag_sparse_cpu,
                v_imag_t_cpu,
            )
            i_imag_t_cpu = torch.sparse.mm(self._Y_real_sparse_cpu, v_imag_t_cpu) + torch.sparse.mm(
                self._Y_imag_sparse_cpu,
                v_real_t_cpu,
            )
            return i_real_t_cpu, i_imag_t_cpu

        if device.type != "cuda" or self._sparse_mm_supported is False:
            i_real_t, i_imag_t = _sparse_mm_cpu(v_real, v_imag)
        else:
            try:
                y_real_sparse = self.Y_real_sparse
                y_imag_sparse = self.Y_imag_sparse
                if y_real_sparse.device != device:
                    y_real_sparse = y_real_sparse.to(device)
                if y_imag_sparse.device != device:
                    y_imag_sparse = y_imag_sparse.to(device)
                if y_real_sparse.layout == torch.sparse_coo:
                    y_real_sparse = y_real_sparse.coalesce()
                if y_imag_sparse.layout == torch.sparse_coo:
                    y_imag_sparse = y_imag_sparse.coalesce()
                v_real_t = v_real.transpose(0, 1)
                v_imag_t = v_imag.transpose(0, 1)
                i_real_t = torch.sparse.mm(y_real_sparse, v_real_t) - torch.sparse.mm(y_imag_sparse, v_imag_t)
                i_imag_t = torch.sparse.mm(y_real_sparse, v_imag_t) + torch.sparse.mm(y_imag_sparse, v_real_t)
                if self._sparse_mm_supported is None:
                    self._sparse_mm_supported = True
            except RuntimeError as exc:
                if self._sparse_mm_supported is not False:
                    warnings.warn(f"Sparse mm failed on device {device}; falling back to CPU. {exc}")
                self._sparse_mm_supported = False
                i_real_t, i_imag_t = _sparse_mm_cpu(v_real, v_imag)

        i_real = i_real_t.transpose(0, 1).to(device)
        i_imag = i_imag_t.transpose(0, 1).to(device)

        p_calc = v_real * i_real + v_imag * i_imag
        q_calc = v_imag * i_real - v_real * i_imag

        # Power balance constraints: injection - calculated_flow = 0
        p_balance = (p_inj - p_calc).abs()
        q_balance = (q_inj - q_calc).abs()

        # Compute RMS of P and Q balance across buses (averaged over batch if present)
        try:
            p_mean = p_balance.mean(dim=0) if p_balance.dim() > 1 else p_balance
            q_mean = q_balance.mean(dim=0) if q_balance.dim() > 1 else q_balance
            p_rms = torch.sqrt(torch.mean(p_mean**2) + 1e-12)
            q_rms = torch.sqrt(torch.mean(q_mean**2) + 1e-12)
            # store as plain Python floats for easy external inspection
            self._last_p_balance_rms = float(p_rms.detach().item())
            self._last_q_balance_rms = float(q_rms.detach().item())
        except (RuntimeError, ValueError) as exc:
            # Log issues while keeping the training loop robust
            warnings.warn(f"Failed to compute power balance RMS: {exc}")
            self._last_p_balance_rms = None
            self._last_q_balance_rms = None

        # Combine P and Q constraints: [P_1, P_2, ..., P_n, Q_1, Q_2, ..., Q_n]
        constraints = torch.cat([p_balance, q_balance], dim=-1)  # Shape: [batch_size, 2*n_bus]

        if single_sample:
            constraints = constraints.squeeze(0)

        # For batch processing, we take the mean across batch dimension
        if constraints.dim() > 1:
            constraints = constraints.mean(dim=0)

        # Normalize by number of buses and scale properly
        if self.normalize_constraints:
            # Scale constraints by RMS value and number of buses for better conditioning
            rms_value = torch.sqrt(torch.mean(constraints**2)) + 1e-8
            # Additional scaling by square root of number of constraints for stability
            scale_factor = torch.sqrt(torch.tensor(float(len(constraints)), device=constraints.device))
            constraints = constraints / (rms_value * scale_factor)

        return constraints

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
