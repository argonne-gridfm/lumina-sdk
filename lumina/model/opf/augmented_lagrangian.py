"""
Augmented Lagrangian Method for ACOPF Neural Network Training

This module implements Framework 17.3 from the optimization literature, treating:
- MSE loss as the objective function f(x)
- Power flow equality constraints as equality constraints c_i(x) = 0
- Line flow limit constraints as inequality constraints (converted to equality with slack variables)

The augmented Lagrangian function is:
..math::
    L_A(x, \\lambda; \\mu) = f(x) - \\sum_i \\lambda_i c_i(x) + (\\mu/2) \\sum_i c_i(x)^2
                        (optional)
                           = [f(x) - \\sum_{i \\in E} \\lambda_i c_i(x)] /\\mu + (1/2) \\sum_i c_i(x)^2

Copyright (c) 2025, Argonne National Laboratory
All rights reserved.
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize


class AugmentedLagrangian(nn.Module):
    """
    General Augmented Lagrangian class for constrained optimization.

    Solves problems of the form:
        min f(x)
        s.t. c_i(x) = 0  for i = 1, ..., m (equality constraints)

    The augmented Lagrangian function is:
        L_A(x, λ, μ) = f(x) - Σ λ_i * c_i(x) + (μ/2) * Σ c_i(x)²
    """

    def __init__(self,
                 objective_func=None,
                 constraint_funcs=None,
                 mu_0: float = 1.0,
                 tolerance: float = 1e-6,
                 mu_increase_factor: float = 1.5,
                 max_mu: float = 1000.0,
                 constraint_tolerance: float = 1e-4,
                 max_outer_iterations: int = 20,
                 max_inner_iterations: int = 50,
                 verbose: bool = True):
        """
        Initialize Augmented Lagrangian solver.

        Args:
            objective_func: Function f(x) to minimize
            constraint_funcs: Single constraint function or list of constraint functions.
                             Each function should return constraint values that should equal 0
            mu_0: Initial penalty parameter
            tolerance: Convergence tolerance
            mu_increase_factor: Factor to increase penalty parameter
            max_mu: Maximum penalty parameter value
            constraint_tolerance: Tolerance for constraint satisfaction
            max_outer_iterations: Maximum outer iterations
            max_inner_iterations: Maximum inner iterations per subproblem
            verbose: Whether to print optimization progress
        """
        super().__init__()

        # User-provided functions
        self.objective_func = objective_func

        # Handle constraint functions - support both single function and list
        if callable(constraint_funcs):
            self.constraint_func = [constraint_funcs]
        elif isinstance(constraint_funcs, (list, tuple)):
            self.constraint_func = list(constraint_funcs)
        else:
            self.constraint_func = constraint_funcs

        # Algorithm parameters
        self.mu_0 = mu_0
        self.tolerance = tolerance
        self.mu_increase_factor = mu_increase_factor
        self.max_mu = max_mu
        self.constraint_tolerance = constraint_tolerance
        self.max_outer_iterations = max_outer_iterations
        self.max_inner_iterations = max_inner_iterations
        self.verbose = verbose

        # Current algorithm state
        self.mu_k = mu_0
        self.lambda_k = None  # Lagrange multipliers
        self.outer_iteration = 0

        # Constraint tracking
        self.constraint_history = []
        self.lambda_history = []
        self.mu_history = []
        self.objective_history = []

    def set_functions(self, objective_func,
                      constraint_funcs):
        """Set the objective and constraint functions.

        Args:
            objective_func: Single objective function f(x)
            constraint_funcs: Single constraint function or list of constraint functions
        """
        self.objective_func = objective_func

        # Handle both single function and list of functions
        if callable(constraint_funcs):
            self.constraint_func = [constraint_funcs]
        elif isinstance(constraint_funcs, (list, tuple)):
            self.constraint_func = list(constraint_funcs)
        else:
            self.constraint_func = constraint_funcs

    def f(self, x: np.ndarray) -> float:
        """Objective function f(x).

        Args:
            x: Current variable values

        Returns:
            (float): Objective function value at x
        """
        if self.objective_func is None:
            raise NotImplementedError("Objective function not set. Use set_functions() or override this method.")
        return self.objective_func(x)

    def c(self, x: np.ndarray) -> np.ndarray:
        """
        Constraint function c(x). Evaluates all constraint functions and returns combined array.

        Args:
            x: Current variable values

        Returns:
            Combined constraint values from all constraint functions
        """
        if self.constraint_func is None:
            raise NotImplementedError("Constraint function not set. Use set_functions() or override this method.")

        # Handle list of constraint functions (always the case after initialization)
        if isinstance(self.constraint_func, (list, tuple)):
            all_constraints = []

            for i, constraint_fn in enumerate(self.constraint_func):
                constraint_val = constraint_fn(x)

                # # Convert to array and flatten if needed
                # if np.isscalar(constraint_val):
                #     constraint_array = np.array([constraint_val])
                # else:
                #     constraint_array = np.array(constraint_val).flatten()

                all_constraints.append(constraint_val)

            return np.array(all_constraints)

        # Fallback for single function stored as non-callable (should not occur)
        return np.array(self.constraint_func(x))

    def augmented_lagrangian(self, x: np.ndarray) -> float:
        """Compute the augmented Lagrangian function.

        Args:
            x: Current variable values

        Returns:
            Augmented Lagrangian value at x
        """
        obj = self.f(x)
        constraints = self.c(x)

        # Handle both scalar and vector constraints
        if np.isscalar(constraints):
            constraints = np.array([constraints])
        else:
            constraints = np.array(constraints)

        # Augmented Lagrangian: L(x,λ,μ) = f(x) - λᵀc(x) + (μ/2)||c(x)||²
        lagrangian_term = -np.dot(self.lambda_k, constraints)
        penalty_term = 0.5 * self.mu_k * np.sum(constraints**2)

        return obj + lagrangian_term + penalty_term

    def update_multipliers(self, x: np.ndarray):
        """Update Lagrange multipliers: λ = λ - μ * c(x).

        Args:
            x: Current variable values
        """
        constraints = self.c(x)
        if hasattr(constraints, 'shape'):
            constraint_array = constraints
        else:
            constraint_array = np.array([constraints]) if np.isscalar(constraints) else np.array(constraints)

        self.lambda_k = self.lambda_k - self.mu_k * constraint_array

    def update_penalty_parameter(self):
        """Increase penalty parameter μ.

        .. math::
            μ_k = min(μ * ρ, μ_{max}),
            where ρ > 1 is the increase factor, and μ_{max} is the maximum allowed value.

        """
        self.mu_k = min(self.mu_k * self.mu_increase_factor, self.max_mu)

    def solve(self, x0: np.ndarray, max_outer_iterations: int = 100, tolerance: float = 1e-6) -> dict:
        """
        Solve the constrained optimization problem using augmented Lagrangian method.

        Args:
            x0: Initial guess for variables
            max_outer_iterations: Maximum number of outer iterations
            tolerance: Convergence tolerance for constraint violation

        Returns:
            Dictionary with solution results
        """
        x = np.array(x0, dtype=float)

        # Initialize constraints to determine dimensions
        constraints_init = self.c(x)
        obj_init = self.f(x)

        self.constraint_history.append(constraints_init)
        self.lambda_history.append(self.lambda_k)
        self.mu_history.append(self.mu_k)
        self.objective_history.append(obj_init)

        # Handle both scalar and vector constraints
        if np.isscalar(constraints_init):
            n_constraints = 1
            constraints_init = np.array([constraints_init])
        else:
            constraints_init = np.array(constraints_init)
            n_constraints = len(constraints_init)

        # Initialize Lagrange multipliers
        if self.lambda_k is None:
            self.lambda_k = np.zeros(n_constraints)

        converged = False

        for iteration in range(max_outer_iterations):
            # Minimize augmented Lagrangian using scipy
            result = minimize(
                self.augmented_lagrangian,
                x,
                method='L-BFGS-B',
                options={'maxiter': self.max_inner_iterations,
                         'disp': False}
            )

            x = result.x

            # Check convergence
            constraints = self.c(x)
            # Handle both scalar and vector constraints
            if np.isscalar(constraints):
                constraints = np.array([constraints])
            else:
                constraints = np.array(constraints)

            constraint_violation = np.linalg.norm(constraints)

            if self.verbose:
                obj_val = self.f(x)
                print(f"Iteration {iteration}: obj={obj_val:.6f}, constraint_violation={constraint_violation:.6f}")

            if constraint_violation < tolerance:
                converged = True
                break

            # # Update Lagrange multipliers: λ := λ - μ * c(x)
            self.lambda_k -= self.mu_k * constraints
            # self.update_multipliers(x)

            # Update penalty parameter
            self.update_penalty_parameter()

            self.constraint_history.append(constraints)
            self.lambda_history.append(self.lambda_k)
            self.mu_history.append(self.mu_k)
            self.objective_history.append(self.f(x))

        return {
            'x': x,
            'fun': self.f(x),
            'success': converged,
            'objective': self.f(x),
            'constraint_violation': constraint_violation,
            'converged': converged,
            'nit': iteration + 1,
            'iterations': iteration + 1,
            'lambda': self.lambda_k,
            'mu': self.mu_k
        }

    def forward(self, x: np.ndarray) -> float:
        """Forward pass - computes augmented Lagrangian value."""
        return self.augmented_lagrangian(x)


class AugmentedLagrangianACOPF(nn.Module):
    """
    Augmented Lagrangian method for ACOPF neural network training.

    Implements augmented Lagrangian with:
    - Objective: MSE loss f(x)
        TODO: add the cost function to the f(x)
    - Equality constraints: Power flow balance c_i(x) = 0
    - Inequality constraints: Line flow limits(converted to equality)
    """

    def __init__(
        self,
        mu_0: float = 1.0,
        tolerance: float = 1e-6,
        mu_increase_factor: float = 1.5,
        max_mu: float = 1000.0,
        constraint_tolerance: float = 1e-4,
        max_outer_iterations: int = 20,
        max_inner_iterations: int = 50,
        normalize_constraints: bool = False,
        verbose: bool = False,
        warmup_epochs: int = 0,
        ema_beta: float = 0.9,
        penalty_check_interval: int = 5,
        penalty_drop_ratio: float = 0.5,
        penalty_increase_factor: Optional[float] = None,
        min_penalty: Optional[float] = None,
        max_penalty: Optional[float] = None,
        multiplier_clip: float = 1000.0,
        multiplier_improve_ratio: float = 0.0,
        multiplier_check_interval: int = 1,
        multiplier_min_improve: float = 0.0,
        **_
    ):
        """
        Initialize Augmented Lagrangian solver.

        Args:
            mu_0(float): Initial penalty parameter
            tolerance(float): Convergence tolerance for overall algorithm
            mu_increase_factor(float): Factor to increase penalty parameter
            max_mu(float): Maximum penalty parameter value
            constraint_tolerance(float): Tolerance for constraint satisfaction
            max_outer_iterations(int): Maximum outer iterations(penalty updates)
            max_inner_iterations(int): Maximum inner iterations(optimization steps)
            normalize_constraints(bool): Whether to normalize constraint violations
            verbose(bool): Unused flag kept for backward compatibility with older configs
            warmup_epochs(int): Number of epochs to run penalty-only (λ = 0)
            ema_beta(float): Exponential moving average factor for constraint violations
            penalty_check_interval(int): Epoch interval for checking penalty increases
            penalty_drop_ratio(float): Required reduction ratio to avoid penalty increase
            penalty_increase_factor(Optional[float]): Factor to scale μ when increasing
            min_penalty(Optional[float]): Lower bound for μ (defaults to mu_0)
            max_penalty(Optional[float]): Upper bound for μ (defaults to max_mu)
            multiplier_clip(float): Clamp range for λ updates
            multiplier_improve_ratio(float): Require violation ≤ ratio * last_violation to update λ (<=0 disables check)
            multiplier_check_interval(int): Minimum steps between multiplier updates
            multiplier_min_improve(float): Absolute improvement required to update λ (0 disables)
            **_: Ignore extra keyword arguments for forward compatibility
        """
        super().__init__()

        # Algorithm parameters
        self.mu_0 = mu_0
        self.tolerance = tolerance
        self.mu_increase_factor = mu_increase_factor
        self.penalty_increase_factor = penalty_increase_factor or mu_increase_factor
        self.max_mu = max_mu
        self.constraint_tolerance = constraint_tolerance
        self.max_outer_iterations = max_outer_iterations
        self.max_inner_iterations = max_inner_iterations
        self.normalize_constraints = normalize_constraints
        self.verbose = verbose
        self.warmup_epochs = warmup_epochs
        self.ema_beta = ema_beta
        self.penalty_check_interval = max(1, penalty_check_interval)
        self.penalty_drop_ratio = penalty_drop_ratio
        self.multiplier_clip = multiplier_clip
        self.multiplier_improve_ratio = multiplier_improve_ratio
        self.multiplier_check_interval = max(1, multiplier_check_interval)
        self.multiplier_min_improve = multiplier_min_improve
        self.min_penalty = min_penalty if min_penalty is not None else mu_0
        self.max_penalty = max_penalty if max_penalty is not None else max_mu
        self.max_mu = self.max_penalty

        # Current algorithm state
        self.mu_k = float(np.clip(mu_0, self.min_penalty, self.max_penalty))
        self.lambda_k = None  # Lagrange multipliers
        self.outer_iteration = 0
        self.current_epoch = 0

        # Network parameters (to be set)
        self.Y_real = None
        self.Y_imag = None
        self.line_limits = None
        self.base_mva = 100.0

        # Constraint tracking
        self.constraint_history = []
        self.lagrange_history = []
        self.penalty_history = []
        self.constraint_ema: Optional[torch.Tensor] = None
        self._latest_constraints: Optional[torch.Tensor] = None
        self._latest_constraint_signal: Optional[torch.Tensor] = None
        self._raw_violation: Optional[float] = None
        self._ema_violation: Optional[float] = None
        self._epochs_since_penalty_check = 0
        self._last_penalty_check_violation: Optional[float] = None
        self._last_multiplier_norm: Optional[float] = None
        self._last_multiplier_violation: Optional[float] = None
        self._multiplier_steps_since_update: int = 0
        self._last_multiplier_updated: bool = False
        # Last computed RMS values for P and Q balance and line limit(float or None)
        self._last_p_balance_rms: Optional[float] = None
        self._last_q_balance_rms: Optional[float] = None
        self._last_line_limit_rms: Optional[float] = None

    def set_network_parameters(
        self,
        Y_real: torch.Tensor,
        Y_imag: torch.Tensor,
        line_limits: Optional[torch.Tensor] = None,
        base_mva: float = 100.0
    ):
        """Set network parameters for constraint computation.

        Args:
            Y_real(torch.Tensor): Real part of admittance matrix(n_bus x n_bus)
            Y_imag(torch.Tensor): Imaginary part of admittance matrix(n_bus x n_bus)
            line_limits(Optional[torch.Tensor]): Line flow limits(n_lines x 2)
            base_mva(float): Base MVA for power system
        """
        self.Y_real = Y_real
        self.Y_imag = Y_imag
        self.line_limits = line_limits
        self.base_mva = base_mva

        # Initialize Lagrange multipliers
        n_bus = Y_real.size(0)
        device = Y_real.device

        # Power flow equality constraints: 2 * n_bus (P and Q balance at each bus)
        n_equality = 2 * n_bus

        # Line flow inequality constraints (if provided)
        n_inequality = len(line_limits) if line_limits is not None else 0

        # Total constraints (inequalities converted to equality with slack)
        total_constraints = n_equality + n_inequality

        # Initialize Lagrange multipliers to zero
        self.lambda_k = torch.zeros(total_constraints, device=device)
        self.constraint_ema = torch.zeros_like(self.lambda_k)
        self._latest_constraints = None
        self._latest_constraint_signal = None
        self._raw_violation = None
        self._ema_violation = None
        self._last_penalty_check_violation = None
        self._epochs_since_penalty_check = 0

        # print("Initialized Augmented Lagrangian with:")
        # print(f"- Power flow constraints: {n_equality}")
        # print(f"- Line flow constraints: {n_inequality}")
        # print(f"- Total constraints: {total_constraints}")
        # print(f"- Initial penalty parameter μ: {self.mu_k}")

    def _should_use_multipliers(self) -> bool:
        """Return True when multiplier updates are allowed (post-warmup)."""
        return self.current_epoch >= self.warmup_epochs

    def _init_constraint_buffers(self, constraints: torch.Tensor):
        """Initialize buffers that depend on the constraint dimension."""
        if constraints is None or constraints.numel() == 0:
            return
        if self.constraint_ema is None or self.constraint_ema.numel() != constraints.numel():
            self.constraint_ema = torch.zeros_like(constraints.detach())
        if self.lambda_k is None or self.lambda_k.numel() != constraints.numel():
            self.lambda_k = torch.zeros(constraints.numel(), device=constraints.device)

    def _compute_violation_norm(self, constraints: Optional[torch.Tensor]) -> float:
        """Compute L2 norm of a constraint vector (safe for None or empty)."""
        if constraints is None or constraints.numel() == 0:
            return 0.0
        return float(torch.norm(constraints, p=2).detach().item())

    def _constraint_signal_for_loss(self, constraints: torch.Tensor) -> torch.Tensor:
        """
        Compute the smoothed constraint signal used for both penalty and multiplier terms.

        EMA buffer is updated without gradients; the returned tensor preserves gradients
        through the current constraints (scaled by 1 - beta).
        """
        if constraints.numel() == 0:
            return constraints

        self._init_constraint_buffers(constraints)

        # EMA for the loss (keeps gradients for the current step)
        ema_for_loss = self.ema_beta * self.constraint_ema + (1 - self.ema_beta) * constraints

        # Update running EMA buffer without tracking gradients
        with torch.no_grad():
            self.constraint_ema = self.ema_beta * self.constraint_ema + (1 - self.ema_beta) * constraints.detach()

        # Cache latest signals for logging/updating
        self._latest_constraints = constraints.detach()
        self._latest_constraint_signal = self.constraint_ema.detach()
        self._raw_violation = self._compute_violation_norm(constraints)
        self._ema_violation = self._compute_violation_norm(self.constraint_ema)

        return ema_for_loss

    def _maybe_update_penalty(self, current_violation: Optional[float], force: bool = False):
        """Increase μ if violations are not shrinking enough based on EMA trend."""
        if current_violation is None:
            return
        if torch.is_tensor(current_violation):
            current_violation = float(current_violation.detach().item())

        # Always ensure μ stays within bounds
        self.mu_k = float(np.clip(self.mu_k, self.min_penalty, self.max_penalty))

        # During warmup, just seed the baseline and skip updates
        if self.current_epoch < self.warmup_epochs and not force:
            self._last_penalty_check_violation = current_violation
            return

        if not force:
            self._epochs_since_penalty_check += 1
            if self._epochs_since_penalty_check < self.penalty_check_interval:
                return
            self._epochs_since_penalty_check = 0

        if self._last_penalty_check_violation is None:
            self._last_penalty_check_violation = current_violation
            self.penalty_history.append(self.mu_k)
            return

        # If violation did not drop enough, bump μ
        if current_violation > self.penalty_drop_ratio * self._last_penalty_check_violation:
            old_mu = self.mu_k
            self.mu_k = min(max(self.mu_k * self.penalty_increase_factor, self.min_penalty), self.max_penalty)
            if self.mu_k > old_mu and self.verbose:
                print(f"Increased penalty parameter from {old_mu:.3e} to μ = {self.mu_k:.3e}")

        self._last_penalty_check_violation = current_violation
        self.penalty_history.append(self.mu_k)

    def _should_update_multipliers_now(self, violation: Optional[float]) -> bool:
        """Decide whether to update λ based on violation improvement and interval."""
        if violation is None:
            return False

        # Default: always update when gating disabled
        gating_disabled = self.multiplier_improve_ratio <= 0.0 and self.multiplier_min_improve <= 0.0
        if gating_disabled:
            return True

        # First observation, allow an update to establish baseline
        if self._last_multiplier_violation is None:
            return True

        # Respect minimum interval between updates
        if self._multiplier_steps_since_update < self.multiplier_check_interval - 1:
            self._multiplier_steps_since_update += 1
            return False

        ratio_ok = (
            self.multiplier_improve_ratio <= 0.0
            or violation <= self.multiplier_improve_ratio * self._last_multiplier_violation
        )
        abs_ok = (
            self.multiplier_min_improve <= 0.0
            or violation <= self._last_multiplier_violation - self.multiplier_min_improve
        )

        return ratio_ok and abs_ok

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
            va (torch.Tensor): Voltage angles at buses [batch_size, n_bus] in degrees
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
        if self.Y_real is None or self.Y_imag is None:
            return torch.tensor([], device=vm_pred.device, requires_grad=True)

        # Detect actual batch size and number of buses
        if vm_pred.dim() == 1:
            # If 1D, it could be a single sample or flattened batch
            n_bus = self.Y_real.shape[0]
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

        # Convert voltage angles from degrees to radians
        va_rad = torch.deg2rad(va_pred)

        # Compute voltage phasors
        v_real = vm_pred * torch.cos(va_rad)
        v_imag = vm_pred * torch.sin(va_rad)

        # Initialize power injections at buses
        p_inj = torch.zeros_like(v_real)
        q_inj = torch.zeros_like(v_imag)

        def _ensure_batch_dims(p_inj, q_inj, vm_pred, va_pred, pg_pred, qg_pred):
            """Guarantee tensors are 2D for scatter_add and keep batch size aligned."""
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
            return p_inj, q_inj, vm_pred, va_pred, pg_pred, qg_pred, p_inj.size(0)

        p_inj, q_inj, vm_pred, va_pred, pg_pred, qg_pred, batch_size = _ensure_batch_dims(
            p_inj, q_inj, vm_pred, va_pred, pg_pred, qg_pred
        )

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
        if load_bus_indices is not None and pd is not None:
            load_indices_expanded = load_bus_indices.unsqueeze(0).expand(batch_size, -1).to(pd.device)
            p_inj.scatter_add_(1, load_indices_expanded, -pd)
            q_inj.scatter_add_(1, load_indices_expanded, -qd)

        # Compute power flows using admittance matrix
        device = v_real.device
        y_real_batch = self.Y_real.to(device).unsqueeze(0).expand(batch_size, -1, -1)
        y_imag_batch = self.Y_imag.to(device).unsqueeze(0).expand(batch_size, -1, -1)

        # Simplified power flow calculation using linearized approximation for stability
        # This avoids the extremely large values from the full AC power flow equations
        # P_calc ≈ sum_j (G_ij * V_j * cos(theta_j) + B_ij * V_j * sin(theta_j))
        # Q_calc ≈ sum_j (G_ij * V_j * sin(theta_j) - B_ij * V_j * cos(theta_j))

        v_cos = vm_pred * torch.cos(va_rad)  # V * cos(theta)
        v_sin = vm_pred * torch.sin(va_rad)  # V * sin(theta)

        # Using matrix-vector multiplication for efficiency
        # Shape: [batch_size, n_bus]
        p_calc = torch.einsum('bij,bj->bi', y_real_batch, v_cos) + torch.einsum('bij,bj->bi', y_imag_batch, v_sin)
        q_calc = torch.einsum('bij,bj->bi', y_real_batch, v_sin) - torch.einsum('bij,bj->bi', y_imag_batch, v_cos)

        # Power balance constraints: injection - calculated_flow = 0
        p_balance = p_inj - p_calc
        q_balance = q_inj - q_calc

        # Compute RMS of P and Q balance across buses (averaged over batch if present)
        try:
            p_mean = p_balance.mean(dim=0) if p_balance.dim() > 1 else p_balance
            q_mean = q_balance.mean(dim=0) if q_balance.dim() > 1 else q_balance
            p_rms = torch.sqrt(torch.mean(p_mean**2) + 1e-12)
            q_rms = torch.sqrt(torch.mean(q_mean**2) + 1e-12)
            # store as plain Python floats for easy external inspection
            self._last_p_balance_rms = float(p_rms.detach().item())
            self._last_q_balance_rms = float(q_rms.detach().item())
        except Exception:
            # In case of unexpected shapes or device issues, leave as None
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

    def compute_line_flow_constraints(
        self,
        vm: torch.Tensor,
        va: torch.Tensor,
        line_edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute line flow inequality constraints converted to equality with slack variables.

        For line flow limit S_max, we convert to equality with slack:
        ..math::
            |S_{ij}|² ≤ S_{max}² ==> |S_{ij}|² - S_{max}² + s² = 0

        where s ≥ 0 is slack variable

        Args:
            vm (torch.Tensor): Voltage magnitudes at buses [batch_size, n_bus]
            va (torch.Tensor): Voltage angles at buses [batch_size, n_bus] in degrees
            line_edge_index (torch.Tensor): Edge indices for lines [2, n_lines]

        Returns:
            Line flow constraint violations
        """
        if line_edge_index is None or self.line_limits is None or self.line_limits.numel() == 0:
            return torch.tensor([], device=vm.device, requires_grad=True)

        batch_size = vm.size(0) if vm.dim() > 1 else 1
        device = vm.device

        # Handle single sample case
        if vm.dim() == 1:
            vm = vm.unsqueeze(0)
            va = va.unsqueeze(0)

        # Convert angles to radians
        va_rad = torch.deg2rad(va)

        # Compute voltage phasors
        v_real = vm * torch.cos(va_rad)
        v_imag = vm * torch.sin(va_rad)

        line_constraints = []
        device = vm.device

        # For each transmission line
        for k, (i, j) in enumerate(line_edge_index.t()):
            if k >= len(self.line_limits):
                break

            line_violations = []

            for b in range(batch_size):
                # Voltage phasors at both ends
                v_i = v_real[b, i] + 1j * v_imag[b, i]
                v_j = v_real[b, j] + 1j * v_imag[b, j]

                # Line admittance elements (simplified - use diagonal elements)
                y_ii = self.Y_real[i, i].to(device) + 1j * self.Y_imag[i, i].to(device)
                y_ij = self.Y_real[i, j].to(device) + 1j * self.Y_imag[i, j].to(device)

                # Current flow from i to j (simplified)
                i_ij = y_ii * v_i - y_ij * v_j

                # Apparent power flow from i to j
                s_ij = v_i * torch.conj(i_ij)
                s_magnitude_squared = torch.real(s_ij * torch.conj(s_ij))

                # Line constraint: |S|² - S_max² ≤ 0
                # Convert to equality with slack: |S|² - S_max² + slack² = 0
                # For now, we'll use the constraint violation directly
                line_limit_val = float(self.line_limits[k])
                violation_squared = s_magnitude_squared - line_limit_val**2
                # Only consider positive violations; scale to keep magnitudes stable
                line_violations.append(torch.sqrt(torch.relu(violation_squared)))

            line_constraint = torch.stack(line_violations).mean()
            line_constraints.append(line_constraint)

        if line_constraints:
            constraints = torch.stack(line_constraints)
        else:
            constraints = torch.tensor([], device=device, requires_grad=True)

        self._last_line_limit_rms = float(torch.sqrt(torch.mean(constraints**2)).detach().item()) if constraints.numel() > 0 else 0.0

        return constraints

    def compute_constraints(
        self,
        predictions: Dict[str, torch.Tensor],
        data: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute all constraints c(x) = [c_equality; c_inequality].

        Args:
            predictions: Model predictions containing 'bus' and 'generator' tensors
            data: Input data containing load and generator indices

        Returns:
            Constraint vector c(x)
        """
        # Extract predictions (bus order is [va, vm])
        bus_pred = predictions['bus']
        gen_pred = predictions['generator']

        va = bus_pred[..., 0]
        vm = bus_pred[..., 1]
        pg = gen_pred[..., 0]
        qg = gen_pred[..., 1]

        # Extract data
        pd = data.get('pd', None)
        qd = data.get('qd', None)
        gen_bus_indices = data.get('gen_bus_indices', None)
        load_bus_indices = data.get('load_bus_indices', None)
        line_edge_index = data.get('line_edge_index', None)

        # Compute power flow equality constraints
        power_flow_constraints = self.compute_power_flow_constraints(
            vm, va, pg, qg, pd, qd, gen_bus_indices, load_bus_indices
        )

        # Compute line flow inequality constraints
        line_flow_constraints = self.compute_line_flow_constraints(
            vm, va, line_edge_index
        )

        # Combine all constraints
        if power_flow_constraints.numel() > 0 and line_flow_constraints.numel() > 0:
            constraints = torch.cat([power_flow_constraints, line_flow_constraints])
        elif power_flow_constraints.numel() > 0:
            constraints = power_flow_constraints
        elif line_flow_constraints.numel() > 0:
            constraints = line_flow_constraints
        else:
            constraints = torch.tensor([], device=vm.device, requires_grad=True)

        return constraints

    def compute_augmented_lagrangian(
        self,
        mse_loss: torch.Tensor,
        constraints: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute augmented Lagrangian function:
        L_A(x, λ; μ) = f(x) - Σ λ_i c_i(x) + (μ/2) Σ c_i(x)^2

        Args:
            mse_loss: Objective function f(x)
            constraints: Constraint vector c(x)

        Returns:
            Tuple of (augmented_lagrangian_value, components_dict)
        """
        if constraints.numel() == 0:
            # No constraints - return pure objective
            zero = torch.tensor(0.0, device=mse_loss.device)
            self._raw_violation = 0.0
            self._ema_violation = 0.0
            self._latest_constraints = None
            self._latest_constraint_signal = None
            return mse_loss, {
                'objective': mse_loss,
                'lagrange_term': zero,
                'penalty_term': zero,
                'constraint_violation': zero,
                'raw_constraint_violation': zero,
                'ema_constraint_violation': zero,
                'multipliers_active': False
            }

        # Smooth constraints for stability
        constraint_signal = self._constraint_signal_for_loss(constraints)
        self._init_constraint_buffers(constraint_signal)

        # Ensure λ and constraints have compatible dimensions
        if self.lambda_k.size(0) != constraint_signal.size(-1):
            warnings.warn(
                f"Lagrange multiplier dimension mismatch. Expected {constraint_signal.size(-1)}, "
                f"got {self.lambda_k.size(0)}. Reinitializing λ."
            )
            self.lambda_k = torch.zeros(constraint_signal.size(-1), device=constraint_signal.device)

        # Lagrange term: -Σ λ_i c_i(x)
        lagrange_term = torch.tensor(0.0, device=mse_loss.device)
        if self._should_use_multipliers():
            lagrange_term = -torch.dot(self.lambda_k, constraint_signal)

        # Penalty term: (μ/2) Σ c_i(x)^2
        penalty_term = (self.mu_k / 2.0) * torch.sum(constraint_signal**2)

        # Augmented Lagrangian
        augmented_lagrangian = mse_loss + penalty_term + (lagrange_term if self._should_use_multipliers() else 0.0)

        # Track violations (EMA-driven)
        violation_value = self._ema_violation if self._ema_violation is not None else self._raw_violation
        if violation_value is None:
            violation_value = self._compute_violation_norm(constraints)
        self.constraint_history.append(violation_value)
        constraint_violation = torch.tensor(violation_value, device=mse_loss.device)
        raw_violation = torch.tensor(self._raw_violation if self._raw_violation is not None else violation_value,
                                     device=mse_loss.device)
        ema_violation = torch.tensor(self._ema_violation if self._ema_violation is not None else violation_value,
                                     device=mse_loss.device)
        last_multiplier_violation = torch.tensor(
            self._last_multiplier_violation if self._last_multiplier_violation is not None else violation_value,
            device=mse_loss.device
        )
        last_multiplier_norm = torch.tensor(
            self._last_multiplier_norm if self._last_multiplier_norm is not None else 0.0,
            device=mse_loss.device
        )

        components = {
            'objective': mse_loss,
            'lagrange_term': lagrange_term,
            'penalty_term': penalty_term,
            'constraint_violation': constraint_violation,
            'raw_constraint_violation': raw_violation,
            'ema_constraint_violation': ema_violation,
            'p_balance_rmse': self._last_p_balance_rms,
            'q_balance_rmse': self._last_q_balance_rms,
            'line_limit_rmse': self._last_line_limit_rms,
            'multipliers_active': self._should_use_multipliers(),
            'multiplier_updated': self._last_multiplier_updated,
            'last_multiplier_norm': last_multiplier_norm,
            'last_multiplier_violation': last_multiplier_violation
        }

        return augmented_lagrangian, components

    def update_lagrange_multipliers(self, constraints: Optional[torch.Tensor] = None):
        """
        Update Lagrange multipliers using equation (17.39):
        λ^{k+1} = λ^k - μ_k c_i(x_k)

        Args:
            constraints: Current constraint vector c(x_k)
        """
        if not self._should_use_multipliers():
            self._last_multiplier_updated = False
            return

        # Default to the latest EMA signal if explicit constraints are not provided
        if constraints is None or constraints.numel() == 0:
            constraint_signal = self.constraint_ema
        else:
            self._init_constraint_buffers(constraints)
            constraint_signal = self.constraint_ema

        if constraint_signal is None or constraint_signal.numel() == 0:
            self._last_multiplier_updated = False
            return

        violation_val = self._ema_violation if self._ema_violation is not None else self._compute_violation_norm(constraint_signal)
        should_update = self._should_update_multipliers_now(violation_val)
        if not should_update:
            self._last_multiplier_updated = False
            return

        self._multiplier_steps_since_update = 0

        # Update multipliers with clipping to prevent explosive growth
        update = self.mu_k * constraint_signal.detach()
        self.lambda_k = self.lambda_k - update

        # Clip multipliers to reasonable range to prevent divergence
        if self.multiplier_clip is not None:
            self.lambda_k = torch.clamp(self.lambda_k, -self.multiplier_clip, self.multiplier_clip)
        # print(f"Updated multiplers: {torch.norm(self.lambda_k).item()}, violation: {violation_val}")

        self._last_multiplier_norm = torch.norm(self.lambda_k).item()
        self._last_multiplier_violation = violation_val
        self._last_multiplier_updated = True

        # Store history
        self.lagrange_history.append(self.lambda_k.clone().cpu().numpy())

    def update_penalty_parameter(self, constraint_violation: float, prev_violation: float = None):
        """
        Update penalty parameter μ if constraint violation is not decreasing sufficiently.

        Args:
            constraint_violation: Current constraint violation norm
            prev_violation: Previous constraint violation norm
        """
        # Allow manual/legacy updates by forcing an immediate penalty check
        self._epochs_since_penalty_check = 0
        if prev_violation is not None:
            self._last_penalty_check_violation = prev_violation
        self._maybe_update_penalty(constraint_violation, force=True)

    def check_convergence(self, constraint_violation: float) -> bool:
        """
        Check convergence based on constraint satisfaction.

        Args:
            constraint_violation: Current constraint violation norm
        """
        converged = constraint_violation < self.constraint_tolerance

        if converged:
            print(f"Converged! Constraint violation: {constraint_violation:.2e}")

        return converged

    def forward(
        self,
        mse_loss: torch.Tensor,
        predictions: Dict[str, torch.Tensor],
        data: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute augmented Lagrangian loss for current iteration.

        Args:
            mse_loss: MSE objective function
            predictions: Model predictions
            data: Input data for constraint computation

        Returns:
            Tuple of (augmented_lagrangian_loss, info_dict)
        """
        # Compute constraints
        constraints = self.compute_constraints(predictions, data)

        # Compute augmented Lagrangian
        aug_lag_loss, components = self.compute_augmented_lagrangian(mse_loss, constraints)

        # Return loss and information
        info = {
            **components,
            'penalty_parameter': self.mu_k,
            'n_constraints': constraints.numel(),
            'constraints': constraints.detach(),
            'multipliers_active': self._should_use_multipliers(),
        }

        return aug_lag_loss, info

    def step_outer_iteration(self, constraint_violation: float):
        """
        Perform outer iteration step: update λ and μ.

        Args:
            constraint_violation: Current constraint violation norm
        """
        # Update penalty parameter if needed
        self.update_penalty_parameter(constraint_violation, prev_violation=None)

        # Update Lagrange multipliers (will be done after constraint computation)
        self.outer_iteration += 1

        print(f"Outer iteration {self.outer_iteration}: "
              f"constraint_violation={constraint_violation:.2e}, μ={self.mu_k:.2e}")

    def step_epoch(self):
        """Advance epoch counter and run scheduled penalty updates."""
        current_violation = self._ema_violation if self._ema_violation is not None else self._raw_violation
        self._maybe_update_penalty(current_violation)
        self.current_epoch += 1

    def reset_for_new_problem(self):
        """Reset algorithm state for a new optimization problem."""
        self.mu_k = float(np.clip(self.mu_0, self.min_penalty, self.max_penalty))
        self.lambda_k = None
        self.outer_iteration = 0
        self.constraint_history = []
        self.lagrange_history = []
        self.penalty_history = []
        self.constraint_ema = None
        self._latest_constraints = None
        self._latest_constraint_signal = None
        self._raw_violation = None
        self._ema_violation = None
        self.current_epoch = 0
        self._epochs_since_penalty_check = 0
        self._last_penalty_check_violation = None
        self._last_multiplier_violation = None
        self._multiplier_steps_since_update = 0
        self._last_multiplier_updated = False
