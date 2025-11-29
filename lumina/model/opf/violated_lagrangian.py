"""
Violated (Violation-based) Lagrangian Method for ACOPF Neural Network Training

This module implements a violation-based Lagrangian approach where the loss function
explicitly penalizes constraint violations using Lagrangian multipliers. Unlike the
augmented Lagrangian which uses quadratic penalties, this approach uses linear
penalties proportional to the violations:

    Loss = MSE + Σ λ_i * |violation_i|

The multipliers are updated periodically based on accumulated violations.

Copyright (c) 2025, Argonne National Laboratory
All rights reserved.
"""

from copy import deepcopy
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from pypower import idx_brch, idx_bus, idx_gen
from pypower.api import ext2int, loadcase, makeYbus


class ViolatedLagrangianACOPF(nn.Module):
    """
    Violation-based Lagrangian for ACOPF constraint enforcement.

    This implementation computes violations of:
    - Equality constraints: Power balance (active and reactive)
    - Inequality constraints: Line thermal limits, voltage angle differences

    The loss combines MSE with weighted constraint violations where weights
    are Lagrangian multipliers that adapt during training.
    """

    def __init__(
        self,
        grid_data: str,
        device: torch.device,
        rho_init: float = 0.001,
        s_init: float = 0.001,
        warmup_iter: int = 10,
        update_frequency: int = 5,
        use_angle_diff: bool = True,
        verbose: bool = False
    ):
        """
        Initialize Violated Lagrangian for ACOPF.

        Args:
            grid_data: Path to power system case file (PyPower format)
            device: PyTorch device (cpu/cuda)
            rho_init: Initial value for inequality multiplier updates
            s_init: Initial value for equality multiplier updates
            warmup_iter: Number of epochs before starting multiplier updates
            update_frequency: Update multipliers every N epochs
            use_angle_diff: Whether to include angle difference constraints
            verbose: Enable verbose logging
        """
        super().__init__()

        # Load power system data
        ppc = loadcase(grid_data)
        ppc = ext2int(ppc)

        self.ppc = ppc
        self.device = device
        self.verbose = verbose

        # System dimensions
        self.nbus = ppc['bus'].shape[0]
        self.ng = ppc['gen'].shape[0]
        self.nl = ppc['branch'].shape[0]
        self.baseMVA = ppc['baseMVA']
        self.genbase = ppc['baseMVA']

        # Bus type indices
        self.slack = np.where(ppc['bus'][:, idx_bus.BUS_TYPE] == 3)[0]
        self.pv = np.where(ppc['bus'][:, idx_bus.BUS_TYPE] == 2)[0]
        self.spv = np.concatenate([self.slack, self.pv])
        self.spv.sort()
        self.pq = np.setdiff1d(range(self.nbus), self.spv)
        self.nonslack_idxes = np.sort(np.concatenate([self.pq, self.pv]))

        # Generator bus mapping
        self.gen_idx = ppc['gen'][:, idx_gen.GEN_BUS]
        self.slack_ = np.array([np.where(x == self.spv)[0][0] for x in self.slack])
        self.pv_ = np.array([np.where(x == self.spv)[0][0] for x in self.pv])

        self.nslack = len(self.slack)
        self.npv = len(self.pv)

        # Cost function parameters
        self.quad_costs = torch.tensor(
            ppc['gencost'][:, 4], dtype=torch.float32, device=device
        )
        self.lin_costs = torch.tensor(
            ppc['gencost'][:, 5], dtype=torch.float32, device=device
        )
        self.const_cost = ppc['gencost'][:, 6].sum()

        # Generator limits
        self.pmax = torch.tensor(
            ppc['gen'][:, idx_gen.PMAX] / self.genbase,
            dtype=torch.float32, device=device
        )
        self.pmin = torch.tensor(
            ppc['gen'][:, idx_gen.PMIN] / self.genbase,
            dtype=torch.float32, device=device
        )
        self.qmax = torch.tensor(
            ppc['gen'][:, idx_gen.QMAX] / self.genbase,
            dtype=torch.float32, device=device
        )
        self.qmin = torch.tensor(
            ppc['gen'][:, idx_gen.QMIN] / self.genbase,
            dtype=torch.float32, device=device
        )

        # Voltage limits
        self.vmax = torch.tensor(
            ppc['bus'][:, idx_bus.VMAX], dtype=torch.float32, device=device
        )
        self.vmin = torch.tensor(
            ppc['bus'][:, idx_bus.VMIN], dtype=torch.float32, device=device
        )
        self.slackva = torch.tensor(
            [np.deg2rad(ppc['bus'][self.slack, idx_bus.VA])],
            dtype=torch.float32
        ).squeeze(-1)

        # Line flow limits
        flow_max = (ppc['branch'][:, idx_brch.RATE_A] / self.baseMVA) ** 2
        flow_max[flow_max == 0] = np.inf
        self.line_limit = torch.tensor(flow_max, dtype=torch.float32, device=device)

        # Angle difference limits
        self.angmin = torch.tensor(
            np.deg2rad(ppc['branch'][:, idx_brch.ANGMIN]),
            dtype=torch.float32, device=device
        )
        self.angmax = torch.tensor(
            np.deg2rad(ppc['branch'][:, idx_brch.ANGMAX]),
            dtype=torch.float32, device=device
        )
        self.br_idx = torch.tensor(
            ppc['branch'][:, [idx_brch.F_BUS, idx_brch.T_BUS]], dtype=torch.long
        )

        # Admittance matrices
        ppc2 = deepcopy(ppc)
        Ybus, Yf, Yt = makeYbus(self.baseMVA, ppc2['bus'], ppc2['branch'])
        Ybus = Ybus.todense()
        self.Ybusr = torch.tensor(np.real(Ybus), dtype=torch.float32, device=device)
        self.Ybusi = torch.tensor(np.imag(Ybus), dtype=torch.float32, device=device)
        self.Yf = Yf
        self.Yt = Yt

        # Lagrangian parameters
        self.rho = rho_init
        self.s = s_init
        self.warmup_iter = warmup_iter
        self.update_frequency = update_frequency
        self.use_angle_diff = use_angle_diff
        self.current_epoch = 0

        # Initialize Lagrangian multipliers
        # Inequality multipliers
        if use_angle_diff:
            self.ineq_lagm_ang = torch.ones(1, 2 * self.nl, device=device)
        else:
            self.ineq_lagm_ang = None
        self.ineq_lagm_line = torch.ones(1, 2 * self.nl, device=device)

        # Equality multipliers (power balance)
        self.eq_lagm_active = torch.ones(1, self.nbus, device=device)
        self.eq_lagm_reactive = torch.ones(1, self.nbus, device=device)

        # History tracking
        self.constraint_history = []
        self.multiplier_history = []

    def objective_function(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute generation cost objective.

        Args:
            predictions: Dictionary with 'generator' key containing [Pg, Qg]

        Returns:
            Cost in per-unit (normalized by genbase^2)
        """
        batch_size = int(predictions['bus'].shape[0] / self.nbus)
        pg = torch.zeros((batch_size, self.ng), device=self.device)

        for b in range(batch_size):
            pg[b, :] = predictions['generator'][b * self.ng:(b + 1) * self.ng, 0]

        pg_mw = pg * self.genbase
        cost = (self.quad_costs * pg_mw ** 2).sum(axis=1) + \
            (self.lin_costs * pg_mw).sum(axis=1) + \
            self.const_cost

        return cost / (self.genbase ** 2)

    def compute_equality_residuals(
        self,
        inputs: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute power balance equality constraint residuals.

        Args:
            inputs: Dictionary with 'load' key containing load data
            predictions: Dictionary with 'bus' and 'generator' predictions

        Returns:
            Tuple of (residuals, predicted_pd, predicted_qd, pd, qd)
        """
        batch_size = int(predictions['bus'].shape[0] / self.nbus)

        vm = torch.zeros((batch_size, self.nbus), device=self.device)
        va = torch.zeros((batch_size, self.nbus), device=self.device)
        pg = torch.zeros((batch_size, self.ng), device=self.device)
        qg = torch.zeros((batch_size, self.ng), device=self.device)
        pd = torch.zeros((batch_size, self.nbus), device=self.device)
        qd = torch.zeros((batch_size, self.nbus), device=self.device)

        # Extract predictions
        for b in range(batch_size):
            vm[b, :] = predictions['bus'][b * self.nbus:(b + 1) * self.nbus, 1]
            va[b, :] = predictions['bus'][b * self.nbus:(b + 1) * self.nbus, 0]
            pg[b, :] = predictions['generator'][b * self.ng:(b + 1) * self.ng, 0]
            qg[b, :] = predictions['generator'][b * self.ng:(b + 1) * self.ng, 1]

        # Aggregate generation at buses (handle multiple gens per bus)
        if self.ng != self.spv.shape[0]:
            spv_pg = np.zeros((batch_size, self.spv.shape[0]))
            spv_qg = np.zeros((batch_size, self.spv.shape[0]))

            unique, inverse = np.unique(self.gen_idx, return_inverse=True)
            for b in range(batch_size):
                pg_cpu = pg[b, :].detach().cpu().numpy()
                qg_cpu = qg[b, :].detach().cpu().numpy()
                np.add.at(spv_pg[b, :], inverse, pg_cpu)
                np.add.at(spv_qg[b, :], inverse, qg_cpu)

            spv_pg_ = torch.tensor(spv_pg, device=self.device, dtype=torch.float32)
            spv_qg_ = torch.tensor(spv_qg, device=self.device, dtype=torch.float32)
        else:
            spv_pg_ = pg
            spv_qg_ = qg

        # Extract load data
        load_num = int(inputs['load'].shape[0] / batch_size)
        for b in range(batch_size):
            pd[b, self.ppc['bus'][:, 2] != 0] = inputs['load'][
                b * load_num:(b + 1) * load_num, 0
            ]
            qd[b, self.ppc['bus'][:, 2] != 0] = inputs['load'][
                b * load_num:(b + 1) * load_num, 1
            ]

        # Compute power injections
        vr = vm * torch.cos(va)
        vi = vm * torch.sin(va)

        tmp1 = vr @ self.Ybusr - vi @ self.Ybusi
        tmp2 = -vr @ self.Ybusi - vi @ self.Ybusr

        # Real power balance
        pg_expand = torch.zeros(batch_size, self.nbus, device=self.device)
        pg_expand[:, self.spv] = spv_pg_
        real_resid = (vr * tmp1 - vi * tmp2) - (pg_expand - pd)
        predicted_pd = pg_expand - (vr * tmp1 - vi * tmp2)

        # Reactive power balance
        qg_expand = torch.zeros(batch_size, self.nbus, device=self.device)
        qg_expand[:, self.spv] = spv_qg_
        react_resid = (vr * tmp2 + vi * tmp1) - (qg_expand - qd)
        predicted_qd = qg_expand - (vr * tmp2 + vi * tmp1)

        # Concatenate residuals
        resids = torch.cat([real_resid, react_resid], dim=1)

        return resids, predicted_pd, predicted_qd, pd, qd

    def compute_inequality_residuals(
        self,
        predictions: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute inequality constraint residuals (line limits, angle differences).

        Args:
            predictions: Dictionary with 'bus' predictions

        Returns:
            Tensor of inequality residuals (positive means violation)
        """
        batch_size = int(predictions['bus'].shape[0] / self.nbus)

        vm = torch.zeros((batch_size, self.nbus), device=self.device)
        va = torch.zeros((batch_size, self.nbus), device=self.device)

        for b in range(batch_size):
            vm[b, :] = predictions['bus'][b * self.nbus:(b + 1) * self.nbus, 1]
            va[b, :] = predictions['bus'][b * self.nbus:(b + 1) * self.nbus, 0]

        # Line thermal limit violations
        vr = vm * torch.cos(va)
        vi = vm * torch.sin(va)
        vz = torch.complex(vr, vi)

        Yf_dense = torch.tensor(self.Yf.todense(), dtype=torch.complex64, device=self.device)
        Yt_dense = torch.tensor(self.Yt.todense(), dtype=torch.complex64, device=self.device)

        If = Yf_dense @ vz.T
        It = Yt_dense @ vz.T

        Sf = vz[:, self.ppc['branch'][:, 0].astype(int)] * torch.conj(If.T)
        St = vz[:, self.ppc['branch'][:, 1].astype(int)] * torch.conj(It.T)

        Sff = Sf * torch.conj(Sf)
        Stt = St * torch.conj(St)

        # Voltage angle difference violations
        i_idx = self.br_idx[:, 0].to(dtype=torch.long, device=self.device)
        j_idx = self.br_idx[:, 1].to(dtype=torch.long, device=self.device)
        va_ang_diff = va[:, i_idx] - va[:, j_idx]

        # Collect all inequality residuals
        residual_list = []

        if self.use_angle_diff:
            residual_list.extend([
                va_ang_diff - self.angmax,
                self.angmin - va_ang_diff,
            ])

        residual_list.extend([
            Sff.real - self.line_limit,
            Stt.real - self.line_limit,
        ])

        resids = torch.cat(residual_list, dim=1)
        return resids

    def compute_inequality_violations(
        self,
        predictions: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute inequality violations (clamped to positive values).

        Args:
            predictions: Dictionary with predictions

        Returns:
            Tensor of violations (0 if satisfied, positive if violated)
        """
        resids = self.compute_inequality_residuals(predictions)
        return torch.clamp(resids, 0)

    def forward(
        self,
        mse_loss: torch.Tensor,
        predictions: Dict[str, torch.Tensor],
        inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute violated Lagrangian loss.

        Args:
            mse_loss: Base MSE loss
            predictions: Model predictions
            inputs: Input data (including loads)

        Returns:
            Tuple of (total_loss, info_dict)
        """
        # Compute constraint violations
        eq_resid, predicted_pd, predicted_qd, pd, qd = self.compute_equality_residuals(
            inputs, predictions
        )
        ineq_dist = self.compute_inequality_violations(predictions)

        # Split inequality violations
        if self.use_angle_diff:
            ineq_ang_diff = ineq_dist[:, :2 * self.nl]
            ineq_line = ineq_dist[:, 2 * self.nl:]
        else:
            ineq_ang_diff = None
            ineq_line = ineq_dist

        # Split equality residuals (use absolute value)
        eq_active = eq_resid[:, :self.nbus].abs()
        eq_reactive = eq_resid[:, self.nbus:].abs()

        # Compute weighted constraint costs
        if self.use_angle_diff:
            ineq_cost = torch.cat([
                self.ineq_lagm_ang * ineq_ang_diff,
                self.ineq_lagm_line * ineq_line
            ], dim=1)
        else:
            ineq_cost = self.ineq_lagm_line * ineq_line

        eq_cost = torch.cat([
            self.eq_lagm_active * eq_active,
            self.eq_lagm_reactive * eq_reactive
        ], dim=1)

        # Normalize by number of constraints
        ineq_loss = (ineq_cost.sum(dim=1) / ineq_cost.shape[1]).mean()
        eq_loss = (eq_cost.sum(dim=1) / eq_cost.shape[1]).mean()
        constraint_loss = ineq_loss + eq_loss

        # Total loss
        total_loss = mse_loss + constraint_loss

        # Compute objective cost for monitoring
        obj_cost = self.objective_function(predictions).mean()

        # Compute total constraint violation for monitoring
        total_violation = (ineq_dist.sum() + eq_resid.abs().sum()) / (
            ineq_dist.numel() + eq_resid.numel()
        )

        info = {
            'total_loss': total_loss.item(),
            'mse_loss': mse_loss.item(),
            'constraint_loss': constraint_loss.item(),
            'ineq_loss': ineq_loss.item(),
            'eq_loss': eq_loss.item(),
            'obj_cost': obj_cost.item(),
            'total_violation': total_violation.item(),
        }

        return total_loss, info

    def update_multipliers(
        self,
        model: nn.Module,
        dataloader,
    ):
        """
        Update Lagrangian multipliers based on constraint violations.

        Args:
            model: Neural network model
            dataloader: Training dataloader
        """
        if self.current_epoch < self.warmup_iter:
            if self.verbose:
                print(f"Epoch {self.current_epoch}: Skipping multiplier update (warmup)")
            return

        if (self.current_epoch - self.warmup_iter) % self.update_frequency != 0:
            if self.verbose:
                print(f"Epoch {self.current_epoch}: Skipping multiplier update (frequency)")
            return

        # Accumulate violations
        total_ineq_ang = 0.0 if self.use_angle_diff else None
        total_ineq_line = 0.0
        total_eq_active = 0.0
        total_eq_reactive = 0.0
        num_samples = 0

        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                predictions = model(batch.x_dict, batch.edge_index_dict)

                # Compute violations
                eq_resid, _, _, _, _ = self.compute_equality_residuals(
                    batch.x_dict, predictions
                )
                ineq_dist = self.compute_inequality_violations(predictions)

                # Accumulate
                if self.use_angle_diff:
                    ineq_ang_diff = ineq_dist[:, :2 * self.nl]
                    ineq_line = ineq_dist[:, 2 * self.nl:]
                    total_ineq_ang += ineq_ang_diff.sum(dim=0)
                else:
                    ineq_line = ineq_dist

                total_ineq_line += ineq_line.sum(dim=0)

                eq_active = eq_resid[:, :self.nbus].abs()
                eq_reactive = eq_resid[:, self.nbus:].abs()

                total_eq_active += eq_active.sum(dim=0)
                total_eq_reactive += eq_reactive.sum(dim=0)

                num_samples += eq_resid.shape[0]

        # Average violations
        if self.use_angle_diff:
            avg_ineq_ang = (total_ineq_ang / num_samples).unsqueeze(0)
        avg_ineq_line = (total_ineq_line / num_samples).unsqueeze(0)
        avg_eq_active = (total_eq_active / num_samples).unsqueeze(0)
        avg_eq_reactive = (total_eq_reactive / num_samples).unsqueeze(0)

        # Update multipliers
        if self.use_angle_diff:
            self.ineq_lagm_ang = self.ineq_lagm_ang + self.rho * avg_ineq_ang
            self.ineq_lagm_ang = torch.clamp(self.ineq_lagm_ang, 0, 100)

        self.ineq_lagm_line = self.ineq_lagm_line + self.rho * avg_ineq_line
        self.ineq_lagm_line = torch.clamp(self.ineq_lagm_line, 0, 100)

        self.eq_lagm_active = self.eq_lagm_active + self.s * avg_eq_active
        self.eq_lagm_active = torch.clamp(self.eq_lagm_active, 0, 100)

        self.eq_lagm_reactive = self.eq_lagm_reactive + self.s * avg_eq_reactive
        self.eq_lagm_reactive = torch.clamp(self.eq_lagm_reactive, 0, 100)

        if self.verbose:
            print(f"Epoch {self.current_epoch}: Updated Lagrangian multipliers")
            print(f"  Avg ineq_line mult: {self.ineq_lagm_line.mean():.6f}")
            print(f"  Avg eq_active mult: {self.eq_lagm_active.mean():.6f}")
            print(f"  Avg eq_reactive mult: {self.eq_lagm_reactive.mean():.6f}")

    def step_epoch(self):
        """Increment epoch counter."""
        self.current_epoch += 1
