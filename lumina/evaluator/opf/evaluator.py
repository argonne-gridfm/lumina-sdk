"""
Comprehensive Constraint Violation Evaluator for ACOPF Problems

This module provides a unified interface to evaluate all types of constraint violations
in AC Optimal Power Flow problems, including:
- Bound constraints (voltage magnitude, voltage angle, generation limits)
- Power flow constraints (active and reactive power balance)
- Line flow constraints (thermal limits)

Copyright (c) 2025, Argonne National Laboratory
All rights reserved.
"""

import torch
import torch.nn as nn
import warnings
from typing import Dict, List, Optional, Tuple, Union


class ACOPFConstraintEvaluator(nn.Module):
    """Comprehensive constraint violation evaluator for ACOPF problems.

    Evaluates bound, power-flow, and line-flow constraint violations given
    model predictions and network parameters (admittance matrix, limits).

    Supported constraint categories:
        - Bound constraints: voltage magnitude/angle, active/reactive generation.
        - Power flow constraints: active and reactive power balance at each bus.
        - Line flow constraints: thermal limits on transmission lines.

    Args:
        voltage_limits (dict[str, torch.Tensor], optional): Dict with ``'vmin'``,
            ``'vmax'`` tensors of shape ``[n_bus]``.
        generation_limits (dict[str, torch.Tensor], optional): Dict with
            ``'pmin'``, ``'pmax'``, ``'qmin'``, ``'qmax'`` tensors of
            shape ``[n_gen]``.
        line_limits (torch.Tensor, optional): Line flow limits ``[n_lines]``.
        Y_real (torch.Tensor, optional): Real part of admittance matrix
            ``[n_bus, n_bus]``.
        Y_imag (torch.Tensor, optional): Imaginary part of admittance matrix
            ``[n_bus, n_bus]``.
        edge_index (torch.Tensor, optional): Line connectivity ``[2, n_lines]``.
        base_mva (float): Base MVA for power scaling.
        slack_bus_indices (list[int], optional): Indices of slack buses.
        device (torch.device, optional): Computation device.
    """

    def __init__(
        self,
        voltage_limits: Optional[Dict[str, torch.Tensor]] = None,
        generation_limits: Optional[Dict[str, torch.Tensor]] = None,
        line_limits: Optional[torch.Tensor] = None,
        Y_real: Optional[torch.Tensor] = None,
        Y_imag: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        base_mva: float = 100.0,
        slack_bus_indices: Optional[List[int]] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the constraint evaluator.

        Args:
            voltage_limits: Dict with 'vmin', 'vmax' voltage limits [n_bus]
            generation_limits: Dict with 'pmin', 'pmax', 'qmin', 'qmax' [n_gen]
            line_limits: Line flow limits [n_lines]
            Y_real: Real part of admittance matrix [n_bus, n_bus]
            Y_imag: Imaginary part of admittance matrix [n_bus, n_bus]
            edge_index: Line connectivity [2, n_lines]
            base_mva: Base MVA for power scaling
            device: Computation device
        """
        super().__init__()

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_mva = base_mva

        # Store network parameters
        self.voltage_limits = voltage_limits
        self.generation_limits = generation_limits
        self.line_limits = line_limits
        self.Y_real = Y_real
        self.Y_imag = Y_imag
        self.edge_index = edge_index
        self.slack_bus_indices = slack_bus_indices or [0]
        self.total_real_power_demand = 0.0

        # Move tensors to device if provided
        self._move_to_device()

    def _move_to_device(self):
        """Move all tensors to the specified device."""
        if self.voltage_limits:
            for key, tensor in self.voltage_limits.items():
                if tensor is not None:
                    self.voltage_limits[key] = tensor.to(self.device)

        if self.generation_limits:
            for key, tensor in self.generation_limits.items():
                if tensor is not None:
                    self.generation_limits[key] = tensor.to(self.device)

        if self.line_limits is not None:
            self.line_limits = self.line_limits.to(self.device)

        if self.Y_real is not None:
            self.Y_real = self.Y_real.to(self.device)

        if self.Y_imag is not None:
            self.Y_imag = self.Y_imag.to(self.device)

        if self.edge_index is not None:
            self.edge_index = self.edge_index.to(self.device)

    def _group_node_predictions(self, node_pred: torch.Tensor, node_batch: torch.Tensor) -> torch.Tensor:
        """Group node-level predictions into per-sample tensors.

        Args:
            node_pred: Tensor of shape [total_nodes, feat]
            node_batch: LongTensor of shape [total_nodes] with sample indices

        Returns:
            grouped: Tensor of shape [batch_size, n_nodes_per_sample, feat]
        """
        # Ensure on same device
        node_pred = node_pred.to(self.device)
        node_batch = node_batch.to(self.device)

        if node_batch.numel() == 0:
            return node_pred.unsqueeze(0)

        batch_size = int(node_batch.max().item()) + 1
        counts = torch.bincount(node_batch.cpu())
        # If number of nodes per sample varies, use the max and pad as needed
        n_nodes = int(counts.max().item())

        feat = node_pred.size(1) if node_pred.dim() > 1 else 1
        grouped = node_pred.new_zeros((batch_size, n_nodes, feat))

        for i in range(batch_size):
            mask = (node_batch == i)
            grp = node_pred[mask]
            if grp.size(0) == 0:
                continue
            if grp.size(0) < n_nodes:
                pad = node_pred.new_zeros((n_nodes - grp.size(0), feat))
                grp = torch.cat([grp, pad], dim=0)
            elif grp.size(0) > n_nodes:
                grp = grp[:n_nodes]
            grouped[i] = grp

        return grouped

    def set_network_parameters(
        self,
        voltage_limits: Optional[Dict[str, torch.Tensor]] = None,
        generation_limits: Optional[Dict[str, torch.Tensor]] = None,
        line_limits: Optional[torch.Tensor] = None,
        Y_real: Optional[torch.Tensor] = None,
        Y_imag: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None
    ):
        """Update network parameters for constraint evaluation.

        Any argument that is not ``None`` replaces the corresponding stored
        parameter.  All tensors are moved to the evaluator's device after
        assignment.

        Args:
            voltage_limits (dict[str, torch.Tensor], optional): Updated voltage
                limits.
            generation_limits (dict[str, torch.Tensor], optional): Updated
                generation limits.
            line_limits (torch.Tensor, optional): Updated line flow limits.
            Y_real (torch.Tensor, optional): Updated real admittance matrix.
            Y_imag (torch.Tensor, optional): Updated imaginary admittance matrix.
            edge_index (torch.Tensor, optional): Updated edge connectivity.
        """
        if voltage_limits is not None:
            self.voltage_limits = voltage_limits
        if generation_limits is not None:
            self.generation_limits = generation_limits
        if line_limits is not None:
            self.line_limits = line_limits
        if Y_real is not None:
            self.Y_real = Y_real
        if Y_imag is not None:
            self.Y_imag = Y_imag
        if edge_index is not None:
            self.edge_index = edge_index

        self._move_to_device()

    def evaluate_bound_constraints(
        self,
        predictions: Dict[str, torch.Tensor],
        batch_data=None,
        return_individual: bool = True
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Evaluate bound constraint violations for voltage and generation.

        Computes ReLU-based violation magnitudes for voltage magnitude
        bounds, active power generation bounds, and reactive power generation
        bounds.

        Args:
            predictions (dict[str, torch.Tensor]): Model predictions with
                ``'bus'`` ``[total_bus, 2]`` (VA, VM) and ``'generator'``
                ``[total_gen, 2]`` (PG, QG) keys.
            batch_data: Batch data (unused, reserved for future extensions).
            return_individual (bool): If ``True``, include per-direction
                violation components (e.g. ``vm_low_violations``,
                ``vm_high_violations``).

        Returns:
            dict[str, torch.Tensor]: Violation metrics including
                ``'total_bound_violations'`` and optionally individual
                component violations.
        """
        violations = {}

        # Extract predictions (concatenated over the batch: [total_nodes, feat])
        # Model outputs bus as [VA, VM], generator as [PG, QG]
        bus_pred = predictions['bus']  # [total_bus_nodes, 2] -> [VA, VM]
        gen_pred = predictions['generator']  # [total_gen_nodes, 2] -> [PG, QG]

        # Voltage magnitude and angle violations
        if self.voltage_limits is not None:
            vm = bus_pred[..., 1]  # Voltage magnitude (concatenated)

            if 'vmin' in self.voltage_limits and 'vmax' in self.voltage_limits:
                vmin = self.voltage_limits['vmin']
                vmax = self.voltage_limits['vmax']

                # If vmin/vmax are per-case (n_bus), expand to concatenated per-batch vector
                try:
                    n_bus = int(vmin.numel())
                    if vm.numel() != n_bus:
                        # If vm length is a multiple of n_bus, repeat per sample
                        if vm.numel() % n_bus == 0:
                            warnings.warn(
                                "Repeated voltage limits across samples because bus vector length does not match base limit length. "
                                "If this batch contains variable-size cases, VM bounds may be assigned to wrong buses."
                            )
                            batch_size = vm.numel() // n_bus
                            vmin_cat = vmin.to(self.device).repeat(batch_size)
                            vmax_cat = vmax.to(self.device).repeat(batch_size)
                        else:
                            # Fallback: broadcast where possible
                            warnings.warn(
                                "Broadcasted voltage limits by repetition/truncation because bus vector length is not a multiple "
                                "of base limit length; this can hide mixed-topology misalignment."
                            )
                            vmin_cat = vmin.to(self.device).reshape(-1).repeat(vm.numel() //
                                                                               vmin.numel() + 1)[:vm.numel()]
                            vmax_cat = vmax.to(self.device).reshape(-1).repeat(vm.numel() //
                                                                               vmax.numel() + 1)[:vm.numel()]
                    else:
                        vmin_cat = vmin.to(self.device)
                        vmax_cat = vmax.to(self.device)
                except Exception:
                    vmin_cat = vmin.to(self.device)
                    vmax_cat = vmax.to(self.device)

                # Voltage magnitude violations
                vm_low_viol = torch.relu(vmin_cat - vm)
                vm_high_viol = torch.relu(vm - vmax_cat)
                vm_violations = vm_low_viol + vm_high_viol

                violations['voltage_magnitude'] = vm_violations.mean()

                if return_individual:
                    violations['vm_low_violations'] = vm_low_viol.mean()
                    violations['vm_high_violations'] = vm_high_viol.mean()

        # Generation limit violations
        if self.generation_limits is not None:
            pg = gen_pred[..., 0]  # Active power generation (concatenated)
            qg = gen_pred[..., 1]  # Reactive power generation (concatenated)

            # Active power violations
            if 'pmin' in self.generation_limits and 'pmax' in self.generation_limits:
                pmin = self.generation_limits['pmin']
                pmax = self.generation_limits['pmax']

                try:
                    n_gen = int(pmin.numel())
                    if pg.numel() != n_gen:
                        # Repeat per sample if sizes align
                        if pg.numel() % n_gen == 0:
                            warnings.warn(
                                "Repeated active-power limits across samples because generator vector length does not match base limit length. "
                                "If this batch contains variable-size cases, PG bounds may be assigned to wrong generators."
                            )
                            batch_size_g = pg.numel() // n_gen
                            pmin_cat = pmin.to(self.device).repeat(batch_size_g)
                            pmax_cat = pmax.to(self.device).repeat(batch_size_g)
                        else:
                            warnings.warn(
                                "Broadcasted active-power limits by repetition/truncation because generator vector length is not a multiple "
                                "of base limit length; this can hide mixed-topology misalignment."
                            )
                            pmin_cat = pmin.to(self.device).reshape(-1).repeat(pg.numel() //
                                                                               pmin.numel() + 1)[:pg.numel()]
                            pmax_cat = pmax.to(self.device).reshape(-1).repeat(pg.numel() //
                                                                               pmax.numel() + 1)[:pg.numel()]
                    else:
                        pmin_cat = pmin.to(self.device)
                        pmax_cat = pmax.to(self.device)
                except Exception:
                    pmin_cat = pmin.to(self.device)
                    pmax_cat = pmax.to(self.device)

                pg_low_viol = torch.relu(pmin_cat - pg)
                pg_high_viol = torch.relu(pg - pmax_cat)
                pg_violations = pg_low_viol + pg_high_viol

                violations['active_power_generation'] = pg_violations.mean()

                if return_individual:
                    violations['pg_low_violations'] = pg_low_viol.mean()
                    violations['pg_high_violations'] = pg_high_viol.mean()

            # Reactive power violations
            if 'qmin' in self.generation_limits and 'qmax' in self.generation_limits:
                qmin = self.generation_limits['qmin']
                qmax = self.generation_limits['qmax']

                try:
                    n_genq = int(qmin.numel())
                    if qg.numel() != n_genq:
                        if qg.numel() % n_genq == 0:
                            warnings.warn(
                                "Repeated reactive-power limits across samples because generator vector length does not match base limit length. "
                                "If this batch contains variable-size cases, QG bounds may be assigned to wrong generators."
                            )
                            batch_size_q = qg.numel() // n_genq
                            qmin_cat = qmin.to(self.device).repeat(batch_size_q)
                            qmax_cat = qmax.to(self.device).repeat(batch_size_q)
                        else:
                            warnings.warn(
                                "Broadcasted reactive-power limits by repetition/truncation because generator vector length is not a multiple "
                                "of base limit length; this can hide mixed-topology misalignment."
                            )
                            qmin_cat = qmin.to(self.device).reshape(-1).repeat(qg.numel() //
                                                                               qmin.numel() + 1)[:qg.numel()]
                            qmax_cat = qmax.to(self.device).reshape(-1).repeat(qg.numel() //
                                                                               qmax.numel() + 1)[:qg.numel()]
                    else:
                        qmin_cat = qmin.to(self.device)
                        qmax_cat = qmax.to(self.device)
                except Exception:
                    qmin_cat = qmin.to(self.device)
                    qmax_cat = qmax.to(self.device)

                qg_low_viol = torch.relu(qmin_cat - qg)
                qg_high_viol = torch.relu(qg - qmax_cat)
                qg_violations = qg_low_viol + qg_high_viol

                violations['reactive_power_generation'] = qg_violations.mean()

                if return_individual:
                    violations['qg_low_violations'] = qg_low_viol.mean()
                    violations['qg_high_violations'] = qg_high_viol.mean()

        # Total bound constraint violation
        total_bound_violation = sum(v for k, v in violations.items()
                                    if not k.endswith('_violations') and isinstance(v, torch.Tensor))
        violations['total_bound_violations'] = total_bound_violation

        return violations

    def evaluate_all_constraints(
        self,
        predictions: Dict[str, torch.Tensor],
        batch_data,
        normalize: bool = True,
        return_individual: bool = True,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Evaluate all constraint violations comprehensively.

        Currently delegates to ``evaluate_bound_constraints`` and prefixes
        results with ``bound_``.  Power-flow and line-flow evaluations are
        planned but not yet active.

        Args:
            predictions (dict[str, torch.Tensor]): Model predictions keyed
                by node type.
            batch_data: Batch data containing network information.
            normalize (bool): Whether to normalize violations (reserved for
                future use).
            return_individual (bool): Whether to include individual violation
                components.

        Returns:
            dict[str, torch.Tensor]: All violation metrics, keyed with a
                ``bound_`` prefix.
        """
        all_violations = {}

        # Evaluate bound constraints
        bound_violations = self.evaluate_bound_constraints(predictions, batch_data, return_individual)
        all_violations.update({f"bound_{k}": v for k, v in bound_violations.items()})

        # Compute total constraint violation
        # violation_keys = ['bound_total_bound_violations', 'real_power_flow_violations', 'reactive_power_flow_violations', 'line_flow_violations']
        # total_violation = sum(all_violations.get(key, torch.tensor(0.0, device=self.device))
        #                       for key in violation_keys)
        # all_violations['total_constraint_violations'] = total_violation

        return all_violations

    def get_violation_summary(
        self,
        violations: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Convert violation tensors to a plain-float summary dictionary.

        Scalar tensors are converted via ``.item()``; multi-element tensors
        are reduced by mean.

        Args:
            violations (dict[str, torch.Tensor]): Dictionary of violation
                tensors as returned by ``evaluate_all_constraints``.

        Returns:
            dict[str, float]: Violation values as Python floats.
        """
        summary = {}

        for key, value in violations.items():
            if isinstance(value, torch.Tensor):
                summary[key] = value.item() if value.numel() == 1 else value.mean().item()
            else:
                summary[key] = float(value)

        return summary


# Convenience function for creating evaluators
def create_constraint_evaluator(
    case_data: Dict,
    device: Optional[torch.device] = None
) -> ACOPFConstraintEvaluator:
    """Create a constraint evaluator from a case-data dictionary.

    Convenience factory that extracts ``voltage_limits``, ``generation_limits``,
    ``line_limits``, ``Y_real``, ``Y_imag``, ``edge_index``, ``base_mva``, and
    ``slack_bus_indices`` from *case_data* and passes them to
    ``ACOPFConstraintEvaluator``.

    Args:
        case_data (dict): Dictionary containing network parameters. Expected
            keys mirror the ``ACOPFConstraintEvaluator`` constructor arguments.
        device (torch.device, optional): Computation device.

    Returns:
        ACOPFConstraintEvaluator: Fully configured evaluator instance.
    """
    # Extract parameters from case data
    voltage_limits = case_data.get('voltage_limits')
    generation_limits = case_data.get('generation_limits')
    line_limits = case_data.get('line_limits')
    Y_real = case_data.get('Y_real')
    Y_imag = case_data.get('Y_imag')
    edge_index = case_data.get('edge_index')
    base_mva = case_data.get('base_mva', 100.0)
    slack_bus_indices = case_data.get('slack_bus_indices')

    return ACOPFConstraintEvaluator(
        voltage_limits=voltage_limits,
        generation_limits=generation_limits,
        line_limits=line_limits,
        Y_real=Y_real,
        Y_imag=Y_imag,
        edge_index=edge_index,
        base_mva=base_mva,
        device=device,
        slack_bus_indices=slack_bus_indices,
    )


if __name__ == "__main__":
    # Test the constraint evaluator
    print("Testing ACOPFConstraintEvaluator...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dummy test data
    batch_size = 32
    n_bus = 14
    n_gen = 5
    n_lines = 20

    # Dummy predictions
    predictions = {
        'bus': torch.randn(batch_size, 2, device=device),  # [VM, VA]
        'generator': torch.randn(batch_size, 2, device=device)  # [PG, QG]
    }

    # Dummy network parameters
    voltage_limits = {
        'vmin': torch.ones(n_bus, device=device) * 0.95,
        'vmax': torch.ones(n_bus, device=device) * 1.05
    }

    generation_limits = {
        'pmin': torch.zeros(n_gen, device=device),
        'pmax': torch.ones(n_gen, device=device) * 2.0,
        'qmin': torch.ones(n_gen, device=device) * -1.0,
        'qmax': torch.ones(n_gen, device=device) * 1.0
    }

    line_limits = torch.ones(n_lines, device=device) * 1.5
    Y_real = torch.randn(n_bus, n_bus, device=device)
    Y_imag = torch.randn(n_bus, n_bus, device=device)
    edge_index = torch.randint(0, n_bus, (2, n_lines), device=device)

    # Create evaluator
    evaluator = ACOPFConstraintEvaluator(
        voltage_limits=voltage_limits,
        generation_limits=generation_limits,
        line_limits=line_limits,
        Y_real=Y_real,
        Y_imag=Y_imag,
        edge_index=edge_index,
        device=device
    )

    # Test evaluation
    violations = evaluator.evaluate_all_constraints(predictions, None)
    summary = evaluator.get_violation_summary(violations)

    print("Constraint violation summary:")
    for key, value in summary.items():
        print(f"  {key}: {value:.6f}")

    print("\nConstraint evaluator test completed successfully!")
