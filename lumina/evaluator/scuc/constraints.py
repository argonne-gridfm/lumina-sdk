"""
SCUC constraint evaluation utilities.

Provides the `SCUCConstraintViolations` helper that mirrors the behaviour from
the monolithic training script, making the violation accounting reusable.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch


class SCUCConstraintViolations:
    """
    Pure-PyTorch module for tracking SCUC constraint violations.

    Tracks violations for:
        - Generator capacity limits
        - Ramping constraints with startup/shutdown surrogates
    """

    def __init__(self, hard_binary: bool = False):
        self.hard_binary = hard_binary

    def extract_generator_params(self, gen_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract generator parameters directly from the feature tensor.

        Expects the feature tensor to contain SCUC-specific columns in the order
        defined by the dataset pipeline (ramp limits, startup/shutdown limits,
        min up/down time, initial status/power, production curve limits).
        """
        rup = gen_features[:, 11]
        rdn = gen_features[:, 12]
        sup = gen_features[:, 13]
        sdn = gen_features[:, 14]
        init_status_hours = gen_features[:, 16]
        init_power = gen_features[:, 17]

        if gen_features.size(1) >= 21:
            pmin = gen_features[:, 19]
            pmax = gen_features[:, 20]
        else:
            device = gen_features.device
            pmin = torch.zeros(gen_features.size(0), device=device)
            pmax = torch.zeros(gen_features.size(0), device=device)

        return {
            "pmin": pmin,
            "pmax": pmax,
            "rup": rup,
            "rdn": rdn,
            "sup": sup,
            "sdn": sdn,
            "init_status_hours": init_status_hours,
            "init_power": init_power,
        }

    def compute_violations(
        self,
        Pg: torch.Tensor,
        ulogits: torch.Tensor,
        pmin: torch.Tensor,
        pmax: torch.Tensor,
        rup: torch.Tensor,
        rdn: torch.Tensor,
        sup: torch.Tensor,
        sdn: torch.Tensor,
        init_status_hours: torch.Tensor,
        init_power: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute capacity and ramping violations for a batch."""
        device = Pg.device
        gen_count, time_steps = Pg.shape

        if self.hard_binary:
            u = (torch.sigmoid(ulogits) >= 0.5).float()
        else:
            u = torch.sigmoid(ulogits)

        u0 = (init_status_hours > 0).float()
        p0 = init_power

        viol_cap_lower = torch.relu(pmin[:, None] * u - Pg)
        viol_cap_upper = torch.relu(Pg - pmax[:, None] * u)

        cap_lower_mask = viol_cap_lower > 0
        cap_upper_mask = viol_cap_upper > 0

        u_ext = torch.cat([u0[:, None], u], dim=1)
        p_ext = torch.cat([p0[:, None], Pg], dim=1)

        du = u_ext[:, 1:] - u_ext[:, :-1]
        startup = torch.relu(du)
        shutdown = torch.relu(-du)

        dP = p_ext[:, 1:] - p_ext[:, :-1]

        viol_ramp_up = torch.relu(dP - (rup[:, None] * u_ext[:, :-1] + sup[:, None] * startup))
        viol_ramp_dn = torch.relu(-dP - (rdn[:, None] * u_ext[:, 1:] + sdn[:, None] * shutdown))

        ramp_up_mask = viol_ramp_up > 0
        ramp_dn_mask = viol_ramp_dn > 0

        total_mw_viol_cap_lower = viol_cap_lower.sum()
        total_mw_viol_cap_upper = viol_cap_upper.sum()
        total_mw_viol_ramp_up = viol_ramp_up.sum()
        total_mw_viol_ramp_dn = viol_ramp_dn.sum()
        total_mw_viol = (
            total_mw_viol_cap_lower
            + total_mw_viol_cap_upper
            + total_mw_viol_ramp_up
            + total_mw_viol_ramp_dn
        )

        num_cap_lower_viol = cap_lower_mask.sum().item()
        num_cap_upper_viol = cap_upper_mask.sum().item()
        num_ramp_up_viol = ramp_up_mask.sum().item()
        num_ramp_dn_viol = ramp_dn_mask.sum().item()
        total_violations = (
            num_cap_lower_viol + num_cap_upper_viol + num_ramp_up_viol + num_ramp_dn_viol
        )

        total_possible = gen_count * time_steps * 4
        violation_fraction = total_violations / total_possible if total_possible else 0.0
        violations_per_generator = total_violations / gen_count if gen_count else 0.0
        violations_per_time = total_violations / time_steps if time_steps else 0.0
        mw_violations_per_generator = total_mw_viol / gen_count if gen_count else 0.0
        avg_mw_violation = total_mw_viol / total_violations if total_violations else 0.0

        summary_str = (
            f"Violations: {violation_fraction:.1%} rate, "
            f"{violations_per_generator:.1f}/gen, "
            f"{violations_per_time:.1f}/time "
            f"(C↓:{num_cap_lower_viol}, C↑:{num_cap_upper_viol}, "
            f"R↑:{num_ramp_up_viol}, R↓:{num_ramp_dn_viol}) "
            f"MW: {total_mw_viol:.0f}"
        )

        return {
            "viol_cap_lower": viol_cap_lower,
            "viol_cap_upper": viol_cap_upper,
            "viol_ramp_up": viol_ramp_up,
            "viol_ramp_dn": viol_ramp_dn,
            "cap_lower_mask": cap_lower_mask,
            "cap_upper_mask": cap_upper_mask,
            "ramp_up_mask": ramp_up_mask,
            "ramp_dn_mask": ramp_dn_mask,
            "total_mw_viol_cap_lower": total_mw_viol_cap_lower.item(),
            "total_mw_viol_cap_upper": total_mw_viol_cap_upper.item(),
            "total_mw_viol_ramp_up": total_mw_viol_ramp_up.item(),
            "total_mw_viol_ramp_dn": total_mw_viol_ramp_dn.item(),
            "total_mw_viol": total_mw_viol.item(),
            "num_cap_lower_viol": num_cap_lower_viol,
            "num_cap_upper_viol": num_cap_upper_viol,
            "num_ramp_up_viol": num_ramp_up_viol,
            "num_ramp_dn_viol": num_ramp_dn_viol,
            "total_violations": total_violations,
            "violation_fraction": violation_fraction,
            "violations_per_generator": violations_per_generator,
            "violations_per_time": violations_per_time,
            "mw_violations_per_generator": mw_violations_per_generator,
            "avg_mw_violation": avg_mw_violation,
            "summary_str": summary_str,
        }


def extract_production_curve_limits(gen_data_dict: Dict[str, Dict], gen_ids) -> Dict[str, torch.Tensor]:
    """
    Extract pmin and pmax from production cost curve data in JSON format.

    Retained for compatibility with scripts that still rely on the helper.
    """
    pmin_list = []
    pmax_list = []

    for gen_id in gen_ids:
        if gen_id in gen_data_dict:
            gen_data = gen_data_dict[gen_id]
            prod_curve_mw = gen_data.get("Production cost curve (MW)", [])
            if len(prod_curve_mw) > 0:
                pmin_list.append(prod_curve_mw[0])
                pmax_list.append(prod_curve_mw[-1])
            else:
                pmin_list.append(0.0)
                pmax_list.append(0.0)
        else:
            pmin_list.append(0.0)
            pmax_list.append(0.0)

    return {
        "pmin": torch.tensor(pmin_list, dtype=torch.float32),
        "pmax": torch.tensor(pmax_list, dtype=torch.float32),
    }


__all__ = ["SCUCConstraintViolations", "extract_production_curve_limits"]

