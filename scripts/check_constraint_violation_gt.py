#!/usr/bin/env python3
import argparse
from typing import Dict, Optional

import torch

from lumina.dataset.opf.opf_dataset import OPFDataset
from lumina.loader.opf.opf_loader import DataLoader
from lumina.model.opf.losses import OPFLossManager
from lumina.trainer.opf.utils import parse_case_name


def to_float32(batch):
    """Convert node features/targets and edge attributes to float32."""
    node_types = getattr(batch, "node_types", [])
    for node_type in node_types:
        if getattr(batch[node_type], "x", None) is not None:
            batch[node_type].x = batch[node_type].x.float()
        if getattr(batch[node_type], "y", None) is not None:
            batch[node_type].y = batch[node_type].y.float()

    edge_types = getattr(batch, "edge_types", [])
    for edge_type in edge_types:
        if getattr(batch[edge_type], "edge_attr", None) is not None:
            batch[edge_type].edge_attr = batch[edge_type].edge_attr.float()

    return batch


def as_float(value: Optional[torch.Tensor]) -> Optional[float]:
    if value is None:
        return None
    if torch.is_tensor(value):
        return float(value.detach().item()) if value.numel() == 1 else float(value.mean().detach().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_predictions_from_targets(batch) -> Dict[str, torch.Tensor]:
    preds = {}
    if "bus" in getattr(batch, "node_types", []) and getattr(batch["bus"], "y", None) is not None:
        preds["bus"] = batch["bus"].y.float()
    if "generator" in getattr(batch, "node_types", []) and getattr(batch["generator"], "y", None) is not None:
        preds["generator"] = batch["generator"].y.float()
    return preds


def get_batch_size(batch, fallback: int) -> int:
    if "bus" in getattr(batch, "node_types", []) and hasattr(batch["bus"], "batch"):
        return int(batch["bus"].batch.max().item()) + 1
    return fallback


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check constraint violations using ground-truth targets as predictions."
    )
    parser.add_argument("--case", default="case14", help="Case name (case14 or full pglib name).")
    parser.add_argument("--group-id", type=int, default=0, help="OPF group id (0-19).")
    parser.add_argument("--root", default="./opf_data", help="Dataset root directory.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for evaluation.")
    parser.add_argument("--max-batches", type=int, default=5, help="Max number of batches to evaluate.")
    parser.add_argument("--max-samples", type=int, default=None, help="Stop after this many samples.")
    parser.add_argument(
        "--normalize-by-rms",
        action="store_true",
        help="Normalize constraint vector by RMS magnitude.",
    )
    parser.add_argument(
        "--no-normalize-by-size",
        action="store_true",
        help="Disable normalization by sqrt(number_of_constraints).",
    )
    parser.add_argument(
        "--log-normalized-violation",
        action="store_true",
        help="Log normalized violation metrics (divide by sqrt(n_constraints)).",
    )
    parser.add_argument(
        "--angles-in-degrees",
        action="store_true",
        help="Interpret voltage angles as degrees (default: radians).",
    )
    parser.add_argument("--device", default=None, help="Device override (e.g., cpu, cuda).")
    args = parser.parse_args()

    case_name = parse_case_name(args.case)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = OPFDataset(root=args.root, case_name=case_name, group_id=args.group_id)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    loss_manager = OPFLossManager(
        loss_type="mse",
        device=device,
    )
    loss_manager.eval()

    metric_sums: Dict[str, float] = {}
    metric_counts: Dict[str, int] = {}
    batches_seen = 0
    samples_seen = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

            batch = to_float32(batch).to(device)
            batch_size = get_batch_size(batch, args.batch_size)
            samples_seen += batch_size

            predictions = build_predictions_from_targets(batch)
            if not predictions:
                print("No bus/generator targets found in batch; cannot build predictions.")
                break

            loss, info = loss_manager.compute_loss(
                predictions,
                batch,
                return_info=True,
                collect_constraints=True,
            )

            metrics = {
                "loss": loss,
                "raw_constraint_violation": info.get("raw_constraint_violation"),
                "raw_constraint_violation_norm": info.get("raw_constraint_violation_norm"),
                "p_balance_rmse": info.get("p_balance_rmse"),
                "q_balance_rmse": info.get("q_balance_rmse"),
                "line_limit_rmse": info.get("line_limit_rmse"),
            }

            pretty_parts = []
            for name, value in metrics.items():
                val = as_float(value)
                if val is None:
                    continue
                metric_sums[name] = metric_sums.get(name, 0.0) + val
                metric_counts[name] = metric_counts.get(name, 0) + 1
                pretty_parts.append(f"{name}={val:.6e}")

            print(f"Batch {batch_idx}: " + ", ".join(pretty_parts))
            batches_seen += 1

            if args.max_samples is not None and samples_seen >= args.max_samples:
                break

    if batches_seen == 0:
        print("No batches evaluated.")
        return

    print("\nAverages over evaluated batches:")
    for name in sorted(metric_sums.keys()):
        count = metric_counts.get(name, 0)
        if count <= 0:
            continue
        avg = metric_sums[name] / count
        print(f"{name}: {avg:.6e}")


if __name__ == "__main__":
    main()
