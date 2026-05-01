"""Evaluate ACOPF constraint violations of a trained OPF model on a dataset.

Loads a checkpoint produced by ``train_opf_ddp.py`` (or ``train_opf_simple.py``)
via ``Modeler.load_model_from_training_checkpoint``, runs predictions over the
specified case, and reports per-batch loss + bound-constraint violation
statistics aggregated over the run.
"""
import argparse

import torch
from tqdm.auto import tqdm

from lumina.dataset.opf.opf_dataset import OPFDataset
from lumina.dataset.opf.transforms import to_float32
from lumina.evaluator.opf.evaluator import ACOPFConstraintEvaluator
from lumina.evaluator.opf.utils import Modeler
from lumina.loader.opf.opf_loader import DataLoader
from lumina.model.opf.losses import OPFLossManager
from lumina.trainer.opf.trainer import BaseOPFTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ACOPF constraint violations on an OPF dataset."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a training checkpoint .pt produced by train_opf_*.py "
             "(must contain model_class / model_kwargs / model_state_dict).",
    )
    parser.add_argument(
        "--root",
        default="./opf_data",
        help="Dataset cache directory (default: ./opf_data).",
    )
    parser.add_argument("--case-name", default="pglib_opf_case14_ieee", help="Case name to load.")
    parser.add_argument("--group-id", type=int, default=0, help="Dataset group id.")
    parser.add_argument("--batch-size", type=int, default=1, help="Evaluation batch size.")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Limit number of batches to evaluate (None = all).")
    parser.add_argument(
        "--loss-type",
        type=str,
        default="mse",
        choices=["mse", "rmse", "mae", "mape", "smooth_l1"],
        help="Loss function type (default: mse).",
    )
    parser.add_argument("--fail-on-missing", action="store_true",
                        help="Raise an error if checkpoint is missing model keys.")
    parser.add_argument(
        "--slack-bus-indices",
        default="0",
        help="Comma-separated slack bus indices (default: 0).",
    )
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading model from checkpoint: {args.checkpoint}")
    modeler = Modeler(
        device=device,
        fail_on_missing=args.fail_on_missing,
        slack_bus_indices=args.slack_bus_indices,
    )
    model = modeler.load_model_from_training_checkpoint(args.checkpoint)
    model.eval()

    print("Loading OPF dataset...")
    dataset = OPFDataset(
        root=args.root,
        case_name=args.case_name,
        group_id=args.group_id,
        transform=to_float32,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    loss_manager = OPFLossManager(loss_type=args.loss_type, device=device)
    loss_manager.eval()

    slack_bus_indices = [int(x) for x in args.slack_bus_indices.split(",") if x.strip() != ""]
    evaluator = ACOPFConstraintEvaluator(device=device)
    evaluator.slack_bus_indices = slack_bus_indices

    metric_map = BaseOPFTrainer.VAL_METRIC_MAP
    metric_sums = {name: 0.0 for _, name in metric_map}
    metric_counts = {name: 0.0 for _, name in metric_map}

    bound_sum = {}
    bound_sq = {}
    bound_weight = {}

    print(f"\n{'=' * 80}")
    print("Running constraint evaluation")
    print(f"{'=' * 80}")
    print(f"Device: {device}")

    with torch.no_grad():
        batches_seen = 0

        try:
            total_batches = len(loader)
        except TypeError:
            total_batches = None
        if total_batches is not None and args.max_batches is not None:
            total_batches = min(total_batches, args.max_batches)

        progress_iter = tqdm(loader, total=total_batches, desc="Evaluating samples")

        for batch_idx, batch in enumerate(progress_iter):
            batch = batch.to(device)

            predictions = model(
                batch.x_dict,
                batch.edge_index_dict,
                batch.edge_attr_dict if hasattr(batch, 'edge_attr_dict') else None,
                minmax_scaling=True,
            )

            _, loss_info = loss_manager.compute_loss(
                predictions,
                batch,
                return_info=True,
            )

            bus_x = batch['bus'].x if hasattr(batch['bus'], 'x') else None
            gen_x = batch['generator'].x if 'generator' in batch.node_types and hasattr(batch['generator'], 'x') else None
            voltage_limits = Modeler.derive_voltage_limits(bus_x, device)
            generation_limits = Modeler.derive_generation_limits(gen_x, device)
            evaluator.set_network_parameters(
                voltage_limits=voltage_limits,
                generation_limits=generation_limits,
            )

            violations = evaluator.evaluate_all_constraints(
                predictions=predictions,
                batch_data=batch,
                normalize=True,
                return_individual=False,
            )
            summary = evaluator.get_violation_summary(violations)

            bound_info = {k: v for k, v in summary.items() if k.startswith("bound_")}

            for info_key, metric_name in metric_map:
                if info_key in loss_info:
                    metric_sums[metric_name] += float(loss_info[info_key])
                    metric_counts[metric_name] += 1.0

            sample_weight = batch['bus'].batch.max().item() + 1 if hasattr(batch['bus'], 'batch') else 1
            for key, value in bound_info.items():
                v = float(value)
                bound_sum[key] = bound_sum.get(key, 0.0) + v * sample_weight
                bound_sq[key] = bound_sq.get(key, 0.0) + v * v * sample_weight
                bound_weight[key] = bound_weight.get(key, 0.0) + sample_weight

            batches_seen += 1
            progress_iter.set_postfix(batches=batches_seen, refresh=False)

            if args.max_batches is not None and (batch_idx + 1) >= args.max_batches:
                progress_iter.write(f"Reached max_batches={args.max_batches}.")
                break

        progress_iter.close()

    if batches_seen > 0:
        print(f"\n{'=' * 80}")
        print(f"Training-style constraint metrics over {batches_seen} batch(es)")
        print(f"{'=' * 80}")
        for metric_name in sorted(metric_sums.keys()):
            count = metric_counts.get(metric_name, 0.0)
            if count == 0:
                continue
            mean = metric_sums[metric_name] / count
            print(f"{metric_name:35s}: mean={mean:.6f}")

        if bound_sum:
            print(f"\n{'=' * 80}")
            print(f"Bound constraint metrics over {batches_seen} batch(es)")
            print(f"{'=' * 80}")
            for key in sorted(bound_sum.keys()):
                weight = bound_weight.get(key, 0.0)
                if weight == 0:
                    continue
                mean = bound_sum[key] / weight
                mean_sq = bound_sq[key] / weight
                var = mean_sq - mean * mean
                print(f"{key:35s}: mean={mean:.6f}, var={var:.6e}")


if __name__ == '__main__':
    main()
