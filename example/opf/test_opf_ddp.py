import argparse
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

from lumina.trainer.opf.trainer import MultiCaseOPFTrainer, OPFTrainer
from lumina.trainer.opf.utils import parse_case_name, parse_cases_arg
from lumina.utils.model import set_seed as _set_seed


def build_parser():
    parser = argparse.ArgumentParser(description="OPF Testing with PyTorch DDP")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require checkpoint keys to exactly match model keys",
    )
    parser.add_argument(
        "--cases",
        type=str,
        nargs="+",
        default=["case14"],
        help="List of case names (short form like case14 or full pglib names)",
    )
    parser.add_argument(
        "--group_ids",
        type=int,
        nargs="+",
        default=[0],
        help="Group IDs for dataset (default: 0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: disabled)",
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument(
        "--hetero_model_config",
        type=str,
        default=None,
        help="Path to hetero model config YAML (heterognn.yaml)",
    )
    parser.add_argument(
        "--homo_model_config",
        type=str,
        default=None,
        help="Path to homo model config YAML (homognn.yaml)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="HeteroGNN",
        choices=["HeteroGNN", "RGAT", "HGT", "HEAT", "GCN", "GAT", "GIN", "Transformer"],
        help="Model type to test (default: HeteroGNN)",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="mse",
        choices=[
            "mse",
            "rmse",
            "mae",
            "mape",
            "smooth_l1",
            "augmented_lagrangian",
            "violated_lagrangian",
        ],
        help="Loss function type (default: mse)",
    )
    parser.add_argument(
        "--minmax_scaling",
        dest="minmax_scaling",
        action="store_true",
        help="Apply min-max scaling to model outputs (default: enabled)",
    )
    parser.add_argument(
        "--violation_eval_p",
        type=float,
        default=None,
        help="Override training.violation_eval_p for test evaluation",
    )
    return parser


def init_ddp():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    global_rank = int(os.environ.get("RANK", 0))

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=global_rank,
        device_id=local_rank,
    )

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return local_rank, global_rank, world_size


def set_all_seeds(seed):
    if seed is None:
        return None
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    _set_seed(seed)
    return seed


def resolve_config_path(config_path):
    if os.path.exists(config_path):
        return config_path

    parent_config = os.path.join(Path(__file__).parent.parent, "config_files", "single.yaml")
    if os.path.exists(parent_config):
        return parent_config

    acopf_config = os.path.join(
        Path(__file__).parent.parent.parent,
        "configs",
        "config.polaris.ddp.yaml",
    )
    if os.path.exists(acopf_config):
        return acopf_config

    return config_path


def load_config(config_path, global_rank, hetero_model_config=None, homo_model_config=None):
    if global_rank == 0:
        print(f"Loading config from: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if "models" in config:
        return config

    config_dir = Path(config_path).parent
    model_configs = (
        ("hetero", hetero_model_config, "heterognn.yaml"),
        ("homo", homo_model_config, "homognn.yaml"),
    )
    for label, requested_path, default_name in model_configs:
        if requested_path:
            model_config_path = Path(requested_path)
            if not model_config_path.exists():
                if global_rank == 0:
                    print(f"Warning: {label} model config not found at: {model_config_path}")
                continue
        else:
            model_config_path = config_dir / "model" / default_name
            if not model_config_path.exists():
                fallback_path = Path(__file__).parent.parent / "configs" / "model" / default_name
                if fallback_path.exists():
                    model_config_path = fallback_path
                else:
                    continue

        if global_rank == 0:
            print(f"Loading additional {label} model config from: {model_config_path}")
        with open(model_config_path, "r") as f:
            model_config = yaml.safe_load(f)
            if "models" in model_config:
                if "models" not in config:
                    config["models"] = {}
                config["models"].update(model_config["models"])

    return config


def normalize_cases_arg(cases_arg):
    if cases_arg is None:
        return []
    if isinstance(cases_arg, str):
        cases_arg = [cases_arg]
    return parse_cases_arg(cases_arg)


def resolve_cases(cases_arg):
    raw_cases = normalize_cases_arg(cases_arg)
    if not raw_cases:
        raise ValueError("No valid cases provided. Use --cases case14 case57 ...")
    return [parse_case_name(case) for case in raw_cases]


def resolve_group_ids(group_ids_arg):
    if group_ids_arg is None:
        return []
    if isinstance(group_ids_arg, int):
        return [group_ids_arg]
    if isinstance(group_ids_arg, str):
        group_ids = parse_cases_arg([group_ids_arg])
    else:
        group_ids = []
        for entry in group_ids_arg:
            if isinstance(entry, str):
                group_ids.extend(parse_cases_arg([entry]))
            else:
                group_ids.append(entry)
    return [int(group_id) for group_id in group_ids]


def _replace_val_prefix(name):
    if name.startswith("val/"):
        return "test/" + name[len("val/") :]
    return name


def build_test_metrics(trainer):
    metric_names = [_replace_val_prefix(name) for name in trainer.val_metric_names]
    metric_map = [(info_key, _replace_val_prefix(metric_name)) for info_key, metric_name in trainer.val_metric_map]
    metric_groups = [[_replace_val_prefix(name) for name in group] for group in trainer.val_metric_groups]
    return metric_names, metric_map, metric_groups


def add_test_score(metric_avgs, score_alpha, log_normalized_violation):
    if not metric_avgs:
        return
    objective = metric_avgs.get("test/loss/objective")
    if objective is None:
        objective = 0.0
    violation_key = "test/feas/total_violation_norm" if log_normalized_violation else "test/feas/total_violation"
    violation = metric_avgs.get(violation_key)
    if violation is None:
        violation = 0.0
    metric_avgs["test/score"] = objective + score_alpha * violation


def iter_test_loaders(trainer):
    if hasattr(trainer, "test_loaders") and trainer.test_loaders:
        case_indices = trainer.test_case_indices or sorted(trainer.test_loaders.keys())
        for case_idx in case_indices:
            loader = trainer.test_loaders.get(case_idx)
            if loader is None:
                continue
            case_label = (
                trainer.case_names[case_idx]
                if hasattr(trainer, "case_names") and case_idx < len(trainer.case_names)
                else f"case_{case_idx}"
            )
            yield case_idx, case_label, loader
        return
    if hasattr(trainer, "test_loader") and trainer.test_loader is not None:
        case_label = trainer.case_name if hasattr(trainer, "case_name") else "case_0"
        yield 0, case_label, trainer.test_loader


def load_checkpoint(trainer, checkpoint_path, strict, global_rank):
    if global_rank == 0:
        print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain a model state dict.")
    load_result = trainer.model.module.load_state_dict(state_dict, strict=strict)
    if global_rank == 0 and not strict:
        if load_result.missing_keys or load_result.unexpected_keys:
            print(
                "[CHECKPOINT LOAD] Missing keys: "
                f"{load_result.missing_keys}, Unexpected keys: {load_result.unexpected_keys}"
            )
    return checkpoint


def run_test(trainer):
    if trainer.violation_eval_p <= 0.0:
        if trainer.global_rank == 0:
            print("violation_eval_p <= 0; skipping test evaluation.")
        return None, None, None, None

    test_metric_names, test_metric_map, test_metric_groups = build_test_metrics(trainer)
    metric_sums, metric_counts = trainer._init_metric_trackers(test_metric_names)
    total_loss = 0.0
    total_task_loss = 0.0
    num_batches = 0
    timed_batches = 0

    test_loaders = list(iter_test_loaders(trainer))
    if not test_loaders:
        if trainer.global_rank == 0:
            print("No test data available.")
        return None, None, None, None

    trainer.model.eval()

    with torch.no_grad():
        for case_idx, case_label, loader in test_loaders:
            loss_manager = (
                trainer.loss_managers[case_idx]
                if hasattr(trainer, "loss_managers")
                else trainer.loss_manager
            )
            rng = trainer._make_violation_eval_rng(case_idx)
            try:
                total_batches = len(loader)
            except Exception:
                total_batches = None
            min_eval_batches = trainer.violation_eval_min_batches
            if total_batches is not None and total_batches > 0:
                min_eval_batches = min(min_eval_batches, total_batches)
            eval_batches = 0

            if trainer.global_rank == 0 and len(test_loaders) > 1:
                print(f"Evaluating test split for {case_label}...")

            for batch_idx, batch in enumerate(loader):
                do_eval = trainer._should_eval_violations(
                    rng,
                    batch_idx,
                    total_batches,
                    eval_batches,
                    min_eval_batches,
                )
                if not do_eval:
                    continue
                eval_batches += 1

                do_timing = trainer._should_time_validation_batch(batch_idx, timed_batches)
                if do_timing:
                    trainer._sync_for_timing()
                    timing_start = time.perf_counter()
                batch = batch.to(trainer.device)
                if do_timing:
                    trainer._sync_for_timing()
                    timing_after_data = time.perf_counter()

                predictions = trainer.forward(batch)
                if do_timing:
                    trainer._sync_for_timing()
                    timing_after_forward = time.perf_counter()
                if loss_manager.lagrangian is None:
                    loss, loss_info = loss_manager.compute_loss(
                        predictions,
                        batch,
                        return_info=True,
                        collect_constraints=True,
                    )
                else:
                    loss, loss_info = loss_manager.compute_loss(
                        predictions,
                        batch,
                        return_info=True,
                    )
                if do_timing:
                    trainer._sync_for_timing()
                    timing_after_loss = time.perf_counter()
                    trainer._add_metric(
                        metric_sums,
                        metric_counts,
                        "test/perf/data_ms",
                        (timing_after_data - timing_start) * 1000.0,
                    )
                    trainer._add_metric(
                        metric_sums,
                        metric_counts,
                        "test/perf/forward_ms",
                        (timing_after_forward - timing_after_data) * 1000.0,
                    )
                    trainer._add_metric(
                        metric_sums,
                        metric_counts,
                        "test/perf/loss_ms",
                        (timing_after_loss - timing_after_forward) * 1000.0,
                    )
                    trainer._add_metric(
                        metric_sums,
                        metric_counts,
                        "test/perf/total_ms",
                        (timing_after_loss - timing_start) * 1000.0,
                    )
                    timed_batches += 1

                loss_value = loss.item()
                total_loss += loss_value
                trainer._add_metric(metric_sums, metric_counts, "test/loss/total", loss_value)
                if "objective" in loss_info:
                    objective_value = trainer._as_float(loss_info["objective"])
                    if objective_value is not None:
                        total_task_loss += objective_value
                        trainer._add_metric(
                            metric_sums,
                            metric_counts,
                            "test/loss/objective",
                            objective_value,
                        )
                for info_key, metric_name in test_metric_map:
                    if info_key in loss_info:
                        trainer._add_metric(metric_sums, metric_counts, metric_name, loss_info[info_key])
                num_batches += 1

    if num_batches == 0:
        return None, None, None, None

    totals = torch.tensor(
        [total_loss, total_task_loss, float(num_batches)],
        device=trainer.device,
    )
    if dist.is_available() and dist.is_initialized() and trainer.world_size > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    total_loss, total_task_loss, total_batches = totals.tolist()
    if total_batches <= 0:
        return None, None, None, None

    avg_loss = total_loss / total_batches
    avg_task_loss = total_task_loss / total_batches

    metric_sums, metric_counts = trainer._reduce_metrics(metric_sums, metric_counts)
    metric_avgs = trainer._compute_metric_avgs(metric_sums, metric_counts)
    metric_avgs["test/perf/eval_batches"] = float(total_batches)
    add_test_score(metric_avgs, trainer.score_alpha, trainer.log_normalized_violation)

    return avg_loss, avg_task_loss, metric_avgs, test_metric_groups


def main():
    parser = build_parser()
    args = parser.parse_args()

    local_rank, global_rank, world_size = init_ddp()

    if global_rank == 0:
        print("ACOPF Testing (PyTorch DDP)")
        print("=" * 60)
        print(f"World Size: {world_size}")

    config_path = resolve_config_path(args.config)
    config = load_config(
        config_path,
        global_rank,
        hetero_model_config=args.hetero_model_config,
        homo_model_config=args.homo_model_config,
    )

    training_config = config.setdefault("training", {})
    if args.violation_eval_p is not None:
        training_config["violation_eval_p"] = float(args.violation_eval_p)

    seed = args.seed
    if seed is None:
        seed = training_config.get("seed")
    if seed is None:
        seed = config.get("seed")
    if seed is not None:
        resolved_seed = set_all_seeds(seed)
        if global_rank == 0:
            print(f"Using seed: {resolved_seed}")

    case_names = resolve_cases(args.cases)
    group_ids = resolve_group_ids(args.group_ids)
    if not group_ids:
        raise ValueError("No valid group IDs provided. Use --group_ids 0 1 ...")
    args.group_ids = group_ids
    is_multi_case = len(case_names) > 1
    if global_rank == 0:
        print(f"Testing on cases: {case_names}")
        print(f"Using group IDs: {group_ids}")

    trainer_kwargs = {
        "config": config,
        "group_ids": group_ids,
        "model_type": args.model_type,
        "loss_type": args.loss_type,
        "minmax_scaling": args.minmax_scaling,
        "local_rank": local_rank,
        "global_rank": global_rank,
        "world_size": world_size,
        "wandb_run_name": None,
        "wandb_group_name": None,
        "wandb_requested": False,
        "wandb_project": "lumina-training",
        "wandb_entity": None,
        "run_metadata": None,
    }

    if is_multi_case:
        trainer_kwargs["case_names"] = case_names
        trainer = MultiCaseOPFTrainer(**trainer_kwargs)
    else:
        trainer_kwargs["case_name"] = case_names[0]
        trainer = OPFTrainer(**trainer_kwargs)

    load_checkpoint(trainer, args.checkpoint, args.strict, global_rank)

    test_loss, test_task_loss, test_metrics, test_metric_groups = run_test(trainer)

    if global_rank == 0:
        if test_loss is None:
            print("Test skipped or no evaluated batches.")
        else:
            print("\nTest results:")
            print(f"  Test Loss: {test_loss:.4f}, Test Task: {test_task_loss:.4f}")
            if test_metrics:
                trainer._print_metric_groups("  Test Metrics:", test_metrics, test_metric_groups)
        print("\nTesting completed!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
