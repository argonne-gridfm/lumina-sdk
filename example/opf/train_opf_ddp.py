import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

from mpi4py import MPI

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

from lumina.trainer.opf.trainer import MultiCaseOPFTrainer, OPFTrainer
from lumina.trainer.opf.utils import (
    apply_nested,
    init_distributed_runtime,
    parse_case_name,
    parse_cases_arg,
)
from lumina.utils.model import set_seed as _set_seed


def sanitize_run_name(run_name):
    if not run_name:
        return "run"
    safe_chars = []
    for ch in str(run_name):
        if ch.isascii() and (ch.isalnum() or ch in ("-", "_", ".")):
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    safe_name = "".join(safe_chars).strip("_")
    return safe_name or "run"


def build_wandb_run_name(args, case_names, is_multi_case):
    if args.wandb_run_name:
        return args.wandb_run_name
    if is_multi_case:
        case_tag = f"{len(case_names)}cases"
        return f"{case_tag}-{args.model_type}-{args.loss_type}"
    return f"{args.model_type}-{args.loss_type}"


def resolve_checkpoint_dir(config, run_name, global_rank, world_size):
    checkpoint_dir = config.get("checkpoint_dir")
    if not checkpoint_dir:
        return
    checkpointing_config = config.get("checkpointing", {})
    run_scoped = checkpointing_config.get("run_scoped", True)
    if not run_scoped:
        return
    run_id = None
    if global_rank == 0 and wandb is not None and wandb.run is not None:
        run_id = getattr(wandb.run, "id", None)
    if dist.is_available() and dist.is_initialized() and world_size > 1:
        obj_list = [run_id] if global_rank == 0 else [None]
        dist.broadcast_object_list(obj_list, src=0)
        run_id = obj_list[0]
    if not run_id:
        return
    safe_name = sanitize_run_name(run_name)
    resolved_dir = os.path.join(checkpoint_dir, f"{safe_name}-{run_id}")
    config["checkpoint_dir"] = resolved_dir
    if global_rank == 0:
        print(f"Using run-scoped checkpoint_dir: {resolved_dir}")
        if wandb is not None and wandb.run is not None:
            try:
                wandb.run.summary["checkpoint_dir"] = resolved_dir
            except Exception:
                pass


def build_parser():
    parser = argparse.ArgumentParser(description="OPF Training with PyTorch DDP")
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Resume full training state from a Lumina .pt checkpoint",
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
        help="Model type to train (default: HeteroGNN)",
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
        ],
        help="Loss function type (default: mse)",
    )
    parser.add_argument(
        "--minmax_scaling",
        dest="minmax_scaling",
        action="store_true",
        help="Apply min-max scaling to model outputs (default: enabled)",
    )
    parser.add_argument("--wandb", action="store_true", default=False, help="Enable Weights & Biases logging")
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="lumina-training",
        help="Weights & Biases project name (default: lumina-training)",
    )
    parser.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity/team name")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Weights & Biases run name override (default: auto)",
    )
    parser.add_argument(
        "--wandb_group_name",
        type=str,
        default=None,
        help="Weights & Biases group name (default: none)",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default=None,
        choices=["online", "offline", "disabled"],
        help="Weights & Biases mode override",
    )
    return parser


def init_ddp():
    world_size = MPI.COMM_WORLD.Get_size()
    global_rank = MPI.COMM_WORLD.Get_rank()
    local_rank = 0
    local_rank_vars = ["MPI_LOCALRANKID", "SLURM_LOCALID", "LOCAL_RANK"] # Local rank environment variables for Polaris and Perlmutter
    for var in local_rank_vars:
        if var in os.environ:
            local_rank = int(os.environ[var])
            break

    local_rank, global_rank, world_size, _ = init_distributed_runtime(
        local_rank=local_rank,
        global_rank=global_rank,
        world_size=world_size,
        backend="nccl",
    )

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


def init_wandb(args, config, global_rank):
    sweep_overrides = {}
    if args.wandb:
        if not WANDB_AVAILABLE:
            if global_rank == 0:
                print("Warning: Weights & Biases is not available. Install wandb or omit --wandb.")
        elif global_rank == 0:
            logging_dir = config.get("logging_dir")
            wandb_kwargs = {"project": args.wandb_project}
            if args.wandb_entity:
                wandb_kwargs["entity"] = args.wandb_entity
            if args.wandb_group_name:
                wandb_kwargs["group"] = args.wandb_group_name
            if logging_dir:
                wandb_kwargs["dir"] = logging_dir
            try:
                wandb.init(**wandb_kwargs)
                sweep_overrides = dict(wandb.config)
            except Exception as exc:
                print(f"Warning: W&B init failed: {exc}")
    return sweep_overrides


def apply_sweep_overrides(args, config, sweep_overrides):
    if not sweep_overrides:
        return

    reserved_override_keys = {"wandb_version"}
    reserved_args = {"wandb", "wandb_mode"}

    for key, value in sweep_overrides.items():
        if not isinstance(key, str):
            continue
        if key in reserved_override_keys or key.startswith("_"):
            continue
        if key in reserved_args:
            continue
        if hasattr(args, key):
            setattr(args, key, value)
            continue
        apply_nested(config, key, value)


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


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode

    local_rank, global_rank, world_size = init_ddp()

    if global_rank == 0:
        print("ACOPF Training (PyTorch DDP)")
        print("=" * 60)
        print(f"World Size: {world_size}")

    config_path = resolve_config_path(args.config)
    config = load_config(
        config_path,
        global_rank,
        hetero_model_config=args.hetero_model_config,
        homo_model_config=args.homo_model_config,
    )

    sweep_overrides = init_wandb(args, config, global_rank)

    if global_rank == 0 and sweep_overrides:
        apply_sweep_overrides(args, config, sweep_overrides)

    if dist.is_available() and dist.is_initialized() and world_size > 1:
        obj_list = [args, config] if global_rank == 0 else [None, None]
        dist.broadcast_object_list(obj_list, src=0)
        args, config = obj_list

    training_config = config.setdefault("training", {})
    seed = args.seed
    if seed is None:
        seed = training_config.get("seed")
    if seed is None:
        seed = config.get("seed")
    if seed is not None:
        resolved_seed = set_all_seeds(seed)
        if global_rank == 0:
            print(f"Using seed: {resolved_seed}")

    global_batch_size = training_config.get("global_batch_size")
    if global_batch_size is not None:
        try:
            global_batch_size = int(global_batch_size)
        except (TypeError, ValueError) as exc:
            raise ValueError("training.global_batch_size must be an integer") from exc
        if global_batch_size > 0:
            loader_config = config.get("loader", {})
            batch_size = int(loader_config.get("batch_size", 1))
            per_step_global = batch_size * world_size
            if global_batch_size < per_step_global:
                raise ValueError(
                    "training.global_batch_size must be >= loader.batch_size * world_size "
                    f"({per_step_global})."
                )
            if global_batch_size % per_step_global != 0:
                raise ValueError(
                    "training.global_batch_size must be divisible by loader.batch_size * world_size "
                    f"({per_step_global})."
                )
            accumulate_grad_batches = global_batch_size // per_step_global
            training_config["accumulate_grad_batches"] = int(accumulate_grad_batches)
            if global_rank == 0:
                print(
                    "Overriding training.accumulate_grad_batches to "
                    f"{training_config['accumulate_grad_batches']} "
                    f"for global_batch_size={global_batch_size}."
                )

    case_names = resolve_cases(args.cases)
    group_ids = resolve_group_ids(args.group_ids)
    if not group_ids:
        raise ValueError("No valid group IDs provided. Use --group_ids 0 1 ...")
    args.group_ids = group_ids
    is_multi_case = len(case_names) > 1
    if global_rank == 0:
        print(f"Training on cases: {case_names}")
        print(f"Using group IDs: {group_ids}")

    run_name = None
    if args.wandb:
        run_name = build_wandb_run_name(args, case_names, is_multi_case)
        if WANDB_AVAILABLE and global_rank == 0 and wandb is not None and wandb.run is not None:
            try:
                wandb.run.name = run_name
            except Exception:
                pass
        resolve_checkpoint_dir(config, run_name, global_rank, world_size)

    run_metadata = None
    if args.wandb:
        run_metadata = {
            "config_path": str(config_path),
            "group_ids": list(group_ids),
            "loss_type": args.loss_type,
            "model_type": args.model_type,
            "world_size": world_size,
            "cli_args": {
                k: v
                for k, v in vars(args).items()
                if k
                not in {
                    "wandb",
                    "wandb_project",
                    "wandb_entity",
                    "wandb_run_name",
                    "wandb_group_name",
                    "wandb_mode",
                }
            },
        }
        if is_multi_case:
            run_metadata["case_names"] = list(case_names)
        else:
            run_metadata["case_name"] = case_names[0]

    trainer_kwargs = {
        "config": config,
        "group_ids": group_ids,
        "model_type": args.model_type,
        "loss_type": args.loss_type,
        "minmax_scaling": args.minmax_scaling,
        "local_rank": local_rank,
        "global_rank": global_rank,
        "world_size": world_size,
        "wandb_run_name": args.wandb_run_name,
        "wandb_group_name": args.wandb_group_name,
        "wandb_requested": args.wandb,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "run_metadata": run_metadata,
    }

    if is_multi_case:
        trainer_kwargs["case_names"] = case_names
        trainer = MultiCaseOPFTrainer(**trainer_kwargs)
    else:
        trainer_kwargs["case_name"] = case_names[0]
        trainer = OPFTrainer(**trainer_kwargs)

    if args.resume_checkpoint:
        trainer.resume_from_checkpoint(args.resume_checkpoint)

    trainer.train()

    if global_rank == 0:
        print("\nTraining completed!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
