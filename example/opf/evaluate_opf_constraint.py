import argparse
import ast
import json
import os
import re

import torch
import yaml
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from tqdm.auto import tqdm

from lumina.dataset.opf.opf_dataset import OPFDataset
from lumina.evaluator.opf.evaluator import ACOPFConstraintEvaluator
from lumina.evaluator.opf.utils import Modeler
from lumina.loader.opf.opf_loader import DataLoader
from lumina.model.opf.hetero_model import OPFHeteroGNN
from lumina.model.opf.losses import OPFLossManager
from lumina.trainer.opf.trainer import BaseOPFTrainer


def convert_checkpoint_key_to_model_key(key: str) -> str:
    """Convert checkpoint key format to model's internal format."""
    pattern = r"<([^>]+)>"

    def replacer(match):
        parts = match.group(1).split('___')
        return f"('{parts[0]}', '{parts[1]}', '{parts[2]}')"

    return re.sub(pattern, replacer, key)


def _load_checkpoint_into_model(
    model: torch.nn.Module,
    checkpoint_dict,
    *,
    fail_on_missing: bool = False,
    verbose: bool = True,
):
    """Load a checkpoint dict into model with reporting of missing/unexpected keys."""
    model_state = model.state_dict()
    used_keys = set()
    missing_keys = []

    remapped_state = {}
    for model_key in model_state.keys():
        ck = convert_checkpoint_key_to_model_key(model_key)
        if ck in checkpoint_dict:
            remapped_state[model_key] = checkpoint_dict[ck]
            used_keys.add(ck)

    unexpected_keys = [k for k in checkpoint_dict.keys() if k not in used_keys]

    load_result = model.load_state_dict(remapped_state, strict=False)
    missing_keys = list(load_result.missing_keys)
    unexpected_keys.extend(list(load_result.unexpected_keys))

    if verbose and (missing_keys or unexpected_keys):
        print(f"[CHECKPOINT LOAD] Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}")
    if fail_on_missing and missing_keys:
        raise ValueError(f"Missing keys during load: {missing_keys}")

    return {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys}


def load_model(
    device: torch.device,
    repo_id: str,
    token: str,
    *,
    fail_on_missing: bool = False,
    verbose: bool = True,
):
    """Load OPF model and config from HuggingFace."""
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json", token=token)
    safetensors_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors", token=token)

    with open(config_path, 'r') as f:
        config_data = json.load(f)

    # Convert metadata edge keys from strings to tuples
    if 'edges' in config_data.get('metadata', {}):
        edges_dict = {}
        for key, value in config_data['metadata']['edges'].items():
            if isinstance(key, str) and key.startswith('('):
                key = ast.literal_eval(key)
            edges_dict[key] = value
        config_data['metadata']['edges'] = edges_dict

    model = OPFHeteroGNN(
        metadata=config_data['metadata'],
        input_channels=config_data['input_channels'],
        hidden_channels=config_data['config']['models']['HeteroGNN']['hidden_channels'],
        num_layers=config_data['config']['models']['HeteroGNN']['num_layers'],
        backend=config_data['config']['models']['HeteroGNN']['backend'],
    ).to(device)

    state_dict = load_file(safetensors_path)
    checkpoint_dict = {convert_checkpoint_key_to_model_key(k): v for k, v in state_dict.items()}

    _load_checkpoint_into_model(
        model,
        checkpoint_dict,
        fail_on_missing=fail_on_missing,
        verbose=verbose,
    )

    model.eval()
    return model, config_data


def to_float32(batch):
    """Convert node features/targets and edge attributes to float32."""
    for node_type in batch.node_types:
        if getattr(batch[node_type], 'x', None) is not None:
            batch[node_type].x = batch[node_type].x.float()
        if getattr(batch[node_type], 'y', None) is not None:
            batch[node_type].y = batch[node_type].y.float()

    for edge_type in batch.edge_types:
        if getattr(batch[edge_type], 'edge_attr', None) is not None:
            batch[edge_type].edge_attr = batch[edge_type].edge_attr.float()

    return batch


def main():
    parser = argparse.ArgumentParser(description="Evaluate ACOPF constraint violations on OPF dataset.")
    parser.add_argument("--repo-id", default="argonne/LUMINA-1B", help="HuggingFace repository ID.")
    parser.add_argument("--hf-token", default=None, help="HuggingFace access token (or set HF_TOKEN env).")
    parser.add_argument("--case-name", default="pglib_opf_case14_ieee", help="Case name to load.")
    parser.add_argument("--batch-size", type=int, default=1, help="Evaluation batch size.")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit number of batches to evaluate (None = all).")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional training config YAML to load lagrangian/log_normalized_violation settings.",
    )
    parser.add_argument(
        "--loss-type",
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
        help="Loss function type (default: mse).",
    )
    parser.add_argument("--fail-on-missing", action="store_true", help="Raise an error if checkpoint is missing model keys.")
    parser.add_argument(
        "--slack-bus-indices",
        default="0",
        help="Comma-separated slack bus indices (default: 0).",
    )
    args = parser.parse_args()

    token = args.hf_token or os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HuggingFace token not provided. Supply --hf-token or set HF_TOKEN.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Downloading model and loading weights...")
    model, _ = load_model(
        device,
        repo_id=args.repo_id,
        token=token,
        fail_on_missing=args.fail_on_missing,
        verbose=True,
    )

    print("Loading OPF dataset...")
    dataset = OPFDataset(root='./opf_data', case_name=args.case_name)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    lag_config = {}
    log_normalized_violation = False
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        lag_config = config.get("lagrangian", {}) or {}
        training_config = config.get("training", {}) or {}
        log_normalized_violation = bool(training_config.get("log_normalized_violation", False))

    loss_manager = OPFLossManager(
        loss_type=args.loss_type,
        device=device,
        lagrangian_config=lag_config,
        log_normalized_violation=log_normalized_violation,
    )
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

        total_batches = None
        try:
            total_batches = len(loader)
        except TypeError:
            total_batches = None
        if total_batches is not None and args.max_batches is not None:
            total_batches = min(total_batches, args.max_batches)

        progress_iter = tqdm(loader, total=total_batches, desc="Evaluating samples")

        for batch_idx, batch in enumerate(progress_iter):
            batch = to_float32(batch).to(device)

            predictions = model(
                batch.x_dict,
                batch.edge_index_dict,
                batch.edge_attr_dict if hasattr(batch, 'edge_attr_dict') else None,
                minmax_scaling=True,
            )

            if loss_manager.lagrangian is None:
                _, loss_info = loss_manager.compute_loss(
                    predictions,
                    batch,
                    return_info=True,
                    collect_constraints=True,
                )
            else:
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
                constraint_backend=loss_manager,
            )
            summary = evaluator.get_violation_summary(violations)

            backend_info = {k[len("backend_"):]: v for k, v in summary.items() if k.startswith("backend_")}
            bound_info = {k: v for k, v in summary.items() if k.startswith("bound_")}

            for info_key, metric_name in metric_map:
                if info_key in backend_info:
                    metric_sums[metric_name] += float(backend_info[info_key])
                    metric_counts[metric_name] += 1.0
                elif info_key in loss_info:
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
