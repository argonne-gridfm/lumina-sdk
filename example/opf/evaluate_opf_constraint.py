import json
import ast
import re
import os

import argparse
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from lumina.model.opf.hetero_model import OPFHeteroGNN
from lumina.dataset.opf.opf_dataset import OPFDataset
from lumina.loader.opf.opf_loader import DataLoader
from lumina.evaluator.opf.evaluator import ACOPFConstraintEvaluator


def convert_checkpoint_key_to_model_key(key: str) -> str:
    """Convert checkpoint key format to model's internal format."""
    pattern = r"<([^>]+)>"

    def replacer(match):
        parts = match.group(1).split('___')
        return f"('{parts[0]}', '{parts[1]}', '{parts[2]}')"

    return re.sub(pattern, replacer, key)


def load_model(device: torch.device, repo_id: str, token: str):
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

    # Manually load parameters and buffers
    for name, param in model.named_parameters():
        ck = convert_checkpoint_key_to_model_key(name)
        if ck in checkpoint_dict:
            param.data.copy_(checkpoint_dict[ck])

    for name, buf in model.named_buffers():
        ck = convert_checkpoint_key_to_model_key(name)
        if ck in checkpoint_dict:
            buf.data.copy_(checkpoint_dict[ck])

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


def _derive_voltage_limits(bus_x: torch.Tensor, device: torch.device):
    """Derive vmin/vmax from bus features (columns 1 and 2) if available."""
    if bus_x is not None and bus_x.size(1) >= 3:
        vmin = bus_x[:, 1].to(device)
        vmax = bus_x[:, 2].to(device)
        return {'vmin': vmin, 'vmax': vmax}
    n_bus = bus_x.size(0) if bus_x is not None else 0
    return {
        'vmin': torch.full((n_bus,), 0.95, device=device),
        'vmax': torch.full((n_bus,), 1.05, device=device),
    }


def _derive_generation_limits(gen_x: torch.Tensor, device: torch.device):
    """
    Derive p/q limits from generator features if present.

    Heuristic mapping based on dataset feature layout:
    - pmin: column 2
    - pmax: column 3
    - qmax: column 1
    - qmin: column 5
    """
    if gen_x is None or gen_x.numel() == 0:
        return None

    n_gen = gen_x.size(0)

    def col_or_default(idx: int, default: float):
        return gen_x[:, idx].to(device) if gen_x.size(1) > idx else torch.full((n_gen,), default, device=device)

    pmin = col_or_default(2, 0.0)
    pmax = col_or_default(3, 2.0)
    qmax = col_or_default(1, 1.0)
    qmin = col_or_default(5, -1.0)

    return {'pmin': pmin, 'pmax': pmax, 'qmin': qmin, 'qmax': qmax}


def _derive_line_params(batch, device: torch.device):
    """
    Build line limits and admittance matrix (Y_real, Y_imag) from ac_line edges.

    Assumes ac_line edge_attr columns:
    [ang_min, ang_max, b_shunt, b_shunt_to, r, x, rate_a, rate_b, rate_c]
    Only r/x/b_shunt and rate_a are used here.
    """
    if ('bus', 'ac_line', 'bus') not in batch.edge_types:
        return None, None, None, None

    edge_index = batch[('bus', 'ac_line', 'bus')].edge_index.to(device)
    edge_attr = batch[('bus', 'ac_line', 'bus')].edge_attr.to(device)
    n_bus = batch['bus'].x.size(0)

    # Line limits: use rate_a (first thermal limit)
    line_limits = edge_attr[:, 6] if edge_attr.size(1) > 6 else torch.ones(edge_index.size(1), device=device)

    # Build admittance matrix
    Y_real = torch.zeros((n_bus, n_bus), device=device, dtype=torch.float32)
    Y_imag = torch.zeros((n_bus, n_bus), device=device, dtype=torch.float32)

    for k in range(edge_index.size(1)):
        i = int(edge_index[0, k])
        j = int(edge_index[1, k])

        r = edge_attr[k, 4].item() if edge_attr.size(1) > 4 else 0.0
        x = edge_attr[k, 5].item() if edge_attr.size(1) > 5 else 0.0
        b_shunt = edge_attr[k, 2].item() if edge_attr.size(1) > 2 else 0.0

        if r == 0.0 and x == 0.0:
            continue

        z = complex(r, x)
        y_series = 1.0 / z
        y_shunt = complex(0.0, b_shunt / 2.0)

        g = y_series.real
        b = y_series.imag + y_shunt.imag

        # Off-diagonal
        Y_real[i, j] -= g
        Y_real[j, i] -= g
        Y_imag[i, j] -= b
        Y_imag[j, i] -= b

        # Diagonal contributions
        Y_real[i, i] += g
        Y_imag[i, i] += b
        Y_real[j, j] += g
        Y_imag[j, j] += b

    return line_limits, Y_real, Y_imag, edge_index


def build_constraint_evaluator(batch, device: torch.device):
    """Build ACOPFConstraintEvaluator using dataset-provided limits when available."""
    bus_x = batch['bus'].x if hasattr(batch['bus'], 'x') else None
    gen_x = batch['generator'].x if 'generator' in batch.node_types and hasattr(batch['generator'], 'x') else None

    voltage_limits = _derive_voltage_limits(bus_x, device)
    generation_limits = _derive_generation_limits(gen_x, device)

    line_limits, Y_real, Y_imag, edge_index = _derive_line_params(batch, device)

    return ACOPFConstraintEvaluator(
        voltage_limits=voltage_limits,
        generation_limits=generation_limits,
        line_limits=line_limits,
        Y_real=Y_real,
        Y_imag=Y_imag,
        edge_index=edge_index,
        base_mva=100.0,
        device=device,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate ACOPF constraint violations on OPF dataset.")
    parser.add_argument("--repo-id", default="argonne/LUMINA-1B", help="HuggingFace repository ID.")
    parser.add_argument("--hf-token", default=None, help="HuggingFace access token (or set HF_TOKEN env).")
    parser.add_argument("--case-name", default="pglib_opf_case14_ieee", help="Case name to load.")
    parser.add_argument("--batch-size", type=int, default=1, help="Evaluation batch size.")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit number of batches to evaluate (None = all).")
    args = parser.parse_args()

    token = args.hf_token or os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HuggingFace token not provided. Supply --hf-token or set HF_TOKEN.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Downloading model and loading weights...")
    model, _ = load_model(device, repo_id=args.repo_id, token=token)

    print("Loading OPF dataset...")
    dataset = OPFDataset(root='./opf_data', case_name=args.case_name)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print(f"\n{'=' * 60}")
    print("Running constraint evaluation")
    print(f"{'=' * 60}")
    print(f"Device: {device}")

    with torch.no_grad():
        accum_sum = {}
        accum_sq = {}
        accum_weight = {}
        batches_seen = 0

        for batch_idx, batch in enumerate(loader):
            batch = to_float32(batch).to(device)

            predictions = model(
                batch.x_dict,
                batch.edge_index_dict,
                batch.edge_attr_dict if hasattr(batch, 'edge_attr_dict') else None,
            )

            evaluator = build_constraint_evaluator(batch, device)
            violations = evaluator.evaluate_all_constraints(
                predictions=predictions,
                batch_data=batch,
                normalize=True,
                return_individual=True,
            )
            summary = evaluator.get_violation_summary(violations)

            # accumulate weighted sums and sums of squares for variance later
            sample_weight = batch['bus'].batch.max().item() + 1 if hasattr(batch['bus'], 'batch') else args.batch_size
            for key, value in summary.items():
                v = float(value)
                accum_sum[key] = accum_sum.get(key, 0.0) + v * sample_weight
                accum_sq[key] = accum_sq.get(key, 0.0) + v * v * sample_weight
                accum_weight[key] = accum_weight.get(key, 0.0) + sample_weight

            batches_seen += 1

            if args.max_batches is not None and (batch_idx + 1) >= args.max_batches:
                print(f"\nReached max_batches={args.max_batches}; stopping early.")
                break

    if batches_seen > 0:
        print(f"\n{'=' * 60}")
        print(f"Constraint violation stats over {batches_seen} batch(es)")
        print(f"{'=' * 60}")
        for key in sorted(accum_sum.keys()):
            weight = accum_weight.get(key, 0.0)
            if weight == 0:
                continue
            mean = accum_sum[key] / weight
            mean_sq = accum_sq[key] / weight
            var = mean_sq - mean * mean
            print(f"{key}: mean={mean:.6f}, var={var:.6e}")


if __name__ == '__main__':
    main()
