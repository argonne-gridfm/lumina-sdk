"""
see args for usage details, three run modes are supported (HF, safetensors, training checkpoint) 
admittance matrix is required for computing some of the node and type level metrics,
we pull the selected cases from pandapower since the homogeneous models do not always carry that info, 
but for some cases you may need to feed in the case data from matpower

Copyright (c) 2025, Argonne National Laboratory
All rights reserved.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import pandapower.networks as pn

from lumina.dataset.opf.opf_dataset import OPFDataset, OPFHomogeneousDataset
from lumina.model.opf.augmented_lagrangian import AugmentedLagrangianACOPF
from lumina.evaluator.opf.utils import Modeler
from lumina.loader.opf.opf_loader import DataLoader

import pandapower as pp

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate ACOPF model on out-of-sample test cases',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate local checkpoint on out-of-sample case
  python evaluate_out_of_sample.py --checkpoint-dir ./checkpoints/case14 \\
      --test-cases pglib_opf_case30_ieee pglib_opf_case57_ieee

  # Evaluate HuggingFace model on multiple test cases
  python evaluate_out_of_sample.py --repo-id argonne/LUMINA-1B \\
      --hf-token YOUR_TOKEN --test-cases pglib_opf_case118_ieee

  # Evaluate with custom batch size and output to JSON
  python evaluate_out_of_sample.py --checkpoint-dir ./checkpoints \\
      --test-cases case14 case30 --batch-size 16 --output results.json
        """
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        '--checkpoint-dir',
        type=str,
        help='Directory containing config.json and model.safetensors'
    )
    source_group.add_argument(
        '--checkpoint-file',
        type=str,
        help='Path to a single .pt training checkpoint file'
    )
    source_group.add_argument(
        '--repo-id',
        type=str,
        help='HuggingFace repository ID (e.g., argonne/LUMINA-1B)'
    )

    parser.add_argument(
        '--hf-token',
        type=str,
        default=None,
        help='HuggingFace access token (or set HF_TOKEN env var)'
    )

    parser.add_argument(
        '--test-cases',
        type=str,
        nargs='+',
        required=True,
        help='Out-of-sample test case names (e.g., pglib_opf_case30_ieee case57)'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='./opf_data',
        help='Root directory for OPF datasets (default: ./opf_data)'
    )
    parser.add_argument(
        '--group-ids',
        type=int,
        nargs='+',
        default=[0],
        help='Dataset group IDs to test on (default: [0])'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for evaluation (default: 32)'
    )
    parser.add_argument(
        '--max-batches',
        type=int,
        default=None,
        help='Maximum number of batches per case (default: None = all)'
    )
    parser.add_argument(
        '--normalize',
        action='store_true',
        default=True,
        help='Normalize constraint violations (default: True)'
    )
    parser.add_argument(
        '--no-normalize',
        action='store_false',
        dest='normalize',
        help='Disable constraint violation normalization'
    )

    parser.add_argument(
        '--fail-on-missing',
        action='store_true',
        help='Raise error if checkpoint has missing model keys'
    )
    parser.add_argument(
        '--slack-bus-indices',
        type=str,
        default='0',
        help='Comma-separated slack bus indices (default: 0)'
    )
    parser.add_argument(
        '--base-mva',
        type=float,
        default=100.0,
        help='Base MVA for power scaling (default: 100.0)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use for inference (default: auto)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file for results (default: None = stdout only)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print verbose output during evaluation'
    )

    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_dir: str, modeler: Modeler):
    """
    for HF model loading from a local cache
    
    Args:
        checkpoint_dir: Directory containing config.json and model.safetensors
        modeler: Modeler instance to load the model into

    Returns:
        Tuple of (model, config_data)
    """
    checkpoint_path = Path(checkpoint_dir)
    config_path = checkpoint_path / 'config.json'
    safetensors_path = checkpoint_path / 'model.safetensors'

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not safetensors_path.exists():
        raise FileNotFoundError(f"Model file not found: {safetensors_path}")

    print(f"Loading model from checkpoint: {checkpoint_dir}")

    with open(config_path, 'r') as f:
        config_data = json.load(f)

    state_dict = load_file(str(safetensors_path))

    return modeler.load_model(config_data, state_dict)


def load_model_from_hub(repo_id: str, token: str, modeler: Modeler):
    """
    for loading from HuggingFace Hub

    Args:
        repo_id: HuggingFace repository ID
        token: HuggingFace access token
        modeler: Modeler instance to load the model into

    Returns:
        Tuple of (model, config_data)
    """
    print(f"Downloading model from HuggingFace Hub: {repo_id}")

    config_path = hf_hub_download(repo_id=repo_id, filename="config.json", token=token)
    safetensors_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors", token=token)

    with open(config_path, 'r') as f:
        config_data = json.load(f)

    state_dict = load_file(safetensors_path)

    return modeler.load_model(config_data, state_dict)


def load_test_datasets(
    test_cases: List[str],
    data_root: str,
    group_ids: List[int],
    batch_size: int,
    homogeneous: bool = False
) -> dict:
    """
    Args:
        test_cases: List of test case names
        data_root: Root directory for datasets
        group_ids: List of group IDs to load
        batch_size: Batch size for data loaders
        homogeneous: Whether to load homogeneous datasets

    Returns:
        Dictionary mapping case names to data loaders
    """
    test_loaders = {}

    for case_name in test_cases:
        print(f"\nLoading test dataset: {case_name}")

        group_id = group_ids[0] if group_ids else 0

        try:
            dataset_cls = OPFHomogeneousDataset if homogeneous else OPFDataset
            dataset = dataset_cls(
                root=data_root,
                case_name=case_name,
                group_id=group_id
            )

            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False
            )

            test_loaders[case_name] = loader
            print(f"  Loaded {len(dataset)} samples")

        except Exception as e:
            print(f"  Warning: Failed to load {case_name}: {e}")
            continue

    if not test_loaders:
        raise ValueError("No test datasets could be loaded")

    return test_loaders


def to_float32(batch):
    
    if hasattr(batch, 'node_types'):
        for node_type in batch.node_types:
            if getattr(batch[node_type], 'x', None) is not None:
                batch[node_type].x = batch[node_type].x.float()
            if getattr(batch[node_type], 'y', None) is not None:
                batch[node_type].y = batch[node_type].y.float()

        for edge_type in batch.edge_types:
            if getattr(batch[edge_type], 'edge_attr', None) is not None:
                batch[edge_type].edge_attr = batch[edge_type].edge_attr.float()
    else:
        if hasattr(batch, 'x') and batch.x is not None:
            batch.x = batch.x.float()
        if hasattr(batch, 'y') and batch.y is not None:
            batch.y = batch.y.float()
        if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
            batch.edge_attr = batch.edge_attr.float()
    return batch


def predict_batch(modeler: Modeler, batch, minmax_scaling: bool = True):
    batch = to_float32(batch).to(modeler.device)
    is_hetero = hasattr(batch, 'x_dict') and batch.x_dict is not None
    
    if is_hetero:
        try:
            predictions = modeler.model(
                batch.x_dict,
                batch.edge_index_dict,
                batch.edge_attr_dict if hasattr(batch, 'edge_attr_dict') else None,
                minmax_scaling=minmax_scaling,
            )
        except (TypeError, AttributeError):
            predictions = modeler.model(batch, minmax_scaling=minmax_scaling)
    else:
        predictions = modeler.model(batch, minmax_scaling=minmax_scaling)

    predictions_cpu = {}
    if isinstance(predictions, torch.Tensor):
        predictions_cpu['output'] = predictions.detach().cpu()

        if hasattr(batch, 'bus_mask') and hasattr(batch, 'gen_mask'):
            predictions_cpu['bus_mask'] = batch.bus_mask.detach().cpu()
            predictions_cpu['gen_mask'] = batch.gen_mask.detach().cpu()
        elif hasattr(batch, 'ptr') and hasattr(batch, 'x'):
            pass

    elif isinstance(predictions, dict):
        for k, v in predictions.items():
            if isinstance(v, torch.Tensor):
                predictions_cpu[k] = v.detach().cpu()
            else:
                predictions_cpu[k] = v
    else:
        predictions_cpu['output'] = predictions

    batch_cpu = batch.to(torch.device('cpu'))
    return predictions_cpu, batch_cpu


def evaluate_case(
    case_name: str,
    loader: DataLoader,
    modeler: Modeler,
    max_batches: Optional[int],
    normalize: bool
) -> dict:
    """Evaluate model on a single test case."""
    print(f"Evaluating case: {case_name}")

    print("Running predictions")
    pred_batch_pairs = []
    total_batches = None
    try:
        total_batches = len(loader)
    except TypeError:
        total_batches = None
    if total_batches is not None and max_batches is not None:
        total_batches = min(total_batches, max_batches)

    from tqdm import tqdm
    progress_iter = tqdm(loader, total=total_batches, desc="Predicting samples")
    for batch_idx, batch in enumerate(progress_iter):
        preds, batch_cpu = predict_batch(modeler, batch, minmax_scaling=True)
        pred_batch_pairs.append((preds, batch_cpu))
        if max_batches is not None and (batch_idx + 1) >= max_batches:
            break
    progress_iter.close()

    print("Evaluating constraint violations")
    batch_has_node_data = not hasattr(pred_batch_pairs[0][1], 'node_types')

    if batch_has_node_data:
        stats = {}
    else:
        stats = modeler.evaluate_from_predictions(
            pred_batch_pairs,
            normalize=normalize,
            cache_key=case_name
        )

    print("Computing prediction errors")
    prediction_errors = compute_prediction_errors(pred_batch_pairs)

    print("Computing field and node metrics")
    feasibility_metrics = compute_feasibility_metrics(pred_batch_pairs, case_name)

    result = {
        'case_name': case_name,
        'num_batches': len(pred_batch_pairs),
        'constraint_violations': stats,
        'prediction_errors': prediction_errors,
        'feasibility_metrics': feasibility_metrics
    }

    return result


def compute_prediction_errors(pred_batch_pairs: list) -> dict:
    """Compute MSE and MAE between predictions and targets."""
    errors = {}
    all_diffs = []
    for predictions, batch in pred_batch_pairs:

        if hasattr(batch, 'node_types'):
            for node_type in batch.node_types:
                if node_type in predictions and hasattr(batch[node_type], 'y') and batch[node_type].y is not None:
                    pred = predictions[node_type]
                    target = batch[node_type].y
                    if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
                        if pred.shape == target.shape:
                            diff = pred - target
                            if node_type not in errors:
                                errors[node_type] = {'mse': [], 'mae': []}
                            errors[node_type]['mse'].append((diff ** 2).mean().item())
                            errors[node_type]['mae'].append(diff.abs().mean().item())
                            all_diffs.append(diff.flatten())
        else:
            if hasattr(batch, 'y') and batch.y is not None:
                pred = predictions.get('output') if isinstance(predictions, dict) else predictions
                target = batch.y
                if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
                    if pred.shape != target.shape and pred.numel() == target.numel():
                        pred = pred.view(target.shape)
                    if pred.shape == target.shape:
                        diff = pred - target
                        node_type = 'all_nodes'
                        if node_type not in errors:
                            errors[node_type] = {'mse': [], 'mae': []}
                        errors[node_type]['mse'].append((diff ** 2).mean().item())
                        errors[node_type]['mae'].append(diff.abs().mean().item())

    result = {}
    for node_type, error_dict in errors.items():
        if error_dict['mse'] and len(error_dict['mse']) > 0:
            result[node_type] = {
                'mse': sum(error_dict['mse']) / len(error_dict['mse']),
                'mae': sum(error_dict['mae']) / len(error_dict['mae'])
            }
    
    if all_diffs:
        all_diffs_tensor = torch.cat(all_diffs)
        result['all_nodes'] = {
            'mse': (all_diffs_tensor ** 2).mean().item(),
            'mae': all_diffs_tensor.abs().mean().item()
        }
        
    return result


def get_pp_net(case_name: str):


    mapping = {
        'pglib_opf_case14_ieee': pn.case14,
        'case14': pn.case14,
        'pglib_opf_case30_ieee': pn.case30,
        'case30': pn.case30,
        'pglib_opf_case57_ieee': pn.case57,
        'case57': pn.case57,
        'pglib_opf_case118_ieee': pn.case118,
        'case118': pn.case118,
        'pglib_opf_case300_ieee': pn.case300,
        'case300': pn.case300,
        'pglib_opf_case1354_pegase': pn.case1354pegase,
        'case1354': pn.case1354pegase
    }

    if case_name in mapping:
        return mapping[case_name]()
    
    for k, v in mapping.items():
        if k in case_name or case_name in k:
            return v()
            
    import re
    match = re.search(r'case(\d+)', case_name)
    if match:
        func_name = f"case{match.group(1)}"
        if hasattr(pn, func_name):
            return getattr(pn, func_name)()
            
    return None


def from_pp_to_ybus(net):
    # pandapower uses internal indices for Y-bus computation that have lazy initialization
    # this forces a call to initialize them
    try:
        pp.runpp(net, numba=False, max_iteration=1)
    except pp.LoadflowNotConverged:
        pass

    base_mva = net.sn_mva
    Y_bus = net._ppc["internal"]["Ybus"]
    
    return Y_bus


def compute_feasibility_metrics(pred_batch_pairs: list, case_name: str) -> dict:
    """
    compute field-level and node-level error metrics based on physical constraints.
    pandapower is used to fetch network parameters if they're not included in the output
    """

    al = AugmentedLagrangianACOPF()

    net = get_pp_net(case_name)
    if net is None:
        print(f"Warning: Could not load pandapower network for {case_name}. Skipping feasibility metrics.")
        return {}

    # pandapower uses MW/MVAr so we have to rescale for p.u.
    base_mva = net.sn_mva

    # see losses.py or train_vio_lagrangian_case30.ipynb for more metrics calculation details
    # bus limits
    vmin = torch.from_numpy(net.bus.min_vm_pu.values).float()
    vmax = torch.from_numpy(net.bus.max_vm_pu.values).float()
    
    # generator limits (gen + ext_grid)
    p_min_gen = net.gen.min_p_mw.values / base_mva
    p_max_gen = net.gen.max_p_mw.values / base_mva
    q_min_gen = net.gen.min_q_mvar.values / base_mva
    q_max_gen = net.gen.max_q_mvar.values / base_mva
    
    p_min_ext = net.ext_grid.min_p_mw.values / base_mva
    p_max_ext = net.ext_grid.max_p_mw.values / base_mva
    q_min_ext = net.ext_grid.min_q_mvar.values / base_mva
    q_max_ext = net.ext_grid.max_q_mvar.values / base_mva
    
    pmin = torch.from_numpy(np.concatenate([p_min_gen, p_min_ext])).float()
    pmax = torch.from_numpy(np.concatenate([p_max_gen, p_max_ext])).float()
    qmin = torch.from_numpy(np.concatenate([q_min_gen, q_min_ext])).float()
    qmax = torch.from_numpy(np.concatenate([q_max_gen, q_max_ext])).float()
    
    # generator bus indices
    gen_bus_indices = torch.from_numpy(np.concatenate([net.gen.bus.values, net.ext_grid.bus.values])).long()
    
    # demand (aggregated at buses)
    pd_bus = np.zeros(len(net.bus))
    qd_bus = np.zeros(len(net.bus))
    for _, load in net.load.iterrows():
        pd_bus[load.bus] += load.p_mw / base_mva
        qd_bus[load.bus] += load.q_mvar / base_mva
    pd_static = torch.from_numpy(pd_bus).float()
    qd_static = torch.from_numpy(qd_bus).float()
    
    # load indices for AL
    load_bus_indices = torch.arange(len(net.bus)).long()

    field_violations = {
        'p_balance': [],
        'q_balance': [],
        'vm_limits': [],
        'pg_limits': [],
        'qg_limits': []
    }
    
    node_violations = {
        'bus_error': [],
        'gen_error': []
    }

    for predictions, batch in pred_batch_pairs:
        is_hetero = hasattr(batch, 'node_types')
        
        batch_size = 1
        if is_hetero:
            if hasattr(batch['bus'], 'batch'):
                batch_size = batch['bus'].batch.max().item() + 1
        else:
            if hasattr(batch, 'batch'):
                batch_size = batch.batch.max().item() + 1

        for i in range(batch_size):
            try:
                # in multi-case evaluation we try to fetch Y matrices from the current sample since there may be multiple topologies

                samples = []
                if hasattr(batch, 'to_data_list'):
                    try:
                        samples = batch.to_data_list()
                    except Exception:
                        samples = [batch]
                else:
                    samples = [batch]
                
                sample = samples[i]

                if is_hetero:
                    sample_preds = {
                        'bus': predictions['bus'][batch['bus'].batch == i] if hasattr(batch['bus'], 'batch') else predictions['bus'],
                        'generator': predictions['generator'][batch['generator'].batch == i] if hasattr(batch['generator'], 'batch') else predictions['generator']
                    }
                else:
                    # in the homo model path, output does not have node type masks
                    if 'bus' in predictions and 'generator' in predictions:
                         sample_preds = {
                            'bus': predictions['bus'][i] if predictions['bus'].dim() > 2 else predictions['bus'],
                            'generator': predictions['generator'][i] if predictions['generator'].dim() > 2 else predictions['generator']
                         }
                    elif 'bus_mask' in predictions and 'gen_mask' in predictions and 'output' in predictions:
                        raw_preds = predictions['output']
                        bus_mask = predictions['bus_mask']
                        gen_mask = predictions['gen_mask']

                        if hasattr(sample, 'bus_mask') and hasattr(sample, 'gen_mask'):
                            sample_preds = {
                                'bus': raw_preds[bus_mask][i*len(net.bus) : (i+1)*len(net.bus)],
                                'generator': raw_preds[gen_mask][i*(len(net.gen)+len(net.ext_grid)) : (i+1)*(len(net.gen)+len(net.ext_grid))]
                            }
                        else:
                            sample_preds = {
                                'bus': raw_preds[bus_mask][i*len(net.bus) : (i+1)*len(net.bus)],
                                'generator': raw_preds[gen_mask][i*(len(net.gen)+len(net.ext_grid)) : (i+1)*(len(net.gen)+len(net.ext_grid))]
                            }
                    else:
                        continue
                
                if sample_preds['bus'].numel() == 0:
                    continue
                
                device = sample_preds['bus'].device

                # network parameters for AL
                if hasattr(sample, 'Y_real_sparse'):
                    al.set_network_parameters(
                        Y_real_sparse=sample.Y_real_sparse.to(device),
                        Y_imag_sparse=sample.Y_imag_sparse.to(device),
                        Y_diag_real=getattr(sample, 'Y_diag_real', None),
                        Y_diag_imag=getattr(sample, 'Y_diag_imag', None),
                    )
                else:
                    # recompute Y from pandapower net if not in sample
                    Y_bus = from_pp_to_ybus(net)
                    Y_bus_sparse = Y_bus.tocoo()
                    indices = torch.from_numpy(np.vstack((Y_bus_sparse.row, Y_bus_sparse.col))).long()
                    values_real = torch.from_numpy(Y_bus_sparse.data.real).float()
                    values_imag = torch.from_numpy(Y_bus_sparse.data.imag).float()
                    
                    Y_real_sparse = torch.sparse_coo_tensor(indices, values_real, (len(net.bus), len(net.bus))).to(device)
                    Y_imag_sparse = torch.sparse_coo_tensor(indices, values_imag, (len(net.bus), len(net.bus))).to(device)
                    
                    al.set_network_parameters(
                        Y_real_sparse=Y_real_sparse,
                        Y_imag_sparse=Y_imag_sparse
                    )

                # violations
                va = sample_preds['bus'][..., 0].float().to(device)
                vm = sample_preds['bus'][..., 1].float().to(device)
                pg = sample_preds['generator'][..., 0].float().to(device)
                qg = sample_preds['generator'][..., 1].float().to(device)
                
                if torch.isnan(va).any() or torch.isnan(vm).any() or torch.isnan(pg).any() or torch.isnan(qg).any():
                    continue

                # static data
                pd_val = pd_static.to(device)
                qd_val = qd_static.to(device)
                gen_idx = gen_bus_indices.to(device)
                load_idx = load_bus_indices.to(device)

                if vm.shape[0] != len(net.bus) or pg.shape[0] != (len(net.gen) + len(net.ext_grid)):
                    continue

                all_pf_mis = al.compute_power_flow_constraints(
                    vm, va, pg, qg, pd_val, qd_val, gen_idx, load_idx
                )
                n_bus = vm.shape[-1]
                p_bal_val = all_pf_mis[:n_bus].abs().mean().item()
                q_bal_val = all_pf_mis[n_bus:].abs().mean().item()

                def get_vio_mean(val, vmin_t, vmax_t):
                    vio = torch.zeros_like(val)
                    if vmin_t is not None:
                        vio = torch.maximum(vio, vmin_t.to(device) - val)
                    if vmax_t is not None:
                        vio = torch.maximum(vio, val - vmax_t.to(device))
                    return vio.abs().mean().item()

                pg_lim_val = get_vio_mean(pg, pmin, pmax)
                qg_lim_val = get_vio_mean(qg, qmin, qmax)
                vm_lim_val = get_vio_mean(vm, vmin, vmax)

                field_violations['p_balance'].append(p_bal_val)
                field_violations['q_balance'].append(q_bal_val)
                field_violations['vm_limits'].append(vm_lim_val)
                field_violations['pg_limits'].append(pg_lim_val)
                field_violations['qg_limits'].append(qg_lim_val)

                node_violations['bus_error'].append((p_bal_val + q_bal_val + vm_lim_val) / 3.0)
                node_violations['gen_error'].append((pg_lim_val + qg_lim_val) / 2.0)

            except Exception:
                continue

    return {k: sum(v) / len(v) for k, v in {**field_violations, **node_violations}.items() if v}

def print_summary(results: List[dict]):
    print(f"\n{'='*80}")
    print(f"Summary Across All Test Cases")
    print(f"{'='*80}")

    for result in results:
        print(f"\n{result['case_name']}:")

        violations = result['constraint_violations']
        key_metrics = [
            'bound_total_bound_violations',
            'real_power_flow_violations',
            'reactive_power_flow_violations',
            'line_flow_violations'
        ]

        for metric in key_metrics:
            if metric in violations:
                mean = violations[metric].get('mean', 0.0)
                print(f"  {metric:40s}: {mean:.6e}")

        errors = result['prediction_errors']
        for node_type in ['bus', 'generator']:
            if node_type in errors:
                mae = errors[node_type]['mae']
                print(f"  {node_type}_mae:                              {mae:.6e}")


def save_results(results: List[dict], output_path: str):
    print(f"\nSaving results to {output_path}")

    json_results = []
    for result in results:
        json_result = {
            'case_name': result['case_name'],
            'num_batches': result['num_batches'],
            'constraint_violations': {},
            'prediction_errors': result['prediction_errors'],
            'feasibility_metrics': result.get('feasibility_metrics', {})
        }

        for key, value in result['constraint_violations'].items():
            json_result['constraint_violations'][key] = {
                'mean': float(value.get('mean', 0.0)),
                'var': float(value.get('var', 0.0)),
                'weight': float(value.get('weight', 0.0))
            }

        json_results.append(json_result)

    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"Results saved successfully")


def main():
    args = parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    modeler = Modeler(
        device=device,
        fail_on_missing=args.fail_on_missing,
        verbose=args.verbose,
        base_mva=args.base_mva,
        slack_bus_indices=args.slack_bus_indices
    )

    if args.checkpoint_dir:
        model, config = load_model_from_checkpoint(args.checkpoint_dir, modeler)
    elif args.checkpoint_file:
        model = modeler.load_model_from_training_checkpoint(args.checkpoint_file)
    else:
        token = args.hf_token or os.getenv('HF_TOKEN')
        if not token:
            raise ValueError(
                "HuggingFace token required. Use --hf-token or set HF_TOKEN env var"
            )
        model, config = load_model_from_hub(args.repo_id, token, modeler)

    print("Model loaded successfully")

    model_type = model.__class__.__name__
    if not args.output:
        args.output = f"results_{model_type}.json"
        print(f"Defaulting output to: {args.output}")

    homogeneous = False
    if args.checkpoint_file:
        try:
            checkpoint = torch.load(args.checkpoint_file, map_location='cpu')
            model_class = checkpoint.get('model_class', '')
            if any(h in model_class for h in ['GAT', 'GCN', 'GIN', 'Transformer', 'TRANSFORMER']):
                homogeneous = True
                print(f"Detected homogeneous model class: {model_class}")
        except Exception:
            pass

    test_loaders = load_test_datasets(
        args.test_cases,
        args.data_root,
        args.group_ids,
        args.batch_size,
        homogeneous=homogeneous
    )

    results = []
    for case_name, loader in test_loaders.items():
        try:
            result = evaluate_case(
                case_name,
                loader,
                modeler,
                args.max_batches,
                args.normalize
            )
            results.append(result)
        except Exception as e:
            print(f"\nError evaluating {case_name}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            continue

    if results:
        print_summary(results)
        if args.output:
            save_results(results, args.output)
    else:
        print("\nNo results generated")

    print(f"\n{'='*80}")
    print("Evaluation complete")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
