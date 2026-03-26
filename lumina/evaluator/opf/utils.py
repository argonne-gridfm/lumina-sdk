"""
Utility functions for working with physics-informed losses in ACOPF problems.

This module provides helper functions to extract network parameters from OPF datasets
and create data structures compatible with the physics loss functions.

Copyright (c) 2025, Argonne National Laboratory
All rights reserved.
"""

import ast
import importlib
import re
import warnings
from typing import Dict, Iterable, List, Optional, Tuple, Union, Any

import torch
from tqdm import tqdm

from lumina.evaluator.opf.evaluator import ACOPFConstraintEvaluator
from lumina.trainer.opf.utils import build_hetero_model_spec, resolve_hetero_model_type

_LINE_CACHE = {}


def extract_network_parameters_from_batch(batch, device: torch.device = None) -> Dict:
    """
    Extract network parameters from a batch of OPF data.

    Args:
        batch: Batch from OPFDataset containing heterogeneous graph data
        device: Target device for tensors

    Returns:
        Dictionary containing extracted network parameters
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    extracted_data = {}

    try:
        # Extract load data (pd, qd)
        if 'load' in batch.x_dict:
            load_data = batch['load'].x  # Shape: [n_loads, features]
            extracted_data['pd'] = load_data[:, 0].to(device)  # Active power demand
            extracted_data['qd'] = load_data[:, 1].to(device)  # Reactive power demand

            # Get load bus indices from edge connections
            if ('load', 'load_link', 'bus') in batch.edge_index_dict:
                # Load to bus connections - get bus indices
                load_bus_edges = batch[('load', 'load_link', 'bus')].edge_index
                extracted_data['load_bus_indices'] = load_bus_edges[1, :].to(device)  # Bus indices
            elif ('bus', 'load_link', 'load') in batch.edge_index_dict:
                # Bus to load connections - get bus indices
                bus_load_edges = batch[('bus', 'load_link', 'load')].edge_index
                extracted_data['load_bus_indices'] = bus_load_edges[0, :].to(device)  # Bus indices

        # Extract generator bus indices
        if ('generator', 'generator_link', 'bus') in batch.edge_index_dict:
            gen_bus_edges = batch[('generator', 'generator_link', 'bus')].edge_index
            extracted_data['gen_bus_indices'] = gen_bus_edges[1, :].to(device)  # Bus indices
        elif ('bus', 'generator_link', 'generator') in batch.edge_index_dict:
            bus_gen_edges = batch[('bus', 'generator_link', 'generator')].edge_index
            extracted_data['gen_bus_indices'] = bus_gen_edges[0, :].to(device)  # Bus indices

        # Extract line edge indices for thermal limits
        if ('bus', 'ac_line', 'bus') in batch.edge_index_dict:
            line_edges = batch[('bus', 'ac_line', 'bus')].edge_index
            extracted_data['line_edge_index'] = line_edges.to(device)

            # Extract line limits from edge attributes if available
            if hasattr(batch[('bus', 'ac_line', 'bus')], 'edge_attr'):
                line_attr = batch[('bus', 'ac_line', 'bus')].edge_attr
                if line_attr.size(1) > 6:  # Assuming thermal limit is 7th column (index 6)
                    extracted_data['line_limits'] = line_attr[:, 6].to(device)

    except Exception as e:
        warnings.warn(f"Error extracting network parameters from batch: {e}")

    return extracted_data


def extract_voltage_and_generation_limits_from_batch(batch, device: torch.device = None) -> Tuple[Dict, Dict]:
    """
    Extract voltage and generation limits from batch data.

    Args:
        batch: Batch from OPFDataset
        device: Target device for tensors

    Returns:
        Tuple of (voltage_limits dict, generation_limits dict)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    voltage_limits = {}
    generation_limits = {}

    try:
        # Extract voltage limits from bus data
        if 'bus' in batch.x_dict:
            bus_data = batch['bus'].x
            # Assuming bus features include [base_kv, vmin, vmax, bus_type_onehot...]
            if bus_data.size(1) >= 3:
                voltage_limits['vmin'] = bus_data[:, 1].to(device)
                voltage_limits['vmax'] = bus_data[:, 2].to(device)

        # Extract generation limits from generator data
        if 'generator' in batch.x_dict:
            gen_data = batch['generator'].x
            # Assuming generator features include [mbase, pg, pmin, pmax, qg, qmin, qmax, vg, costs...]
            if gen_data.size(1) >= 7:
                generation_limits['pmin'] = gen_data[:, 2].to(device)
                generation_limits['pmax'] = gen_data[:, 3].to(device)
                generation_limits['qmin'] = gen_data[:, 5].to(device)
                generation_limits['qmax'] = gen_data[:, 6].to(device)

    except Exception as e:
        warnings.warn(f"Error extracting limits from batch: {e}")

    return voltage_limits, generation_limits


def extract_generation_costs_from_batch(batch, device: torch.device = None) -> Optional[torch.Tensor]:
    """
    Extract generation cost coefficients from batch data.

    Args:
        batch: Batch from OPFDataset
        device: Target device for tensors

    Returns:
        Tensor of generation cost coefficients or None
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        if 'generator' in batch.x_dict:
            gen_data = batch['generator'].x
            # Assuming cost coefficients are the last 3 features
            if gen_data.size(1) >= 3:
                return gen_data[:, -3:].to(device)

    except Exception as e:
        warnings.warn(f"Error extracting generation costs from batch: {e}")

    return None


def denormalize_predictions(
    predictions: Dict[str, torch.Tensor],
    batch,
    voltage_range: Tuple[float, float] = (0.95, 1.05),
    angle_range: Tuple[float, float] = (-180, 180),
    power_range: Tuple[float, float] = (0, 100)  # Will be replaced by actual limits
) -> Dict[str, torch.Tensor]:
    """
    Denormalize model predictions from [0,1] range to physical units.

    Args:
        predictions: Normalized predictions from model
        batch: Batch containing limit information
        voltage_range: Default voltage magnitude range (per unit)
        angle_range: Default voltage angle range (degrees)
        power_range: Default power range (MW/MVAr)

    Returns:
        Denormalized predictions in physical units
    """
    denorm_predictions = {}

    # Denormalize bus predictions (voltage magnitude and angle)
    if 'bus' in predictions:
        bus_pred = predictions['bus'].clone()

        # Extract actual limits from batch if available
        if 'bus' in batch.x_dict and batch['bus'].x.size(1) >= 3:
            vmin = batch['bus'].x[:, 1]
            vmax = batch['bus'].x[:, 2]

            # Denormalize voltage magnitude
            vm_denorm = bus_pred[..., 0] * (vmax - vmin) + vmin

        else:
            # Use default range
            vm_denorm = bus_pred[..., 0] * (voltage_range[1] - voltage_range[0]) + voltage_range[0]

        # Denormalize voltage angle
        va_denorm = bus_pred[..., 1] * (angle_range[1] - angle_range[0]) + angle_range[0]

        denorm_predictions['bus'] = torch.stack([vm_denorm, va_denorm], dim=-1)

    # Denormalize generator predictions (active and reactive power)
    if 'generator' in predictions:
        gen_pred = predictions['generator'].clone()

        # Extract actual limits from batch if available
        if 'generator' in batch.x_dict and batch['generator'].x.size(1) >= 7:
            pmin = batch['generator'].x[:, 2]
            pmax = batch['generator'].x[:, 3]
            qmin = batch['generator'].x[:, 5]
            qmax = batch['generator'].x[:, 6]

            # Denormalize active power
            pg_denorm = gen_pred[..., 0] * (pmax - pmin) + pmin

            # Denormalize reactive power
            qg_denorm = gen_pred[..., 1] * (qmax - qmin) + qmin

        else:
            # Use default range
            pg_denorm = gen_pred[..., 0] * (power_range[1] - power_range[0]) + power_range[0]
            qg_denorm = gen_pred[..., 1] * (power_range[1] - power_range[0]) + power_range[0]

        denorm_predictions['generator'] = torch.stack([pg_denorm, qg_denorm], dim=-1)

    return denorm_predictions


class Modeler:
    """
    Modeler wraps model loading, prediction, and evaluation logic for OPF tasks.

    This class serves as user-friendly wrapper to encapsulate the usage of lumina trained models including:
    Configuration setup, loading Weights, run batch predictions, and run evaluation on saved predictions.

    Args:
        device (torch.device): Device to run model inference on (e.g., "cpu" or "cuda").
        fail_on_missing (bool, optional): If True, raise when expected model keys
            are missing during checkpoint load. Defaults to False.
        verbose (bool, optional): If True, print diagnostic messages during
            checkpoint loading. Defaults to True.
        base_mva (float, optional): Base MVA used by the evaluator. Defaults to 100.0.
        slack_bus_indices (str, optional): Comma-separated slack bus indices
            (default: "0"). Converted to a list of ints and stored on the instance.

    Attributes:
        device (torch.device): See Args.
        fail_on_missing (bool): See Args.
        verbose (bool): See Args.
        base_mva (float): See Args.
        slack_bus_indices (List[int]): Slack bus indices parsed from the
            `slack_bus_indices` constructor argument.
        model (Optional[torch.nn.Module]): Loaded and prepared model (None until `load_model` is called).
        config_data (Optional[dict]): Parsed model configuration loaded during `load_model`.

    Example:
        >>> modeler = Modeler(torch.device("cpu"), slack_bus_indices="0,1")
        >>> config = json.load(open("config.json"))
        >>> state_dict = load_file("model.safetensors")
        >>> modeler.load_model(config, state_dict)
        >>> loader = DataLoader(OPFDataset(root="./opf_data", case_name="pglib_opf_case14_ieee"), batch_size=1)
        >>> preds = modeler.run_predictions(loader)
        >>> stats = modeler.evaluate_from_predictions(preds, cache_key="pglib_opf_case14_ieee")
    """

    def __init__(
        self,
        device: torch.device,
        *,
        fail_on_missing: bool = False,
        verbose: bool = True,
        base_mva: float = 100.0,
        slack_bus_indices: str = "0",
    ):
        self.device = device
        self.fail_on_missing = fail_on_missing
        self.verbose = verbose
        self.base_mva = base_mva
        self.slack_bus_indices = [int(x) for x in slack_bus_indices.split(",") if x.strip() != ""]
        self.model: Optional[torch.nn.Module] = None
        self.config_data = None

    # -- checkpoint key conversion and loading -----------------------------------------------------------------
    @staticmethod
    def convert_checkpoint_key_to_model_key(key: str) -> str:
        """
        Convert checkpoint keys to model keys by transforming
        underscore-delimited items to tuple string representation.

        Args:
            key: Current key with triple underscore delimiters inside angle brackets

        Returns:
            String with angle bracket contents converted to tuple representation

        Example:
            >>> Modeler.convert_checkpoint_key_to_model_key("<bus___ac_line___weight>")
            "('bus', 'ac_line', 'weight')"
        """

        pattern = r"<([^>]+)>"

        def replacer(match):
            parts = match.group(1).split('___')
            return f"('{parts[0]}', '{parts[1]}', '{parts[2]}')"

        return re.sub(pattern, replacer, key)

    def load_checkpoint_into_model(
        self,
        model: torch.nn.Module,
        checkpoint_dict,
        *,
        fail_on_missing: bool = False,
        verbose: bool = True,
    ):
        """
        Load a checkpoint dictionary into a model and report missing/unexpected keys.

        This method remaps checkpoint keys to model keys using
        `convert_checkpoint_key_to_model_key` and then calls `load_state_dict`
        with `strict=False` to allow partial loads.

        Args:
            model (torch.nn.Module): The model to populate.
            checkpoint_dict (dict): Mapping of checkpoint keys to tensors.
            fail_on_missing (bool, optional): If True, raise ValueError when
                missing keys remain after the load. Defaults to False.
            verbose (bool, optional): If True, print missing/unexpected keys.

        Returns:
            dict: A dictionary with keys "missing_keys" and "unexpected_keys",
                each mapping to a list of key names observed.

        Raises:
            ValueError: If `fail_on_missing` is True and missing keys are found.

        Example:
            >>> result = modeler._load_checkpoint_into_model(model, ckpt_dict)
            >>> print(result["missing_keys"])
        """

        model_state = model.state_dict()
        used_keys = set()
        missing_keys = []

        remapped_state = {}
        for model_key in model_state.keys():
            ck = self.convert_checkpoint_key_to_model_key(model_key)
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

    # -- model loading ---------------------------------------------------------------------------------------
    def load_model(self, config_data: dict, state_dict: dict):
        """
        Construct a hetero OPF model from provided configuration and state dict.

        Note:
            Downloads and file I/O for the configuration and safetensors are
            expected to be performed outside this method; the parsed `config_data`
            and in-memory `state_dict` should be passed here.

        Args:
            config_data (dict): Parsed JSON configuration describing model metadata
                and architecture.
            state_dict (dict): Raw state dictionary as returned by `safetensors.torch.load_file`.

        Returns:
            Tuple[torch.nn.Module, dict]: The constructed model (in eval mode)
            and the config_data used to build it.

        Raises:
            ValueError: If `fail_on_missing` is True and required keys are missing
                from the checkpoint (raised from `_load_checkpoint_into_model`).

        Example:
            >>> config = json.load(open("config.json"))
            >>> state = load_file("model.safetensors")
            >>> model, cfg = modeler.load_model(config, state)
        """
        # Convert metadata edge keys from strings to tuples if needed
        if 'edges' in config_data.get('metadata', {}):
            edges_dict = {}
            for key, value in config_data['metadata']['edges'].items():
                if isinstance(key, str) and key.startswith('('):
                    key = ast.literal_eval(key)
                edges_dict[key] = value
            config_data['metadata']['edges'] = edges_dict

        model_type = resolve_hetero_model_type(
            model_type=config_data.get("model"),
            model_class_path=config_data.get("model_class"),
            default="HeteroGNN",
        )
        model_class, model_kwargs, _, used_fallback = build_hetero_model_spec(
            model_type=model_type,
            metadata=config_data["metadata"],
            input_channels=config_data["input_channels"],
            models_config=config_data.get("config", {}).get("models", {}),
            out_channels=config_data.get("out_channels", 2),
        )
        if used_fallback and self.verbose:
            print(f"[MODEL LOAD] Config for {model_type} not found; using HeteroGNN config.")

        model = model_class(**model_kwargs).to(self.device)

        # state_dict is the raw output of safetensors.load_file; remap its keys
        checkpoint_dict = {self.convert_checkpoint_key_to_model_key(k): v for k, v in state_dict.items()}

        self.load_checkpoint_into_model(
            model,
            checkpoint_dict,
            fail_on_missing=self.fail_on_missing,
            verbose=self.verbose,
        )

        model.eval()
        self.model = model
        self.config_data = config_data
        return model, config_data


    def load_model_from_training_checkpoint(self,
            ckpt_path: Union[str, "os.PathLike[str]"],
            *,
            strict: bool = True,
    ) -> torch.nn.Module:
        """
        training checkpoint formats differ slightly from HF safetensor serialization
        minimal checkpoint keys for self-contained checkpoints:
          - model_class: str
              Fully-qualified class name, e.g.:
                "lumina.model.opf.hetero_model.HGT"
          - model_kwargs: dict
              Keyword arguments to reconstruct the model __init__(**model_kwargs)
              (e.g., metadata/input_channels/hidden_channels/... for hetero models).
          - model_state_dict: dict[str, Tensor]
              Weights.

        Returns:
          torch.nn.Module (not DDP wrapped), moved to `device` if provided.
        """
        ckpt: Dict[str, Any] = torch.load(ckpt_path, map_location=self.device)
        class_path = ckpt.get("model_class")

        model_kwargs = ckpt.get("model_kwargs", {})
        state_dict = ckpt.get("model_state")

        if state_dict is None:
            state_dict = ckpt.get("model_state_dict")

        normalized_state_dict = {key.replace('module.', ''): val for key, val in state_dict.items()}

        # N.b. we should switch to using a model registry

        module_name, _, cls_name = class_path.rpartition(".")

        if not module_name:
            raise ValueError(
                f"Invalid model class in checkpoint: '{class_path}'. Expected fully-qualified path like 'pkg.module.ClassName'."
            )

        module = importlib.import_module(module_name)
        cls = getattr(module, cls_name)

        model: torch.nn.Module = cls(**model_kwargs)
        model.load_state_dict(normalized_state_dict, strict=strict)
        model = model.to(torch.device(self.device))

        model.eval()
        self.model = model
        return model


    # -- helpers for tensors --------------------------------------------------------------------------------
    @staticmethod
    def to_float32(batch):
        """
        Convert node features/targets and edge attributes to float32.

        Iterates through all node types and edge types in the batch,
        converting 'x' (features), 'y' (targets), and 'edge_attr' tensors
        to float32 precision.

        Args:
            batch: A batch data object from DataLoader containing node_types
                   and edge_types attributes

        Returns:
            The same batch object with `.x`, `.y` and `.edge_attr` converted to float32
            where present.

        Note:
            Modifies the input batch in-place and returns it.
            Assumes batch has 'node_types' and 'edge_types' properties,
            typical of heterogeneous graph data structures.

        Example:
            >>> batch = modeler.to_float32(batch)
        """
        for node_type in batch.node_types:
            if getattr(batch[node_type], 'x', None) is not None:
                batch[node_type].x = batch[node_type].x.float()
            if getattr(batch[node_type], 'y', None) is not None:
                batch[node_type].y = batch[node_type].y.float()

        for edge_type in batch.edge_types:
            if getattr(batch[edge_type], 'edge_attr', None) is not None:
                batch[edge_type].edge_attr = batch[edge_type].edge_attr.float()

        return batch

    # -- limit/params derivation -------------------------------------------------------------------------------
    @staticmethod
    def derive_voltage_limits(bus_x: torch.Tensor, device: torch.device):
        """
        Derive per-bus voltage limits from bus feature matrix.

        If bus feature tensor contains columns for vmin/vmax (columns 1 and 2),
        those are used; otherwise sensible defaults (0.95/1.05) are returned.

        Args:
            bus_x (torch.Tensor): Bus node feature matrix or None.
            device (torch.device): Device on which returned tensors should be allocated.

        Returns:
            dict: Dictionary with keys 'vmin' and 'vmax' mapping to 1-D tensors of length n_bus.

        Example:
            >>> vlims = Modeler.derive_voltage_limits(bus_x, torch.device("cpu"))
            >>> vlims['vmin'].shape
        """
        if bus_x is not None and bus_x.size(1) >= 3:
            vmin = bus_x[:, 1].to(device)
            vmax = bus_x[:, 2].to(device)
            return {'vmin': vmin, 'vmax': vmax}
        n_bus = bus_x.size(0) if bus_x is not None else 0
        return {
            'vmin': torch.full((n_bus,), 0.95, device=device),
            'vmax': torch.full((n_bus,), 1.05, device=device),
        }

    @staticmethod
    def derive_generation_limits(gen_x: torch.Tensor, device: torch.device):
        """
        Derive generator P/Q limits from generator feature matrix.

        Heuristic mapping based on dataset feature layout:
          - pmin: column 2
          - pmax: column 3
          - qmin: column 5
          - qmax: column 6

        Args:
            gen_x (torch.Tensor): Generator node feature matrix or None.
            device (torch.device): Device for output tensors.

        Returns:
            Optional[dict]: Dictionary with keys 'pmin', 'pmax', 'qmin', 'qmax' mapping
                to 1-D tensors of length n_gen, or None if gen_x is None/empty.

        Example:
            >>> glims = Modeler.derive_generation_limits(gen_x, torch.device("cpu"))
            >>> glims['pmax'].shape
        """
        if gen_x is None or gen_x.numel() == 0:
            return None

        n_gen = gen_x.size(0)

        def col_or_default(idx: int, default: float):
            return gen_x[:, idx].to(device) if gen_x.size(1) > idx else torch.full((n_gen,), default, device=device)

        pmin = col_or_default(2, 0.0)
        pmax = col_or_default(3, 2.0)
        qmin = col_or_default(5, -1.0)
        qmax = col_or_default(6, 1.0)

        return {'pmin': pmin, 'pmax': pmax, 'qmin': qmin, 'qmax': qmax}

    def derive_line_params(self, batch, device: torch.device, cache_key: str = None):
        """
        Build line limits and dense admittance matrices (Y_real, Y_imag) from ac_line edges.

        The method reads the ('bus', 'ac_line', 'bus') edge type attributes and
        computes the per-line thermal limits and the dense Y matrix for the network.
        Results are cached in the module-level `_LINE_CACHE` when a cache_key is provided.

        Args:
            batch: Batched data object containing 'ac_line' edge attributes.
            device (torch.device): Device for intermediate tensors.
            cache_key (str, optional): Key to use for caching results in `_LINE_CACHE`.

        Returns:
            Tuple[torch.Tensor or None, torch.Tensor or None, torch.Tensor or None, torch.Tensor or None]:
                (line_limits, Y_real, Y_imag, edge_index) where any element may be None
                if required data is not present in `batch`.

        Example:
            >>> line_limits, Yr, Yi, idx = modeler.derive_line_params(batch, torch.device("cpu"))
        """
        global _LINE_CACHE
        if cache_key and cache_key in _LINE_CACHE:
            return _LINE_CACHE[cache_key]

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

        result = (line_limits, Y_real, Y_imag, edge_index)
        if cache_key:
            _LINE_CACHE[cache_key] = result
        return result

    # -- evaluator construction ------------------------------------------------------------------------------
    def build_constraint_evaluator(self, batch, device: torch.device, cache_key: str = None):
        """
        Build an ACOPFConstraintEvaluator using limits derived from the dataset.

        Args:
            batch: Batched data object containing node/edge features required by the evaluator.
            device (torch.device): Device on which evaluator tensors will be placed.
            cache_key (str, optional): Cache key passed to `derive_line_params` to enable reuse.

        Returns:
            ACOPFConstraintEvaluator: Configured evaluator instance ready to run constraint checks.

        Example:
            >>> evaluator = modeler.build_constraint_evaluator(batch, torch.device("cpu"), cache_key="case14")
        """
        bus_x = batch['bus'].x if hasattr(batch['bus'], 'x') else None
        gen_x = batch['generator'].x if 'generator' in batch.node_types and hasattr(batch['generator'], 'x') else None

        voltage_limits = self.derive_voltage_limits(bus_x, device)
        generation_limits = self.derive_generation_limits(gen_x, device)

        line_limits, Y_real, Y_imag, edge_index = self.derive_line_params(batch, device, cache_key=cache_key)

        return ACOPFConstraintEvaluator(
            voltage_limits=voltage_limits,
            generation_limits=generation_limits,
            line_limits=line_limits,
            Y_real=Y_real,
            Y_imag=Y_imag,
            edge_index=edge_index,
            base_mva=self.base_mva,
            device=device,
        )

    # -- prediction and evaluation stages --------------------------------------------------------------------
    def predict_batch(self, batch, minmax_scaling: bool = True):
        """
        Run a forward pass on a single batch and return predictions (on CPU) with the batch (on CPU).

        Args:
            batch: Batched data object containing inputs for the model.
            minmax_scaling (bool, optional): Whether to apply min-max scaling in the model's forward pass.

        Returns:
            Tuple[dict, object]: A tuple (predictions_cpu, batch_cpu) where `predictions_cpu` maps output
            names to CPU tensors and `batch_cpu` is the input batch moved to CPU.

        Raises:
            RuntimeError: If the model has not been loaded via `load_model()`.

        Example:
            >>> preds, batch_cpu = modeler.predict_batch(batch)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        batch = self.to_float32(batch).to(self.device)

        predictions = self.model(
            batch.x_dict,
            batch.edge_index_dict,
            batch.edge_attr_dict if hasattr(batch, 'edge_attr_dict') else None,
            minmax_scaling=minmax_scaling,
        )

        # Move predictions to CPU and detach to allow storing
        predictions_cpu = {}
        for k, v in predictions.items():
            if isinstance(v, torch.Tensor):
                predictions_cpu[k] = v.detach().cpu()
            else:
                predictions_cpu[k] = v

        # Move batch to CPU for later evaluation/storage. Keep a copy (not on device).
        batch_cpu = batch.to(torch.device('cpu'))
        return predictions_cpu, batch_cpu

    def run_predictions(self, loader: Iterable, max_batches: Optional[int] = None, minmax_scaling: bool = True):
        """
        Run predictions over a data loader and return collected prediction/batch pairs.

        This method separates the forward pass from evaluation so predictions can
        be stored or evaluated later (e.g., on CPU-only machines).

        Args:
            loader (Iterable): Iterable data loader yielding batches.
            max_batches (Optional[int], optional): Limit on number of batches to process. Defaults to None (process all).
            minmax_scaling (bool, optional): Passed to `predict_batch`. Defaults to True.

        Returns:
            List[Tuple[dict, object]]: List of (predictions_cpu, batch_cpu) tuples.

        Raises:
            RuntimeError: If the model has not been loaded via `load_model()`.

        Example:
            >>> pairs = modeler.run_predictions(loader, max_batches=10)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        pred_batch_pairs: List[Tuple[dict, object]] = []
        total_batches = None
        try:
            total_batches = len(loader)
        except TypeError:
            total_batches = None
        if total_batches is not None and max_batches is not None:
            total_batches = min(total_batches, max_batches)

        progress_iter = tqdm(loader, total=total_batches, desc="Predicting samples")
        for batch_idx, batch in enumerate(progress_iter):
            preds, batch_cpu = self.predict_batch(batch, minmax_scaling=minmax_scaling)
            pred_batch_pairs.append((preds, batch_cpu))

            progress_iter.set_postfix(predictions=len(pred_batch_pairs), refresh=False)
            if max_batches is not None and (batch_idx + 1) >= max_batches:
                progress_iter.write(f"Reached max_batches={max_batches}.")
                break
        progress_iter.close()
        return pred_batch_pairs

    def evaluate_from_predictions(
        self,
        pred_batch_pairs: List[Tuple[dict, object]],
        normalize: bool = True,
        cache_key: Optional[str] = None,
    ):
        """
        Evaluate constraints using previously computed predictions and their corresponding batches.

        Args:
            pred_batch_pairs (List[Tuple[dict, object]]): List of (predictions_cpu, batch_cpu) tuples
                produced by `run_predictions`.
            normalize (bool, optional): Whether to normalize violations in the evaluator. Defaults to True.
            cache_key (Optional[str], optional): Cache key to pass to `derive_line_params` for reusing line matrices.

        Returns:
            dict: Aggregated statistics keyed by violation name. Each value is a dict with keys:
                - 'mean' (float): Weighted mean violation
                - 'var' (float): Weighted variance of the violation
                - 'weight' (float): Total sample weight used for aggregation

        Raises:
            ValueError: If `pred_batch_pairs` is empty.

        Example:
            >>> stats = modeler.evaluate_from_predictions(pred_batch_pairs, cache_key="case14")
        """
        if len(pred_batch_pairs) == 0:
            raise ValueError("No predictions provided for evaluation.")

        # Run evaluation on CPU to avoid requiring GPU at evaluation time
        eval_device = torch.device('cpu')

        accum_sum = {}
        accum_sq = {}
        accum_weight = {}
        batches_seen = 0

        progress_iter = tqdm(pred_batch_pairs, desc="Evaluating predictions")
        for predictions, batch in progress_iter:
            # Batch is already on CPU; ensure types
            batch = self.to_float32(batch).to(eval_device)

            evaluator = self.build_constraint_evaluator(batch, device=eval_device, cache_key=cache_key)
            evaluator.slack_bus_indices = self.slack_bus_indices

            # predictions currently CPU tensors; evaluator will operate on same device (cpu)
            violations = evaluator.evaluate_all_constraints(
                predictions=predictions,
                batch_data=batch,
                normalize=normalize,
                return_individual=False,
            )
            summary = evaluator.get_violation_summary(violations)

            sample_weight = batch['bus'].batch.max().item() + 1 if hasattr(batch['bus'], 'batch') else 1
            for key, value in summary.items():
                v = float(value)
                accum_sum[key] = accum_sum.get(key, 0.0) + v * sample_weight
                accum_sq[key] = accum_sq.get(key, 0.0) + v * v * sample_weight
                accum_weight[key] = accum_weight.get(key, 0.0) + sample_weight

            batches_seen += 1
            progress_iter.set_postfix(batches=batches_seen, refresh=False)

        progress_iter.close()

        # compute mean/var
        stats = {}
        if batches_seen > 0:
            for key in sorted(accum_sum.keys()):
                weight = accum_weight.get(key, 0.0)
                if weight == 0:
                    continue
                mean = accum_sum[key] / weight
                mean_sq = accum_sq[key] / weight
                var = mean_sq - mean * mean
                stats[key] = {"mean": mean, "var": var, "weight": weight}
        return stats
