import json

import torch
import torch.distributed as dist

from lumina.model.opf.hetero_model import HEAT, HGT, RGAT, OPFHeteroGNN

HETERO_MODEL_TYPES = {"HeteroGNN", "RGAT", "HEAT", "HGT"}
HETERO_MODEL_CLASSES = {
    "HeteroGNN": OPFHeteroGNN,
    "RGAT": RGAT,
    "HEAT": HEAT,
    "HGT": HGT,
}
_HETERO_MODEL_TYPE_LOOKUP = {
    "heterognn": "HeteroGNN",
    "opfheterognn": "HeteroGNN",
    "rgat": "RGAT",
    "heat": "HEAT",
    "hgt": "HGT",
}

_CASE_NAME_MAPPING = {
    "case14": "pglib_opf_case14_ieee",
    "case30": "pglib_opf_case30_ieee",
    "case57": "pglib_opf_case57_ieee",
    "case118": "pglib_opf_case118_ieee",
    "case500": "pglib_opf_case500_goc",
    "case2000": "pglib_opf_case2000_goc",
    "case4661": "pglib_opf_case4661_sdet",
    "case6470": "pglib_opf_case6470_rte",
    "case10000": "pglib_opf_case10000_goc",
    "case13659": "pglib_opf_case13659_pegase",
}


def get_case_name_mapping():
    return dict(_CASE_NAME_MAPPING)


def parse_case_name(case_input: str) -> str:
    case_mapping = get_case_name_mapping()

    if case_input.startswith("pglib_opf_"):
        return case_input

    if case_input in case_mapping:
        return case_mapping[case_input]

    if not case_input.startswith("case"):
        case_input = "case" + case_input
        if case_input in case_mapping:
            return case_mapping[case_input]

    available_short = list(case_mapping.keys())
    available_full = list(case_mapping.values())
    raise ValueError(
        f"Invalid case name '{case_input}'. Available short names: {available_short}, "
        f"or use full names: {available_full}"
    )


def parse_cases_arg(cases_arg):
    expanded = []
    for entry in cases_arg:
        entry = entry.strip()
        if not entry:
            continue
        if entry.startswith("["):
            expanded.extend(json.loads(entry))
        elif "," in entry:
            expanded.extend(x.strip() for x in entry.split(",") if x.strip())
        else:
            expanded.append(entry)
    return expanded


def _canonical_hetero_model_type(model_type):
    if not isinstance(model_type, str):
        return None
    return _HETERO_MODEL_TYPE_LOOKUP.get(model_type.strip().lower())


def resolve_hetero_model_type(model_type=None, model_class_path=None, default="HeteroGNN"):
    if isinstance(model_class_path, str) and model_class_path.strip():
        class_name = model_class_path.rsplit(".", 1)[-1]
        normalized = _canonical_hetero_model_type(class_name)
        if normalized is not None:
            return normalized
        raise ValueError(
            f"Unsupported hetero model class path '{model_class_path}'. "
            f"Supported classes: {sorted(HETERO_MODEL_CLASSES.keys()) + ['OPFHeteroGNN']}"
        )

    if isinstance(model_type, str) and model_type.strip():
        normalized = _canonical_hetero_model_type(model_type)
        if normalized is not None:
            return normalized
        raise ValueError(
            f"Unsupported hetero model type '{model_type}'. Supported types: {sorted(HETERO_MODEL_TYPES)}"
        )

    normalized_default = _canonical_hetero_model_type(default)
    if normalized_default is not None:
        return normalized_default

    supported = ", ".join(sorted(HETERO_MODEL_TYPES))
    raise ValueError(
        f"Unable to resolve hetero model type from model_type='{model_type}' "
        f"and model_class_path='{model_class_path}'. Supported types: {supported}"
    )


def build_hetero_model_spec(
    model_type,
    metadata,
    input_channels,
    models_config,
    out_channels=2,
):
    normalized_type = resolve_hetero_model_type(model_type=model_type, default=None)
    if normalized_type not in HETERO_MODEL_CLASSES:
        supported = ", ".join(sorted(HETERO_MODEL_CLASSES.keys()))
        raise ValueError(f"Unsupported hetero model type '{normalized_type}'. Supported types: {supported}")

    if not isinstance(models_config, dict):
        models_config = {}

    model_config = models_config.get(normalized_type)
    used_fallback = False
    if not isinstance(model_config, dict):
        fallback = models_config.get("HeteroGNN")
        if isinstance(fallback, dict):
            model_config = fallback
            used_fallback = normalized_type != "HeteroGNN"
        else:
            model_config = {}

    model_kwargs = {
        "metadata": metadata,
        "input_channels": input_channels,
    }
    model_kwargs.update(model_config)
    model_kwargs["out_channels"] = int(out_channels)

    model_kwargs.setdefault("hidden_channels", 64)
    model_kwargs.setdefault("num_layers", 3)
    model_kwargs.setdefault("backend", "sage")

    if normalized_type in {"RGAT", "HGT"}:
        model_kwargs.setdefault("num_heads", 1)
    if normalized_type == "HEAT":
        model_kwargs.setdefault("attention_heads", 1)

    return HETERO_MODEL_CLASSES[normalized_type], model_kwargs, model_config, used_fallback


def apply_nested(target_dict, dotted_key, value):
    if not isinstance(target_dict, dict):
        return
    if not isinstance(dotted_key, str):
        return

    keys = dotted_key.split(".")
    current = target_dict
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def initialize_model(model, sample_data, device):
    if dist.get_rank() == 0:
        print("Initializing model parameters...")

    model = model.to(device)
    sample_data = sample_data.to(device)

    model.eval()
    with torch.no_grad():
        try:
            if isinstance(sample_data, (dict, torch.nn.ParameterDict)) or hasattr(sample_data, "x_dict"):
                x_dict = {k: v.float() for k, v in sample_data.x_dict.items()}
                _ = model(x_dict, sample_data.edge_index_dict)
            else:
                if hasattr(sample_data, "x"):
                    sample_data.x = sample_data.x.float()
                _ = model(sample_data)
            if dist.get_rank() == 0:
                print("Model parameters initialized successfully!")
        except Exception as exc:
            if dist.get_rank() == 0:
                print(f"Warning: Model initialization failed: {exc}")
                print("Model may still work during training...")

    return model
