import argparse
from pathlib import Path

import yaml

from lumina.model.base.utils import describe_model
from lumina.model.opf.hetero_model import HEAT, HGT, RGAT, OPFHeteroGNN
from lumina.model.opf.homo_model import get_gnnNets
from lumina.trainer.opf.utils import HETERO_MODEL_TYPES

DEFAULT_NODE_TYPES = ("bus", "generator", "load", "shunt")
DEFAULT_EDGE_TYPES = (
    ("bus", "ac_line", "bus"),
    ("bus", "transformer", "bus"),
    ("generator", "generator_link", "bus"),
    ("bus", "generator_link", "generator"),
    ("load", "load_link", "bus"),
    ("bus", "load_link", "load"),
    ("shunt", "shunt_link", "bus"),
    ("bus", "shunt_link", "shunt"),
)
EDGE_TYPES_WITH_ATTRS = {
    ("bus", "ac_line", "bus"),
    ("bus", "transformer", "bus"),
}


def resolve_config_path(model_type, hetero_config, homo_config):
    repo_root = Path(__file__).resolve().parent.parent
    if model_type in HETERO_MODEL_TYPES:
        return Path(hetero_config) if hetero_config else repo_root / "configs" / "model" / "heterognn.yaml"
    return Path(homo_config) if homo_config else repo_root / "configs" / "model" / "homognn.yaml"


def load_model_config(config_path, model_type):
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found: {config_path}")
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f) or {}
    models = config_data.get("models", {})
    if model_type not in models:
        available = ", ".join(sorted(models.keys()))
        raise KeyError(f"Model type '{model_type}' not found in {config_path}. Available: {available}")
    return dict(models[model_type])


def build_dummy_metadata(input_dim, edge_dim):
    node_dims = {node_type: input_dim for node_type in DEFAULT_NODE_TYPES}
    edge_dims = {
        edge_type: (edge_dim if edge_type in EDGE_TYPES_WITH_ATTRS else 0)
        for edge_type in DEFAULT_EDGE_TYPES
    }
    return {"nodes": node_dims, "edges": edge_dims}


def build_hetero_model(model_type, model_config, input_dim, output_dim, edge_dim):
    metadata = build_dummy_metadata(input_dim, edge_dim)
    input_channels = dict(metadata["nodes"])
    kwargs = {
        "metadata": metadata,
        "input_channels": input_channels,
        "hidden_channels": model_config["hidden_channels"],
        "out_channels": output_dim,
        "num_layers": model_config["num_layers"],
        "backend": model_config.get("backend", "sage"),
    }
    if model_type in {"RGAT", "HGT"}:
        kwargs["num_heads"] = model_config.get("num_heads", 1)
    if model_type == "HEAT":
        kwargs["attention_heads"] = model_config.get("attention_heads", 1)

    model_class = {
        "HeteroGNN": OPFHeteroGNN,
        "RGAT": RGAT,
        "HGT": HGT,
        "HEAT": HEAT,
    }[model_type]
    return model_class(**kwargs)


def build_homo_model(model_type, model_config, input_dim, output_dim, edge_dim):
    model_params = dict(model_config)
    model_params["model_name"] = model_type
    if "edge_dim" not in model_params or model_params["edge_dim"] is None:
        model_params["edge_dim"] = edge_dim
    return get_gnnNets(
        input_dim=input_dim,
        output_dim=output_dim,
        model_params=model_params,
    )


def main():
    parser = argparse.ArgumentParser(description="Show model summary from config.")
    parser.add_argument("--model_type", type=str, required=True, help="Model type to describe.")
    parser.add_argument(
        "--hetero_model_config",
        type=str,
        default=None,
        help="Path to hetero model config YAML (heterognn.yaml).",
    )
    parser.add_argument(
        "--homo_model_config",
        type=str,
        default=None,
        help="Path to homo model config YAML (homognn.yaml).",
    )
    parser.add_argument("--input_dim", type=int, default=64, help="Input feature dimension.")
    parser.add_argument("--output_dim", type=int, default=2, help="Output feature dimension.")
    parser.add_argument("--edge_dim", type=int, default=32, help="Edge feature dimension.")
    args = parser.parse_args()

    model_type = args.model_type
    config_path = resolve_config_path(model_type, args.hetero_model_config, args.homo_model_config)
    model_config = load_model_config(config_path, model_type)

    if model_type in HETERO_MODEL_TYPES:
        model = build_hetero_model(
            model_type,
            model_config,
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            edge_dim=args.edge_dim,
        )
    else:
        model = build_homo_model(
            model_type,
            model_config,
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            edge_dim=args.edge_dim,
        )

    print(f"Loaded model config: {config_path}")
    describe_model(model, model_type=model_type, model_config=model_config)


if __name__ == "__main__":
    main()
