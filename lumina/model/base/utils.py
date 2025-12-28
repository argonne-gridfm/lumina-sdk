import torch
try:
    from torch.nn.parameter import UninitializedParameter
except Exception:
    UninitializedParameter = None


def from_adj_to_edge_index_torch(adj):
    """ Convert a dense adjacency matrix to a sparse edge index and edge attribute tensor.
    The edge attribute tensor is the non-zero values (weights) of the adjacency matrix.

    Args:
        adj (torch.Tensor): Dense adjacency matrix.

    Returns:
        edge_index (torch.Tensor): Sparse edge index tensor.
        edge_attr (torch.Tensor): Edge attribute tensor
    """
    adj_sparse = adj.to_sparse()
    edge_index = adj_sparse.indices().to(dtype=torch.long)
    edge_attr = adj_sparse.values()
    return edge_index, edge_attr


def describe_model(model, model_type=None, model_config=None, print_fn=print):
    """Format and optionally print a short model summary."""
    base_model = model.module if hasattr(model, "module") else model
    model_name = model_type or base_model.__class__.__name__
    total_params = 0
    trainable_params = 0
    uninitialized_params = 0
    total_param_bytes = 0
    for param in base_model.parameters():
        if UninitializedParameter is not None and isinstance(param, UninitializedParameter):
            uninitialized_params += 1
            continue
        try:
            param_count = param.numel()
            param_bytes = param_count * param.element_size()
        except (ValueError, RuntimeError):
            uninitialized_params += 1
            continue
        total_params += param_count
        total_param_bytes += param_bytes
        if param.requires_grad:
            trainable_params += param_count

    detail_parts = []
    if isinstance(model_config, dict):
        for key in (
            "model_name",
            "hidden_channels",
            "hidden_dim",
            "num_layers",
            "num_heads",
            "attention_heads",
            "backend",
            "dropout",
            "readout",
            "edge_dim",
        ):
            if key in model_config and model_config[key] is not None:
                detail_parts.append(f"{key}={model_config[key]}")

    if total_param_bytes >= 1024**3:
        memory_label = f"{total_param_bytes / 1024**3:.2f} GB"
    else:
        memory_label = f"{total_param_bytes / 1024**2:.2f} MB"

    lines = [
        f"Model: {model_name}",
        f"Parameters: {total_params:,} (trainable: {trainable_params:,})",
        f"Parameter memory: {memory_label}",
    ]
    if detail_parts:
        lines.append("Config: " + ", ".join(detail_parts))
    if uninitialized_params:
        lines.append(f"Uninitialized params: {uninitialized_params} (excluded from counts)")

    summary = "\n".join(lines)
    if print_fn is not None:
        print_fn(summary)
    return summary
