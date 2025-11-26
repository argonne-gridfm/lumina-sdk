import torch


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
