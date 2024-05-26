import torch
from torch import Tensor


def get_batch_mask_for_nodes(natm: Tensor) -> Tensor:
    r"""Get fragment index for each node
    Example: Tensor([2, 0, 3]) -> [0, 0, 2, 2, 2]

    Args:
        natm (Tensor): number of nodes per small fragment

    Returns:
        Tensor: [n_node], the natural index of fragment a node belongs to
    """
    return torch.repeat_interleave(
        torch.arange(natm.size(0), device=natm.device), natm
    ).to(natm.device)


def get_full_edges_index(
    nodes_mask: Tensor,
    remove_self_edge: bool = False,
) -> Tensor:
    adj = nodes_mask[:, None] == nodes_mask[None, :]
    if remove_self_edge:
        adj = adj.fill_diagonal_(False)
    edges = torch.stack(torch.where(adj), dim=0)
    return edges
