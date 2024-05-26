import torch
from typing import List
from torch_scatter import scatter_mean
from torch import Tensor


def remove_mean_batch(x, indices):
    mean = scatter_mean(x, indices, dim=0)
    x = x - mean[indices]
    return x


def sample_center_gravity_zero_gaussian_batch(
    size: List[int],
    indices: Tensor
) -> Tensor:
    assert len(size) == 2
    x = torch.randn(size, device=indices.device)

    # This projection only works because Gaussian is rotation invariant
    # around zero and samples are independent!
    x_projected = remove_mean_batch(x, indices)
    return x_projected

