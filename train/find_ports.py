import torch
from typing import Tuple


def unravel_index(
        indices: torch.LongTensor,
        shape: Tuple[int, ...],
) -> torch.LongTensor:
    shape = torch.tensor(shape, device=indices.device)
    indices = indices % shape.prod()  # prevent out-of-bounds indices

    coord = torch.zeros(indices.size() + shape.size(), dtype=torch.long, device=indices.device)

    for i, dim in enumerate(reversed(shape)):
        coord[..., i] = indices % dim
        indices = torch.div(indices, dim, rounding_mode='trunc')

    return coord.flip(-1)


def optimize_indices(erro: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    B, F, H, W = erro.shape
    # Flatten the last two dimensions and find the argmin across this flattened dimension
    flat_indices = torch.argmin(erro.view(B, F, -1), dim=-1)

    # Unravel the flat indices into the original shape of the last two dimensions
    coords = unravel_index(flat_indices.flatten(), (H, W))

    # Reshape to match the original batch and feature dimensions
    index_x = coords[:, 0].view(B, F)
    index_y = coords[:, 1].view(B, F)

    return index_x, index_y

