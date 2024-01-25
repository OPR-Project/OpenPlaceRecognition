"""Basic fusion modules implementation."""
from typing import Dict

import torch
from torch import Tensor, nn

from .gem import SeqGeM


class Concat(nn.Module):
    """Concatenation module."""

    def __init__(self) -> None:
        """Concatenation module."""
        super().__init__()

    def forward(self, data: Dict[str, Tensor]) -> Tensor:  # noqa: D102
        data = {key: value for key, value in data.items() if value is not None}
        fusion_global_descriptor = torch.concat(list(data.values()), dim=1)
        return fusion_global_descriptor


class Add(nn.Module):
    """Addition module."""

    def __init__(self) -> None:
        """Addition module."""
        super().__init__()

    def forward(self, data: Dict[str, Tensor]) -> Tensor:  # noqa: D102
        data = {key: value for key, value in data.items() if value is not None}
        fusion_global_descriptor = torch.stack(list(data.values()), dim=0).sum(dim=0)
        if len(fusion_global_descriptor.shape) == 1:
            fusion_global_descriptor = fusion_global_descriptor.unsqueeze(0)

        return fusion_global_descriptor


class GeMFusion(nn.Module):
    """GeM fusion module."""

    def __init__(self, p: int = 3, eps: float = 1e-6) -> None:
        """Generalized-Mean fusion module.

        Args:
            p (int): Initial value of learnable parameter 'p', see paper for more details. Defaults to 3.
            eps (float): Negative values will be clamped to `eps` (ReLU). Defaults to 1e-6.
        """
        super().__init__()
        self.gem = SeqGeM(p=p, eps=eps)

    def forward(self, data: Dict[str, Tensor]) -> Tensor:  # noqa: D102
        data = {key: value for key, value in data.items() if value is not None}
        descriptors = list(data.values())
        descriptors = torch.stack(descriptors, dim=len(descriptors[0].shape))
        print(descriptors.shape)
        return self.gem(descriptors)
