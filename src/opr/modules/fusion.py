"""Basic fusion modules implementation."""
from typing import Dict

import torch
from torch import Tensor, nn


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
        return fusion_global_descriptor
