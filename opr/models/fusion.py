"""Basic fusion layers implementation."""
from typing import Dict

import torch
from torch import Tensor

from opr.models.base_models import FusionModule


class Concat(FusionModule):
    """Concatenation module for all modalities."""

    def __init__(self) -> None:
        """Concatenation module for all modalities."""
        super().__init__()

    def forward(self, data: Dict[str, Tensor]) -> Tensor:  # noqa: D102
        # assert "image" in data
        # assert "cloud" in data
        fusion_global_descriptor = torch.concat(list(data.values()), dim=1)
        return fusion_global_descriptor


class Add(FusionModule):
    """Addition module for all modalities."""

    def __init__(self) -> None:
        """Addition module for all modalities."""
        super().__init__()

    def forward(self, data: Dict[str, Tensor]) -> Tensor:  # noqa: D102
        # assert "image" in data
        # assert "cloud" in data
        fusion_global_descriptor = torch.stack(list(data.values()), dim=0).sum(dim=0)
        return fusion_global_descriptor
