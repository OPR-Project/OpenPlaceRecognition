"""Basic fusion layers implementation."""
from typing import Dict, Union

import MinkowskiEngine as ME  # noqa: N817
import torch
from torch import Tensor

from opr.models.base_models import FusionModule


class Concat(FusionModule):
    """Concatenation module for 'image' and 'cloud' modalities."""

    def __init__(self) -> None:
        """Concatenation module for 'image' and 'cloud' modalities."""
        super().__init__()

    def forward(self, data: Dict[str, Union[Tensor, ME.SparseTensor]]) -> Tensor:  # noqa: D102
        assert "image" in data
        assert "cloud" in data
        fusion_global_descriptor = torch.concat([data["image"], data["cloud"]], dim=1)
        return fusion_global_descriptor
