"""Generalized-Mean pooling layer implementation.

Radenović, Filip, Giorgos Tolias, and Ondřej Chum. "Fine-tuning CNN image retrieval
with no human annotation." IEEE transactions on pattern analysis and
machine intelligence 41.7 (2018): 1655-1668.

Paper: https://arxiv.org/abs/1711.02512
Code adopted from the repository: https://github.com/jac99/MinkLocMultimodal, MIT License
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from opr.optional_deps import lazy

# Lazy-load MinkowskiEngine - will return real module or helpful stub
ME = lazy("MinkowskiEngine", feature="sparse convolutions")


class MinkGeM(nn.Module):
    """GeM pooling layer for sparse tensors with MinkowskiEngine."""

    sparse: bool = True

    def __init__(self, p: int = 3, eps: float = 1e-6) -> None:
        """Generalized-Mean pooling layer for sparse tensors with MinkowskiEngine.

        Original paper: https://arxiv.org/abs/1711.02512

        Args:
            p (int): Initial value of learnable parameter 'p', see paper for more details. Defaults to 3.
            eps (float): Negative values will be clamped to `eps` (ReLU). Defaults to 1e-6.
        """
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.f = ME.MinkowskiGlobalAvgPooling()

    def forward(self, x: object) -> Tensor:  # noqa: D102
        # This implicitly applies ReLU on x (clamps negative values)
        temp = ME.SparseTensor(x.F.clamp(min=self.eps).pow(self.p), coordinates=x.C)
        temp = self.f(temp)  # Apply ME.MinkowskiGlobalAvgPooling
        return temp.F.pow(1.0 / self.p)  # Return (batch_size, n_features) tensor


class GeM(nn.Module):
    """GeM pooling layer."""

    def __init__(self, p: int = 3, eps: float = 1e-6) -> None:
        """Generalized-Mean pooling layer.

        Original paper: https://arxiv.org/abs/1711.02512

        Args:
            p (int): Initial value of learnable parameter 'p', see paper for more details. Defaults to 3.
            eps (float): Negative values will be clamped to `eps` (ReLU). Defaults to 1e-6.
        """
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        return self._gem(x, p=self.p, eps=self.eps)

    def _gem(self, x: Tensor, p: nn.Parameter, eps: float) -> Tensor:
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p).squeeze()


class SeqGeM(GeM):
    """1D GeM pooling layer."""

    def _gem(self, x: Tensor, p: nn.Parameter, eps: float) -> Tensor:
        return F.avg_pool1d(x.clamp(min=eps).pow(p), x.size(-1)).pow(1.0 / p).squeeze()


class GlobalAvgPooling(nn.Module):
    """Global average pooling fusion module."""

    def __init__(self) -> None:
        """Global average pooling fusion module."""
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        # Handle both [N,L,C] and [N,H,W,C] cases
        if len(x.shape) == 3:  # [N,L,C] case after stacking
            out = x.mean(dim=1)  # Pool over L dimension
        elif len(x.shape) == 4:  # [N,H,W,C] case after stacking
            out = x.mean(dim=(1,2))  # Pool over H,W dimensions
        return out


class GlobalMaxPooling(nn.Module):
    """Global max pooling fusion module."""

    def __init__(self) -> None:
        """Global max pooling fusion module."""
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
       # Handle both [N,L,C] and [N,H,W,C] cases
        if len(x.shape) == 3:  # [N,L,C] case after stacking
            out = x.max(dim=0)[0]  # Pool over L dimension
        elif len(x.shape) == 4:  # [N,H,W,C] case after stacking
            out = x.max(dim=1)[0]  # Pool over H dimension
            out = out.max(dim=2)[0]  # Pool over W dimension
        return out
