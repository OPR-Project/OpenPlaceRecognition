"""Normalization modules."""
import torch
from torch import Tensor, nn


class L2Norm(nn.Module):
    """L2 Normalization layer for input embeddings.

    This module normalizes the input embeddings to have a unit L2 norm.

    Attributes:
        eps (float): A small value to avoid division by zero.
    """

    def __init__(self, eps: float = 1e-10) -> None:
        """Initializes the L2Norm module.

        Args:
            eps (float): A small value to avoid division by zero. Default is 1e-10.
        """
        super().__init__()
        self.eps: float = eps

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for the L2Norm module.

        Args:
            x (Tensor): Input tensor of shape (N, D) where N is the batch size and D is the embedding dimension.

        Returns:
            Tensor: L2 normalized tensor of the same shape as input.
        """
        norm = torch.norm(x, p=2, dim=1, keepdim=True) + self.eps
        return x / norm
