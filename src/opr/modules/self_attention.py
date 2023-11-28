"""Self-attention modules."""
from typing import Dict, Union

import torch
from torch import Tensor, nn


class SelfAttention(nn.Module):
    """Self-attention module."""

    def __init__(self, embed_size: int) -> None:
        """Self-attention module.

        Args:
            embed_size (int): Embedding size.
        """
        super().__init__()
        self.embed_size = embed_size
        self.to_v = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.to_k = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.to_q = nn.Linear(self.embed_size, self.embed_size, bias=False)

    def forward(self, x: Union[Tensor, Dict[str, Tensor]]) -> Union[Tensor, Dict[str, Tensor]]:  # noqa: D102
        if isinstance(x, Tensor):
            return self._apply_self_attention(x)
        elif isinstance(x, Dict):
            data = {key: value for key, value in x.items() if value is not None}
            values = torch.stack(list(data.values()), dim=1)
            out = self._apply_self_attention(values)
            out_dict = {key: value.squeeze(1) for key, value in zip(data.keys(), torch.split(out, 1, dim=1))}
            return out_dict

    def _apply_self_attention(self, x: Tensor) -> Tensor:
        values = self.to_v(x)
        keys = self.to_k(x)
        queries = self.to_q(x)
        energy = torch.bmm(queries, keys.transpose(1, 2)) / (self.embed_size**0.5)
        attention = torch.softmax(energy, dim=2)
        out = torch.bmm(attention, values)
        return out
