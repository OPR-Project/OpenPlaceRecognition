from typing import Dict

import torch
from torch import nn, Tensor


# TODO refactor ASAP
class CrossAttention(nn.Module):
    """Cross-Attention layer for images"""

    def __init__(self, in_dim: int, reduction: int = 8) -> None:
        super().__init__()
        self.chanel_in = in_dim
        self.to_q = nn.Linear(in_dim, in_dim // reduction, bias=False)
        self.to_k = nn.Linear(in_dim, in_dim // reduction, bias=False)
        self.to_v = nn.Linear(in_dim, in_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        inputs :
            im1 : input image1 feature maps ( B x C x W x H)
            im2 : input image2 feature maps ( B x C x W x H)
        returns :
            out : self attention value + input feature
            attention: B x N x N (N is Width*Height)
        """
        assert len(data.values()) == 2
        im1 = list(data.values())[0]
        im2 = list(data.values())[1]

        im1_query = self.to_q(im1)
        im1_key = self.to_k(im1)
        im1_value = self.to_v(im1)

        im2_query = self.to_q(im2)
        im2_key = self.to_k(im2)
        im2_value = self.to_v(im2)

        im1_attn = im2_query @ im1_key.T
        im1_attn = im1_attn.softmax(dim=-1)

        im2_attn = im1_query @ im2_key.T
        im2_attn = im2_attn.softmax(dim=-1)

        im1_out = im1_attn @ im1_value
        im2_out = im2_attn @ im2_value
        im1_out = im1_out + im1
        im2_out = im2_out + im2

        out_dict = {"im1": im1_out, "im2": im2_out}

        return out_dict
