from typing import Dict

import torch
from torch import nn, Tensor


# TODO refactor ASAP
class CrossAttention(nn.Module):
    """Cross-Attention layer for 2 image descriptors.

    Works in a strange way - for 2 embeddings only. Computes cross-attention scores between those embeddings
    and return 'cross-attentioned' values.

    TODO: should be re-thinkend.
    """

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
            data : dict with 2 image descriptors that needs to be cross-attentioned
        returns :
            out : cross attention value + input feature
        """
        assert len(data.values()) == 2
        im1 = list(data.values())[0]
        im2 = list(data.values())[1]

        im1_query = self.to_q(im1).unsqueeze(1)  # (B x 1 x C_qk)
        im1_key = self.to_k(im1).unsqueeze(1)  # (B x 1 x C_qk)
        im1_value = self.to_v(im1).unsqueeze(1)  # (B x 1 x C)

        im2_query = self.to_q(im2).unsqueeze(1)
        im2_key = self.to_k(im2).unsqueeze(1)
        im2_value = self.to_v(im2).unsqueeze(1)

        im1_attn = torch.bmm(im2_query, im1_key.permute(0, 2, 1))  # (B x 1 x 1)  attention score
        im1_attn = im1_attn.softmax(dim=-1)

        im2_attn = torch.bmm(im1_query, im2_key.permute(0, 2, 1))  # (B x 1 x 1)  attention score
        im2_attn = im2_attn.softmax(dim=-1)

        im1_out = torch.bmm(im1_attn, im1_value).squeeze()  # (B x C)
        im2_out = torch.bmm(im2_attn, im2_value).squeeze()
        im1_out = im1_out + im1
        im2_out = im2_out + im2

        out_dict = {"im1": im1_out, "im2": im2_out}

        return out_dict


# TODO refactor (tmp solution for experiments)
class CrossAttentionFusion(nn.Module):
    """Cross-Attention layer for images"""

    def __init__(self, in_dim: int, reduction: int = 8) -> None:
        super().__init__()
        self.chanel_in = in_dim
        self.to_q = nn.Linear(in_dim, in_dim // reduction, bias=False)
        self.to_k = nn.Linear(in_dim, in_dim // reduction, bias=False)
        self.to_v = nn.Linear(in_dim, in_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, data: Dict[str, Tensor]) -> Tensor:
        """
        inputs :
            im1 : input image1 descriptors ( B x C )
            im2 : input image2 descriptors ( B x C )
        returns :
            out : cross attention value + input feature
        """
        assert len(data.values()) == 2
        im1 = list(data.values())[0]
        im2 = list(data.values())[1]

        im1_key = self.to_k(im1).unsqueeze(1)  # (B x 1 x C_qk)
        im1_value = self.to_v(im1).unsqueeze(1)  # (B x 1 x C)

        im2_query = self.to_q(im2).unsqueeze(1)  # (B x 1 x C_qk)

        im1_attn = torch.bmm(im2_query, im1_key.permute(0, 2, 1))  # (B x 1 x 1)  attention score
        im1_attn = im1_attn.softmax(dim=-1)

        im1_out = torch.bmm(im1_attn, im1_value).squeeze()  # (B x C)
        im1_out = im1_out + im1

        return im1_out
