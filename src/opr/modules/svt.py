"""Implementation of ASVT and CSVT modules.

Citation:
    Fan, Zhaoxin, et al. "Svt-net: Super light-weight sparse voxel transformer
    for large scale place recognition." Proceedings of the AAAI Conference on Artificial Intelligence.
    Vol. 36. No. 1. 2022.

Source: https://github.com/ZhenboSong/SVTNet
Paper: https://arxiv.org/abs/2105.00149
"""
from __future__ import annotations

import torch
from loguru import logger
from torch import nn

try:
    import MinkowskiEngine as ME  # type: ignore

    minkowski_available = True
except ImportError:
    logger.warning("MinkowskiEngine is not installed. Some features may not be available.")
    minkowski_available = False


class ASVT(nn.Module):
    """ASVT - Atom-Based Sparse Voxel Transformer."""

    def __init__(self, in_dim: int, reduction: int = 8) -> None:
        """ASVT - Atom-Based Sparse Voxel Transformer.

        Args:
            in_dim (int): Input dimension.
            reduction (int): Reduction ratio. Defaults to 8.

        Raises:
            RuntimeError: If MinkowskiEngine is not installed.
        """
        if not minkowski_available:
            raise RuntimeError("MinkowskiEngine is not installed. ASVT requires MinkowskiEngine.")
        super().__init__()
        self.q_conv = ME.MinkowskiConvolution(in_dim, in_dim // reduction, 1, dimension=3, bias=False)
        self.k_conv = ME.MinkowskiConvolution(in_dim, in_dim // reduction, 1, dimension=3, bias=False)

        self.v_conv = ME.MinkowskiConvolution(in_dim, in_dim, 1, dimension=3, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        self.trans_conv = ME.MinkowskiConvolution(in_dim, in_dim, 1, dimension=3, bias=False)
        self.after_norm = ME.MinkowskiBatchNorm(in_dim)
        self.act = ME.MinkowskiReLU(inplace=True)

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:  # noqa: D102
        x_q = self.q_conv(x)
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)

        bath_size = torch.max(x.C[:, 0], 0)[0] + 1
        start_id = 0
        x_feat = list()
        for i in range(bath_size):
            end_id = start_id + torch.sum(x.C[:, 0] == i)
            dq = x_q.F[start_id:end_id, :]  # N*C
            dk = x_k.F[start_id:end_id, :].T  # C*N
            dv = x_v.F[start_id:end_id, :]  # N*C
            de = torch.matmul(dq, dk)  # N*N
            da = self.softmax(de)  # N*N
            # da = da / (1e-9 + da.sum(dim=1, keepdim=True))
            dr = torch.matmul(da, dv)  # N*C
            x_feat.append(dr)
            start_id = end_id
        x_r = torch.cat(x_feat, dim=0)
        x_r = ME.SparseTensor(
            # coordinates=x.coordinates,
            features=x_r,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )

        x_r = x + self.act(self.after_norm(self.trans_conv(x_r)))
        return x_r


class CSVT(nn.Module):
    """CSVT - Cluster-Based Sparse Voxel Transformer."""

    def __init__(self, channels: int, num_tokens: int = 16) -> None:
        """CSVT - Cluster-Based Sparse Voxel Transformer.

        Args:
            channels (int): Number of input channels.
            num_tokens (int): Number of tokens. Defaults to 16.

        Raises:
            RuntimeError: If MinkowskiEngine is not installed.
        """
        if not minkowski_available:
            raise RuntimeError("MinkowskiEngine is not installed. CSVT requires MinkowskiEngine.")
        super().__init__()

        # layers for generate tokens
        self.q_conv = ME.MinkowskiConvolution(channels, channels, 1, dimension=3, bias=False)
        self.k_conv = ME.MinkowskiConvolution(channels, num_tokens, 1, dimension=3, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        # layers for tranformer
        self.convvalues = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.convkeys = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.convquries = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.embedding1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        # layers for projector
        self.p_conv = ME.MinkowskiConvolution(channels, channels, 1, dimension=3, bias=False)
        self.T_conv = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        # hidden state
        self.trans_conv = ME.MinkowskiConvolution(channels, channels, 1, dimension=3, bias=False)
        self.after_norm = ME.MinkowskiBatchNorm(channels)
        self.act = ME.MinkowskiReLU(inplace=True)

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:  # noqa: D102
        # generate tokens
        x_q = self.q_conv(x)
        x_k = self.k_conv(x)

        bath_size = torch.max(x.C[:, 0], 0)[0] + 1
        start_id = 0
        x_feat = list()
        for i in range(bath_size):
            end_id = start_id + torch.sum(x.C[:, 0] == i)
            dq = x_q.F[start_id:end_id, :]  # N*C
            dk = x_k.F[start_id:end_id, :].T  # num_tokens*N
            dk = self.softmax(dk)  # N*num_tokens

            de = torch.matmul(dk, dq).T  # C*num_tokens
            # da = da / (1e-9 + da.sum(dim=1, keepdim=True))
            de = torch.unsqueeze(de, dim=0)
            x_feat.append(de)
            start_id = end_id
        tokens = torch.cat(x_feat, dim=0)  # B*C*num_tokens

        # visul transormers on multi tokens
        vt_values = self.convvalues(tokens)
        vt_keys = self.convkeys(tokens)  # B*C*num_tokens
        vt_quires = self.convquries(tokens)  # B*C*num_tokens
        vt_map = torch.matmul(vt_keys.transpose(1, 2), vt_quires)  # B*num_tokens*num_tokens
        vt_map = self.softmax(vt_map)  # B*num_tokens*num_tokens
        T_middle = torch.matmul(vt_map, vt_values.transpose(1, 2)).transpose(1, 2)  # B*C*num_tokens
        # T_out = tokens + self.actembedding1(self.bnembedding1(self.embedding1(T_middle)))  # B*C*num_tokens
        T_out = tokens + self.embedding1(T_middle)

        # projector
        x_p = self.p_conv(x)
        T_P = self.T_conv(T_out)

        start_id = 0
        x_feat2 = list()
        for i in range(bath_size):
            end_id = start_id + torch.sum(x.C[:, 0] == i)
            dp = x_p.F[start_id:end_id, :]  # N*C
            dt = T_P[i]  # C*num_tokens

            dm = torch.matmul(dp, dt)  # N*num_tokens
            dm = self.softmax(dm)  # N*num_tokens

            df = torch.matmul(dm, dt.T)  # N*C
            x_feat2.append(df)
            start_id = end_id
        x_r = torch.cat(x_feat2, dim=0)

        x_r = ME.SparseTensor(
            # coordinates=x.coordinates,
            features=x_r,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )
        x_r = x + self.act(self.after_norm(self.trans_conv(x_r)))
        return x_r
