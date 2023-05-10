import torch
from torch import nn
import MinkowskiEngine as ME


class ASVT(nn.Module):
    """ASVT - self-attention module for sparse tensors."""

    def __init__(self, in_dim: int, reduction: int = 8) -> None:
        super().__init__()
        self.q_conv = ME.MinkowskiConvolution(in_dim, in_dim // reduction, 1, dimension=3, bias=False)
        self.k_conv = ME.MinkowskiConvolution(in_dim, in_dim // reduction, 1, dimension=3, bias=False)

        self.v_conv = ME.MinkowskiConvolution(in_dim, in_dim, 1, dimension=3, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        self.trans_conv = ME.MinkowskiConvolution(in_dim, in_dim, 1, dimension=3, bias=False)
        self.after_norm = ME.MinkowskiBatchNorm(in_dim)
        self.act = ME.MinkowskiReLU(inplace=True)

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
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
