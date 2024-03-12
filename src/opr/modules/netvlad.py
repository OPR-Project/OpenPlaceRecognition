"""NetVLAD layer implementation."""
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch import Tensor, nn
from torch.nn import functional as F


# Source: https://github.com/Nanne/pytorch-NetVlad/blob/master/netvlad.py
# based on https://github.com/lyakaap/NetVLAD-pytorch/blob/master/netvlad.py
class NetVLAD(nn.Module):
    """NetVLAD layer implementation."""

    def __init__(
        self, num_clusters: int = 64, dim: int = 128, normalize_input: bool = True, vladv2: bool = False
    ) -> None:
        """Initialize NetVLAD layer.

        Args:
            num_clusters (int): Number of VLAD clusters. Defaults to 64.
            dim (int): Dimension of input descriptors. Defaults to 128.
            normalize_input (bool): Whether to normalize input data or not. Defaults to True.
            vladv2 (bool): Use vladv2 init params method. Defaults to False.
        """
        super().__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 0
        self.vladv2 = vladv2
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=vladv2)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self.out_projection = nn.Linear(num_clusters * dim, dim)

    def init_params(self, clsts: np.ndarray, traindescs: np.ndarray) -> None:
        """Initialize NetVLAD layer parameters."""
        clsts = torch.from_numpy(clsts)
        traindescs = torch.from_numpy(traindescs)

        if self.vladv2 is False:
            clstsAssign = clsts / torch.norm(clsts, dim=1, keepdim=True)
            dots = torch.mm(clstsAssign, traindescs.t())
            dots, _ = dots.sort(dim=0, descending=True)

            self.alpha = (-torch.log(torch.tensor(0.01)) / torch.mean(dots[0, :] - dots[1, :])).item()
            self.centroids = nn.Parameter(clsts)
            self.conv.weight = nn.Parameter((self.alpha * clstsAssign).unsqueeze(2).unsqueeze(3))
            self.conv.bias = None
        else:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(traindescs.cpu().numpy())
            dsSq = torch.square(torch.tensor(knn.kneighbors(clsts.cpu().numpy(), 2)[1]))
            del knn
            self.alpha = (-torch.log(torch.tensor(0.01)) / torch.mean(dsSq[:, 1] - dsSq[:, 0])).item()
            self.centroids = nn.Parameter(clsts)
            del clsts, dsSq

            self.conv.weight = nn.Parameter((2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1))
            self.conv.bias = nn.Parameter(-self.alpha * torch.norm(self.centroids, dim=1))

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)

        # calculate residuals to each clusters
        vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for C in range(self.num_clusters):  # slower than non-looped, but lower memory usage
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - self.centroids[C : C + 1, :].expand(
                x_flatten.size(-1), -1, -1
            ).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign[:, C : C + 1, :].unsqueeze(2)
            vlad[:, C : C + 1, :] = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        out = self.out_projection(vlad)  # fc layer

        return out
