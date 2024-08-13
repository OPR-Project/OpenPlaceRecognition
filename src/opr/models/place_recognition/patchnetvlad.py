"""Implementation of PatchNetVLAD model."""
from typing import Literal

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch import Tensor, nn
from typing import Tuple, Dict

from opr.modules.feature_extractors import (
    ResNet18FPNFeatureExtractor,
    ResNet50FPNFeatureExtractor,
    VGG16FeatureExtractor,
)

from .base import ImageModel


def get_integral_feature(feat_in: Tensor) -> Tensor:
    """
    Input/Output as [N,D,H,W] where N is batch size and D is descriptor dimensions
    For VLAD, D = K x d where K is the number of clusters and d is the original descriptor dimensions
    """
    feat_out = torch.cumsum(feat_in, dim=-1)
    feat_out = torch.cumsum(feat_out, dim=-2)
    feat_out = F.pad(feat_out, (1, 0, 1, 0), "constant", 0)
    return feat_out


def get_square_regions_from_integral(feat_integral: Tensor, patch_size: int, patch_stride: int) -> Tensor:
    """
    Input as [N,D,H+1,W+1] where additional 1s for last two axes are zero paddings
    regSize and regStride are single values as only square regions are implemented currently
    """
    N, D, H, W = feat_integral.shape

    conv_weight = torch.ones(D, 1, 2, 2, device=feat_integral.device.type)
    conv_weight[:, :, 0, -1] = -1
    conv_weight[:, :, -1, 0] = -1
    feat_regions = F.conv2d(feat_integral, conv_weight, stride=patch_stride, groups=D, dilation=patch_size)
    return feat_regions / (patch_size**2)


class PatchNetVLAD(ImageModel):
    """Patch-NetVLAD: Multi-Scale Fusion of Locally-Global Descriptors for Place Recognition.

    Paper: https://arxiv.org/abs/2103.01486
    Code is adopted from original repository: https://github.com/QVPR/Patch-NetVLAD
    """

    def __init__(
        self,
        backbone: Literal["resnet18", "resnet50", "vgg16"] = "vgg16",
        num_clusters: int = 64,
        normalize_input: bool = True,
        vladv2: bool = False,
        use_faiss: bool = True,
        patch_sizes: Tuple[int] = (4,),
        strides: Tuple[int] = (1,),
    ) -> None:
        """Initialize PatchNetVLAD model.

        Args:
            backbone (str): Backbone architecture. Defaults to "vgg16".
            num_clusters (int): Number of VLAD clusters. Defaults to 64.
            normalize_input (bool): Whether to normalize input data or not. Defaults to True.
            vladv2 (bool): Use vladv2 init params method. Defaults to False.
            use_faiss (bool): Use Faiss for faster nearest neighbor search. Defaults to True.
            patch_sizes (tuple): Patch sizes for Patch-NetVLAD. Defaults to (4,).
            strides (tuple): Strides for Patch-NetVLAD. Defaults to (1,).

        Raises:
            NotImplementedError: If given backbone is unknown.
        """
        nn.Module.__init__(self)
        if backbone == "resnet18":
            self.backbone = ResNet18FPNFeatureExtractor()
            dim = 256
        elif backbone == "resnet50":
            self.backbone = ResNet50FPNFeatureExtractor()
            dim = 256
        elif backbone == "vgg16":
            self.backbone = VGG16FeatureExtractor()
            dim = 512
        else:
            raise NotImplementedError(f"Backbone {backbone} is not supported.")

        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 0
        self.vladv2 = vladv2
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=vladv2)
        # noinspection PyArgumentList
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self.use_faiss = use_faiss
        self.padding_size = 0
        self.patch_sizes = []
        self.strides = []
        for patch_size, stride in zip(patch_sizes, strides):
            self.patch_sizes.append(int(patch_size))
            self.strides.append(int(stride))

    def init_params(self, clsts: np.ndarray, traindescs: np.ndarray) -> None:
        """Initialize NetVLAD layer parameters."""
        if not self.vladv2:
            clsts_assign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
            dots = np.dot(clsts_assign, traindescs.T)
            dots.sort(0)
            dots = dots[::-1, :]  # sort, descending

            self.alpha = (-np.log(0.01) / np.mean(dots[0, :] - dots[1, :])).item()
            # noinspection PyArgumentList
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            # noinspection PyArgumentList
            self.conv.weight = nn.Parameter(
                torch.from_numpy(self.alpha * clsts_assign).unsqueeze(2).unsqueeze(3)
            )
            self.conv.bias = None
        else:
            if not self.use_faiss:
                knn = NearestNeighbors(n_jobs=-1)
                knn.fit(traindescs)
                del traindescs
                ds_sq = np.square(knn.kneighbors(clsts, 2)[1])
                del knn
            else:
                index = faiss.IndexFlatL2(traindescs.shape[1])
                # noinspection PyArgumentList
                index.add(traindescs)
                del traindescs
                # noinspection PyArgumentList
                ds_sq = index.search(clsts, 2)[1]
                del index

            self.alpha = (-np.log(0.01) / np.mean(ds_sq[:, 1] - ds_sq[:, 0])).item()
            # noinspection PyArgumentList
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            del clsts, ds_sq

            # noinspection PyArgumentList
            self.conv.weight = nn.Parameter((2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1))
            # noinspection PyArgumentList
            self.conv.bias = nn.Parameter(-self.alpha * self.centroids.norm(dim=1))

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:  # noqa: D102
        img_descriptors = {}
        for key, value in batch.items():
            if key.startswith("images_"):
                features = self.backbone(value)

                N, C, H, W = features.shape

                if self.normalize_input:
                    features = F.normalize(features, p=2, dim=1)  # across descriptor dim

                # soft-assignment
                soft_assign = self.conv(features).view(N, self.num_clusters, H, W)
                soft_assign = F.softmax(soft_assign, dim=1)

                # calculate residuals to each cluster
                store_residual = torch.zeros(
                    [N, self.num_clusters, C, H, W],
                    dtype=features.dtype,
                    layout=features.layout,
                    device=features.device,
                )
                for j in range(self.num_clusters):  # slower than non-looped, but lower memory usage
                    residual = features.unsqueeze(0).permute(1, 0, 2, 3, 4) - self.centroids[
                        j : j + 1, :
                    ].expand(features.size(2), features.size(3), -1, -1).permute(2, 3, 0, 1).unsqueeze(0)

                    residual *= soft_assign[:, j : j + 1, :].unsqueeze(
                        2
                    )  # residual should be size [N K C H W]
                    store_residual[:, j : j + 1, :, :, :] = residual

                vlad_global = store_residual.view(N, self.num_clusters, C, -1)
                vlad_global = vlad_global.sum(dim=-1)
                store_residual = store_residual.view(N, -1, H, W)

                ivlad = get_integral_feature(store_residual)
                vladflattened = []
                for patch_size, stride in zip(self.patch_sizes, self.strides):
                    vladflattened.append(
                        get_square_regions_from_integral(ivlad, int(patch_size), int(stride))
                    )

                vlad_local = []
                for (
                    thisvlad
                ) in vladflattened:  # looped to avoid GPU memory issues with certain config combinations
                    thisvlad = thisvlad.view(N, self.num_clusters, C, -1)
                    thisvlad = F.normalize(thisvlad, p=2, dim=2)
                    thisvlad = thisvlad.view(features.size(0), -1, thisvlad.size(3))
                    thisvlad = F.normalize(thisvlad, p=2, dim=1)
                    vlad_local.append(thisvlad)

                vlad_global = F.normalize(vlad_global, p=2, dim=2)
                vlad_global = vlad_global.view(features.size(0), -1)
                vlad_global = F.normalize(vlad_global, p=2, dim=1)

                img_descriptors[f"{key}_vlad_local"] = vlad_local
                img_descriptors[f"{key}_vlad_global"] = vlad_global

        return img_descriptors
