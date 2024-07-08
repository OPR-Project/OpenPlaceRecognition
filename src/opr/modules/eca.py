"""Implementation of Efficient Channel Attention ECA block.

Wang, Qilong, et al. "ECA-Net: Efficient channel attention for deep convolutional neural networks."
Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.

Paper: https://arxiv.org/abs/1910.03151
Code for Mink version adopted from the repository: https://github.com/jac99/MinkLoc3Dv2, MIT License
"""
from __future__ import annotations
from typing import Optional

import numpy as np
from loguru import logger
from torch import nn

try:
    import MinkowskiEngine as ME  # type: ignore
    from MinkowskiEngine.modules.resnet_block import BasicBlock

    minkowski_available = True
except ImportError:
    logger.warning("MinkowskiEngine is not installed. Some features may not be available.")
    BasicBlock = nn.Module
    minkowski_available = False


class MinkECALayer(nn.Module):
    """Efficient Channel Attention layer for sparse tensors with MinkowskiEngine."""

    def __init__(self, channels: int, gamma: int = 2, b: int = 1) -> None:
        """Efficient Channel Attention layer for sparse tensors with MinkowskiEngine.

        Original paper: https://arxiv.org/abs/1910.03151

        Args:
            channels (int): Number of channels in input.
            gamma (int): Gamma parameter, see paper for more details. Defaults to 2.
            b (int): b parameter, see paper for more details. Defaults to 1.

        Raises:
            RuntimeError: If MinkowskiEngine is not installed.
        """
        if not minkowski_available:
            raise RuntimeError("MinkowskiEngine is not installed. MinkECALayer requires MinkowskiEngine.")
        super().__init__()
        t = int(abs((np.log2(channels) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.avg_pool = ME.MinkowskiGlobalPooling()
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.broadcast_mul = ME.MinkowskiBroadcastMultiplication()

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:  # noqa: D102
        # feature descriptor on the global spatial information
        y_sparse = self.avg_pool(x)
        # Apply 1D convolution along the channel dimension
        y = self.conv(y_sparse.F.unsqueeze(1)).squeeze(1)
        # y is (batch_size, channels) tensor
        y = self.sigmoid(y)
        # y is (batch_size, channels) tensor
        y_sparse = ME.SparseTensor(
            y, coordinate_manager=y_sparse.coordinate_manager, coordinate_map_key=y_sparse.coordinate_map_key
        )
        # y must be features reduced to the origin
        return self.broadcast_mul(x, y_sparse)


class MinkECABasicBlock(BasicBlock):
    """Efficient Channel Attention BasicBlock for ResNet with MinkowskiEngine."""

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: Optional[nn.Module] = None,
        dimension: int = 3,
    ) -> None:
        """Efficient Channel Attention BasicBlock for ResNet with MinkowskiEngine.

        Args:
            inplanes (int): Number of input channels.
            planes (int): Number of output channels.
            stride (int): Convolution stride size. Defaults to 1.
            dilation (int): Convolution dilation size. Defaults to 1.
            downsample (nn.Module, optional): Downsample layer, if needed. Defaults to None.
            dimension (int): Number of dimensions. Defaults to 3.

        Raises:
            RuntimeError: If MinkowskiEngine is not installed.
        """
        if not minkowski_available:
            raise RuntimeError(
                "MinkowskiEngine is not installed. MinkECABasicBlock requires MinkowskiEngine."
            )
        super().__init__(
            inplanes, planes, stride=stride, dilation=dilation, downsample=downsample, dimension=dimension
        )
        self.eca = MinkECALayer(planes, gamma=2, b=1)

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:  # noqa: D102
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.eca(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
