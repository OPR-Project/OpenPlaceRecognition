"""Implementations of MinkLoc models."""
from typing import Tuple

from opr.modules import MinkGeM
from opr.modules.feature_extractors import MinkResNetFPNFeatureExtractor

from .base import CloudModel


class MinkLoc3D(CloudModel):
    """MinkLoc3D: Point Cloud Based Large-Scale Place Recognition.

    Paper: https://arxiv.org/abs/2011.04530
    Code is adopted from the original repository: https://github.com/jac99/MinkLoc3Dv2, MIT License
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 256,
        num_top_down: int = 1,
        conv0_kernel_size: int = 5,
        block: str = "BasicBlock",
        layers: Tuple[int, ...] = (1, 1, 1),
        planes: Tuple[int, ...] = (32, 64, 64),
        pooling: str = "gem",
    ) -> None:
        """MinkLoc3D: Point Cloud Based Large-Scale Place Recognition.

        Paper: https://arxiv.org/abs/2011.04530
        Code is adopted from the original repository: https://github.com/jac99/MinkLoc3Dv2, MIT License

        Args:
            in_channels (int): Number of input channels. Defaults to 1.
            out_channels (int): Number of output channels. Defaults to 256.
            num_top_down (int): Number of top-down blocks. Defaults to 1.
            conv0_kernel_size (int): Kernel size of the first convolution. Defaults to 5.
            block (str): Type of the network block. Defaults to "BasicBlock".
            layers (Tuple[int, ...]): Number of blocks in each layer. Defaults to (1, 1, 1).
            planes (Tuple[int, ...]): Number of channels in each layer. Defaults to (32, 64, 64).
            pooling (str): Type of pooling. Defaults to "gem".

        Raises:
            NotImplementedError: If given pooling method is unknown.
        """
        feature_extractor = MinkResNetFPNFeatureExtractor(
            in_channels, out_channels, num_top_down, conv0_kernel_size, block, layers, planes
        )
        if pooling == "gem":
            pooling = MinkGeM()
        else:
            raise NotImplementedError("Unknown pooling method: {}".format(pooling))

        super().__init__(
            backbone=feature_extractor,
            head=pooling,
        )
