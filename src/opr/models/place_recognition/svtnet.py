"""SVT-Net: Super Light-Weight Sparse Voxel Transformer for Large Scale Place Recognition.

Citation:
    Fan, Zhaoxin, et al. "Svt-net: Super light-weight sparse voxel transformer
    for large scale place recognition." Proceedings of the AAAI Conference on Artificial Intelligence.
    Vol. 36. No. 1. 2022.

Source: https://github.com/ZhenboSong/SVTNet
Paper: https://arxiv.org/abs/2105.00149
"""
from opr.modules import MinkGeM
from opr.modules.feature_extractors import SVTNetFeatureExtractor

from .base import CloudModel
from typing import Tuple, Dict


class SVTNet(CloudModel):
    """SVT-Net: Super Light-Weight Sparse Voxel Transformer for Large Scale Place Recognition.

    Citation:
        Fan, Zhaoxin, et al. "Svt-net: Super light-weight sparse voxel transformer
        for large scale place recognition." Proceedings of the AAAI Conference on Artificial Intelligence.
        Vol. 36. No. 1. 2022.

    Source: https://github.com/ZhenboSong/SVTNet
    Paper: https://arxiv.org/abs/2105.00149
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 256,
        conv0_kernel_size: int = 5,
        block: str = "ECABasicBlock",
        asvt: bool = True,
        csvt: bool = True,
        layers: Tuple[int, ...] = (1, 1, 1),
        planes: Tuple[int, ...] = (32, 64, 64),
        pooling: str = "gem",
    ) -> None:
        """SVT-Net: Super Light-Weight Sparse Voxel Transformer for Large Scale Place Recognition.

        Args:
            in_channels (int): Number of input channels. Defaults to 1.
            out_channels (int): Number of output channels. Defaults to 256.
            conv0_kernel_size (int): Kernel size of the first convolution. Defaults to 5.
            block (str): Type of the network block. Defaults to "ECABasicBlock".
            asvt (bool): Whether to use ASVT. Defaults to True.
            csvt (bool): Whether to use CSVT. Defaults to True.
            layers (Tuple[int, ...]): Number of blocks in each layer. Defaults to (1, 1, 1).
            planes (Tuple[int, ...]): Number of channels in each layer. Defaults to (32, 64, 64).
            pooling (str): Type of pooling. Defaults to "gem".

        Raises:
            NotImplementedError: If given pooling method is unknown.
        """
        feature_extractor = SVTNetFeatureExtractor(
            in_channels, out_channels, conv0_kernel_size, block, asvt, csvt, layers, planes
        )
        if pooling == "gem":
            pooling_head = MinkGeM()
        else:
            raise NotImplementedError("Unknown pooling method: {}".format(pooling))

        super().__init__(
            backbone=feature_extractor,
            head=pooling_head,
        )
