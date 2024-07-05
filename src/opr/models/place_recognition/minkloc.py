"""Implementations of MinkLoc models."""
from typing import Tuple

from opr.modules import Concat, MinkGeM
from opr.modules.feature_extractors import MinkResNetFPNFeatureExtractor

from .base import CloudModel, LateFusionModel
from .resnet import ResNet18


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


class MinkLoc3Dv2(MinkLoc3D):
    """Improving Point Cloud Based Place Recognition with Ranking-based Loss and Large Batch Training.

    Paper: https://arxiv.org/abs/2203.00972
    Code is adopted from the original repository: https://github.com/jac99/MinkLoc3Dv2, MIT License
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 256,
        num_top_down: int = 2,
        conv0_kernel_size: int = 5,
        block: str = "ECABasicBlock",
        layers: Tuple[int, ...] = (1, 1, 1, 1),
        planes: Tuple[int, ...] = (64, 128, 64, 32),
        pooling: str = "gem",
    ) -> None:
        """Improving Point Cloud Based Place Recognition with Ranking-based Loss and Large Batch Training.

        Paper: https://arxiv.org/abs/2203.00972
        Code is adopted from the original repository: https://github.com/jac99/MinkLoc3Dv2, MIT License

        Args:
            in_channels (int): Number of input channels. Defaults to 1.
            out_channels (int): Number of output channels. Defaults to 256.
            num_top_down (int): Number of top-down blocks. Defaults to 2.
            conv0_kernel_size (int): Kernel size of the first convolution. Defaults to 5.
            block (str): Type of the network block. Defaults to "ECABasicBlock".
            layers (Tuple[int, ...]): Number of blocks in each layer. Defaults to (1, 1, 1, 1).
            planes (Tuple[int, ...]): Number of channels in each layer. Defaults to (64, 128, 64, 32).
            pooling (str): Type of pooling. Defaults to "gem".
        """
        super().__init__(
            in_channels,
            out_channels,
            num_top_down,
            conv0_kernel_size,
            block,
            layers,
            planes,
            pooling,
        )


class MinkLocMultimodal(LateFusionModel):
    """MinkLoc++: Lidar and Monocular Image Fusion for Place Recognition.

    Paper: https://arxiv.org/pdf/2104.05327.pdf
    Code is adopted from the original repository: https://github.com/jac99/MinkLocMultimodal, MIT License
    """

    def __init__(
        self,
        lidar_in_channels: int = 1,
        lidar_out_channels: int = 256,
        lidar_num_top_down: int = 2,
        lidar_conv0_kernel_size: int = 5,
        lidar_block: str = "ECABasicBlock",
        lidar_layers: Tuple[int, ...] = (1, 1, 1, 1),
        lidar_planes: Tuple[int, ...] = (64, 128, 64, 32),
        lidar_pooling: str = "gem",
        image_in_channels: int = 3,
        image_out_channels: int = 256,
        image_num_top_down: int = 0,
        image_pooling: str = "gem",
        image_pretrained: bool = True,
        fusion_type: str = "concat",
    ) -> None:
        """MinkLoc++: Lidar and Monocular Image Fusion for Place Recognition.

        Paper: https://arxiv.org/pdf/2104.05327.pdf
        Code is adopted from the original repository: https://github.com/jac99/MinkLocMultimodal, MIT License

        Args:
            lidar_in_channels (int): Number of input channels. Defaults to 1.
            lidar_out_channels (int): Number of output channels. Defaults to 256.
            lidar_num_top_down (int): Number of top-down blocks. Defaults to 2.
            lidar_conv0_kernel_size (int): Kernel size of the first convolution. Defaults to 5.
            lidar_block (str): Type of the network block. Defaults to "ECABasicBlock".
            lidar_layers (Tuple[int, ...]): Number of blocks in each layer. Defaults to (1, 1, 1, 1).
            lidar_planes (Tuple[int, ...]): Number of channels in each layer. Defaults to (64, 128, 64, 32).
            lidar_pooling (str): Type of pooling. Defaults to "gem".
            image_in_channels (int): Number of input channels. Defaults to 3.
            image_out_channels (int): Number of output channels. Defaults to 256.
            image_num_top_down (int): Number of top-down layers. Defaults to 0.
            image_pooling (str): Pooling method to use. Currently only "gem" is supported. Defaults to "gem".
            image_pretrained (bool): Whether to use pretrained weights. Defaults to True.

        Raises:
            NotImplementedError: If given pooling method is unknown.
        """

        cloud_module = MinkLoc3Dv2(
            in_channels=lidar_in_channels,
            out_channels=lidar_out_channels,
            num_top_down=lidar_num_top_down,
            conv0_kernel_size=lidar_conv0_kernel_size,
            block=lidar_block,
            layers=lidar_layers,
            planes=lidar_planes,
            pooling=lidar_pooling,
        )
        image_module = ResNet18(
            in_channels=image_in_channels,
            out_channels=image_out_channels,
            num_top_down=image_num_top_down,
            pooling=image_pooling,
            pretrained=image_pretrained,
        )
        if fusion_type == "concat":
            fusion_module = Concat()
        else:
            raise NotImplementedError("Unknown fusion type in MinkLocMultimodal: {}".format(fusion_type))
        super().__init__(
            image_module=image_module,
            cloud_module=cloud_module,
            fusion_module=fusion_module,
        )
