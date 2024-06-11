"""Implementation of feature extraction model from SVT-Net.

Citation:
    Fan, Zhaoxin, et al. "Svt-net: Super light-weight sparse voxel transformer
    for large scale place recognition." Proceedings of the AAAI Conference on Artificial Intelligence.
    Vol. 36. No. 1. 2022.

Source: https://github.com/ZhenboSong/SVTNet
Paper: https://arxiv.org/abs/2105.00149
"""
from loguru import logger
from torch import nn

from opr.modules.eca import MinkECABasicBlock as ECABasicBlock
from opr.modules.feature_extractors.mink_resnet import MinkResNetBase
from opr.modules.svt import ASVT, CSVT

try:
    import MinkowskiEngine as ME  # type: ignore
    from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck

    minkowski_available = True
except ImportError:
    logger.warning("MinkowskiEngine is not installed. Some features may not be available.")
    minkowski_available = False


class SVTNetFeatureExtractor(MinkResNetBase):
    """Feature extraction model from SVT-Net.

    Source: https://github.com/ZhenboSong/SVTNet
    """

    sparse = True

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 256,
        conv0_kernel_size: int = 5,
        block: str = "ECABasicBlock",
        asvt: bool = True,
        csvt: bool = True,
        layers: tuple[int, ...] = (1, 1, 1),
        planes: tuple[int, ...] = (32, 64, 64),
    ) -> None:
        """Feature extraction model from SVT-Net.

        Args:
            in_channels (int): Number of input channels. Defaults to 1.
            out_channels (int): Number of output channels. Defaults to 256.
            conv0_kernel_size (int): Kernel size of the first convolution. Defaults to 5.
            block (str): Type of the network block. Defaults to "ECABasicBlock".
            asvt (bool): Whether to use ASVT. Defaults to True.
            csvt (bool): Whether to use CSVT. Defaults to True.
            layers (tuple[int, ...]): Number of blocks in each layer. Defaults to (1, 1, 1).
            planes (tuple[int, ...]): Number of channels in each layer. Defaults to (32, 64, 64).

        Raises:
            RuntimeError: If MinkowskiEngine is not installed.
            ValueError: If the number of layers and planes is not the same.
            ValueError: If the number of layers is less than 1.
        """
        if not minkowski_available:
            raise RuntimeError(
                "MinkowskiEngine is not installed. SVTNetFeatureExtractor requires MinkowskiEngine."
            )
        if not len(layers) == len(planes):
            raise ValueError("The number of layers and planes must be the same.")
        if not 1 <= len(layers):
            raise ValueError("The number of layers must be at least 1.")
        self.num_bottom_up = len(layers)
        self.conv0_kernel_size = conv0_kernel_size
        self.block = self._create_resnet_block(block_name=block)
        self.layers = layers
        self.planes = planes
        self.lateral_dim = out_channels
        self.init_dim = planes[0]
        self.is_asvt = asvt
        self.is_csvt = csvt
        MinkResNetBase.__init__(self, in_channels, out_channels, dimension=3)

    def _create_resnet_block(self, block_name: str) -> nn.Module:
        if block_name == "BasicBlock":
            block_module = BasicBlock
        elif block_name == "Bottleneck":
            block_module = Bottleneck
        elif block_name == "ECABasicBlock":
            block_module = ECABasicBlock
        else:
            raise NotImplementedError(f"Unsupported network block: {block_name}")

        return block_module

    def _network_initialization(self, in_channels: int, out_channels: int, dimension: int) -> None:
        self.convs = nn.ModuleList()  # Bottom-up convolutional blocks with stride=2
        self.bn = nn.ModuleList()  # Bottom-up BatchNorms
        self.blocks = nn.ModuleList()  # Bottom-up blocks
        self.tconvs = nn.ModuleList()  # Top-down tranposed convolutions
        self.conv1x1 = nn.ModuleList()  # 1x1 convolutions in lateral connections

        # The first convolution is special case, with kernel size = 5
        self.inplanes = self.planes[0]
        self.conv0 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=self.conv0_kernel_size, dimension=dimension
        )
        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        for plane, layer in zip(self.planes, self.layers):
            self.convs.append(
                ME.MinkowskiConvolution(
                    self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=dimension
                )
            )
            self.bn.append(ME.MinkowskiBatchNorm(self.inplanes))
            self.blocks.append(self._make_layer(self.block, plane, layer))

        self.conv1x1.append(
            ME.MinkowskiConvolution(
                self.inplanes, self.lateral_dim, kernel_size=1, stride=1, dimension=dimension
            )
        )

        # before_lateral_dim=plane
        after_reduction = max(self.lateral_dim / 8, 8)
        reduction = int(self.lateral_dim // after_reduction)

        if self.is_asvt:
            self.asvt = ASVT(self.lateral_dim, reduction)
        if self.is_csvt:
            self.csvt = CSVT(self.lateral_dim, 8)

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:  # noqa: D102
        # *** BOTTOM-UP PASS ***
        # First bottom-up convolution is special (with bigger stride)
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)

        # BOTTOM-UP PASS
        for _, (conv, bn, block) in enumerate(zip(self.convs, self.bn, self.blocks)):
            x = conv(x)  # Decreases spatial resolution (conv stride=2)
            x = bn(x)
            x = self.relu(x)
            x = block(x)

        x = self.conv1x1[0](x)

        if self.is_csvt:
            x_csvt = self.csvt(x)
        if self.is_asvt:
            x_asvt = self.asvt(x)

        if self.is_csvt and self.is_asvt:
            x = x_csvt + x_asvt
        elif self.is_csvt:
            x = x_csvt
        elif self.is_asvt:
            x = x_asvt

        return x
