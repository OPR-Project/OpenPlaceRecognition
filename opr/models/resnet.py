"""ResNet-based image feature extractors."""
from typing import Tuple

from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18

from opr.models.base_models import ImageFeatureExtractor


class ResNet18FPNExtractor(ImageFeatureExtractor):
    """ResNet18 feature extractor with FPN block.

    The code is adopted from the repository: https://github.com/jac99/MinkLocMultimodal, MIT License
    """

    layers: Tuple[int, ...] = (64, 64, 128, 256, 512)

    def __init__(
        self,
        lateral_dim: int = 256,
        fh_num_bottom_up: int = 4,
        fh_num_top_down: int = 0,
        pretrained: bool = True,
    ) -> None:
        """ResNet18 feature extractor with FPN block.

        Args:
            lateral_dim (int): Output dimension for lateral connections. Defaults to 256.
            fh_num_bottom_up (int): Number of bottom-up steps. Defaults to 4.
            fh_num_top_down (int): Number of top-down steps. Defaults to 0.
            pretrained (bool): Whether to load ImageNet-pretrained model. Defaults to True.
        """
        super().__init__()
        assert 0 < fh_num_bottom_up <= 5
        assert 0 <= fh_num_top_down < fh_num_bottom_up

        self.lateral_dim = lateral_dim
        self.fh_num_bottom_up = fh_num_bottom_up
        self.fh_num_top_down = fh_num_top_down

        model = resnet18(weights=(ResNet18_Weights.IMAGENET1K_V1 if pretrained else None))
        # Last 2 blocks are AdaptiveAvgPool2d and Linear (get rid of them)
        self.resnet_fe = nn.ModuleList(list(model.children())[: 3 + self.fh_num_bottom_up])

        # Lateral connections and top-down pass for the feature extraction head
        self.fh_tconvs = nn.ModuleDict()  # Top-down transposed convolutions in feature head
        self.fh_conv1x1 = nn.ModuleDict()  # 1x1 convolutions in lateral connections to the feature head
        for i in range(self.fh_num_bottom_up - self.fh_num_top_down, self.fh_num_bottom_up):
            self.fh_conv1x1[str(i + 1)] = nn.Conv2d(
                in_channels=self.layers[i], out_channels=self.lateral_dim, kernel_size=1
            )
            self.fh_tconvs[str(i + 1)] = nn.ConvTranspose2d(
                in_channels=self.lateral_dim, out_channels=self.lateral_dim, kernel_size=2, stride=2
            )

        # One more lateral connection
        temp = self.fh_num_bottom_up - self.fh_num_top_down
        self.fh_conv1x1[str(temp)] = nn.Conv2d(
            in_channels=self.layers[temp - 1], out_channels=self.lateral_dim, kernel_size=1
        )

    def forward(self, image: Tensor) -> Tensor:  # noqa: D102
        x = image
        feature_maps = {}

        # 0, 1, 2, 3 = first layers: Conv2d, BatchNorm, ReLu, MaxPool2d
        x = self.resnet_fe[0](x)
        x = self.resnet_fe[1](x)
        x = self.resnet_fe[2](x)
        x = self.resnet_fe[3](x)
        feature_maps["1"] = x

        # sequential blocks, build from BasicBlock or Bottleneck blocks
        for i in range(4, self.fh_num_bottom_up + 3):
            x = self.resnet_fe[i](x)
            feature_maps[str(i - 2)] = x

        assert len(feature_maps) == self.fh_num_bottom_up
        # x is (batch_size, 512, H=20, W=15) for 640x480 input image

        # FEATURE HEAD TOP-DOWN PASS
        xf = self.fh_conv1x1[str(self.fh_num_bottom_up)](feature_maps[str(self.fh_num_bottom_up)])
        for i in range(self.fh_num_bottom_up, self.fh_num_bottom_up - self.fh_num_top_down, -1):
            xf = self.fh_tconvs[str(i)](xf)  # Upsample using transposed convolution
            xf = xf + self.fh_conv1x1[str(i - 1)](feature_maps[str(i - 1)])

        return xf

class ResNet18FPNExtractorMono(ImageFeatureExtractor):
    """ResNet18 feature extractor with FPN block adopted to single-channel images (semantic masks).

    The code is adopted from the repository: https://github.com/jac99/MinkLocMultimodal, MIT License
    """

    layers: Tuple[int, ...] = (64, 64, 128, 256, 512)

    def __init__(
        self,
        lateral_dim: int = 256,
        fh_num_bottom_up: int = 4,
        fh_num_top_down: int = 0,
        pretrained: bool = False,
    ) -> None:
        """ResNet18 feature extractor with FPN block.

        Args:
            lateral_dim (int): Output dimension for lateral connections. Defaults to 256.
            fh_num_bottom_up (int): Number of bottom-up steps. Defaults to 4.
            fh_num_top_down (int): Number of top-down steps. Defaults to 0.
            pretrained (bool): Whether to load ImageNet-pretrained model. Defaults to False (Not implemented).
        """
        super().__init__()

        if pretrained:
            raise NotImplementedError(f"There are currently no pre-trained models for semantic masks :c")

        assert 0 < fh_num_bottom_up <= 5
        assert 0 <= fh_num_top_down < fh_num_bottom_up

        self.lateral_dim = lateral_dim
        self.fh_num_bottom_up = fh_num_bottom_up
        self.fh_num_top_down = fh_num_top_down

        # model = resnet18(weights=(ResNet18_Weights.IMAGENET1K_V1 if pretrained else None))
        model = resnet18()
        # Last 2 blocks are AdaptiveAvgPool2d and Linear (get rid of them)
        self.resnet_fe = nn.ModuleList(list(model.children())[: 3 + self.fh_num_bottom_up])

        #* Fix: semantuc masks has only single channel
        self.resnet_fe[0] = nn.Conv2d(
            in_channels=1,
            out_channels=self.resnet_fe[0].out_channels,
            kernel_size=self.resnet_fe[0].kernel_size,
            stride=self.resnet_fe[0].kernel_size,
            padding=self.resnet_fe[0].padding,
            dilation=self.resnet_fe[0].dilation,
            bias=self.resnet_fe[0].bias,
            groups=self.resnet_fe[0].groups
        )

        # Lateral connections and top-down pass for the feature extraction head
        self.fh_tconvs = nn.ModuleDict()  # Top-down transposed convolutions in feature head
        self.fh_conv1x1 = nn.ModuleDict()  # 1x1 convolutions in lateral connections to the feature head
        for i in range(self.fh_num_bottom_up - self.fh_num_top_down, self.fh_num_bottom_up):
            self.fh_conv1x1[str(i + 1)] = nn.Conv2d(
                in_channels=self.layers[i], out_channels=self.lateral_dim, kernel_size=1
            )
            self.fh_tconvs[str(i + 1)] = nn.ConvTranspose2d(
                in_channels=self.lateral_dim, out_channels=self.lateral_dim, kernel_size=2, stride=2
            )

        # One more lateral connection
        temp = self.fh_num_bottom_up - self.fh_num_top_down
        self.fh_conv1x1[str(temp)] = nn.Conv2d(
            in_channels=self.layers[temp - 1], out_channels=self.lateral_dim, kernel_size=1
        )

    def forward(self, image: Tensor) -> Tensor:  # noqa: D102
        x = image
        feature_maps = {}

        # 0, 1, 2, 3 = first layers: Conv2d, BatchNorm, ReLu, MaxPool2d
        x = self.resnet_fe[0](x)
        x = self.resnet_fe[1](x)
        x = self.resnet_fe[2](x)
        x = self.resnet_fe[3](x)
        feature_maps["1"] = x

        # sequential blocks, build from BasicBlock or Bottleneck blocks
        for i in range(4, self.fh_num_bottom_up + 3):
            x = self.resnet_fe[i](x)
            feature_maps[str(i - 2)] = x

        assert len(feature_maps) == self.fh_num_bottom_up
        # x is (batch_size, 512, H=20, W=15) for 640x480 input image

        # FEATURE HEAD TOP-DOWN PASS
        xf = self.fh_conv1x1[str(self.fh_num_bottom_up)](feature_maps[str(self.fh_num_bottom_up)])
        for i in range(self.fh_num_bottom_up, self.fh_num_bottom_up - self.fh_num_top_down, -1):
            xf = self.fh_tconvs[str(i)](xf)  # Upsample using transposed convolution
            xf = xf + self.fh_conv1x1[str(i - 1)](feature_maps[str(i - 1)])

        return xf
    

class ResNet18FPNExtractorOneHot(ImageFeatureExtractor):
    """ResNet18 feature extractor with FPN block adopted to single-channel images (semantic masks).

    The code is adopted from the repository: https://github.com/jac99/MinkLocMultimodal, MIT License
    """

    layers: Tuple[int, ...] = (64, 64, 128, 256, 512)

    def __init__(
        self,
        lateral_dim: int = 256,
        fh_num_bottom_up: int = 4,
        fh_num_top_down: int = 0,
        pretrained: bool = False,
    ) -> None:
        """ResNet18 feature extractor with FPN block.

        Args:
            lateral_dim (int): Output dimension for lateral connections. Defaults to 256.
            fh_num_bottom_up (int): Number of bottom-up steps. Defaults to 4.
            fh_num_top_down (int): Number of top-down steps. Defaults to 0.
            pretrained (bool): Whether to load ImageNet-pretrained model. Defaults to False (Not implemented).
        """
        super().__init__()

        if pretrained:
            raise NotImplementedError(f"There are currently no pre-trained models for semantic masks :c")

        assert 0 < fh_num_bottom_up <= 5
        assert 0 <= fh_num_top_down < fh_num_bottom_up

        self.lateral_dim = lateral_dim
        self.fh_num_bottom_up = fh_num_bottom_up
        self.fh_num_top_down = fh_num_top_down

        # model = resnet18(weights=(ResNet18_Weights.IMAGENET1K_V1 if pretrained else None))
        model = resnet18()
        # Last 2 blocks are AdaptiveAvgPool2d and Linear (get rid of them)
        self.resnet_fe = nn.ModuleList(list(model.children())[: 3 + self.fh_num_bottom_up])

        #* Fix: semantuc masks has only single channel
        self.resnet_fe[0] = nn.Conv2d(
            in_channels=65, #! magic number len(stuff_classes) in augmentations.py
            out_channels=self.resnet_fe[0].out_channels,
            kernel_size=self.resnet_fe[0].kernel_size,
            stride=self.resnet_fe[0].kernel_size,
            padding=self.resnet_fe[0].padding,
            dilation=self.resnet_fe[0].dilation,
            bias=self.resnet_fe[0].bias,
            groups=self.resnet_fe[0].groups
        )

        # Lateral connections and top-down pass for the feature extraction head
        self.fh_tconvs = nn.ModuleDict()  # Top-down transposed convolutions in feature head
        self.fh_conv1x1 = nn.ModuleDict()  # 1x1 convolutions in lateral connections to the feature head
        for i in range(self.fh_num_bottom_up - self.fh_num_top_down, self.fh_num_bottom_up):
            self.fh_conv1x1[str(i + 1)] = nn.Conv2d(
                in_channels=self.layers[i], out_channels=self.lateral_dim, kernel_size=1
            )
            self.fh_tconvs[str(i + 1)] = nn.ConvTranspose2d(
                in_channels=self.lateral_dim, out_channels=self.lateral_dim, kernel_size=2, stride=2
            )

        # One more lateral connection
        temp = self.fh_num_bottom_up - self.fh_num_top_down
        self.fh_conv1x1[str(temp)] = nn.Conv2d(
            in_channels=self.layers[temp - 1], out_channels=self.lateral_dim, kernel_size=1
        )

    def forward(self, image: Tensor) -> Tensor:  # noqa: D102
        x = image
        feature_maps = {}

        # 0, 1, 2, 3 = first layers: Conv2d, BatchNorm, ReLu, MaxPool2d
        x = self.resnet_fe[0](x)
        x = self.resnet_fe[1](x)
        x = self.resnet_fe[2](x)
        x = self.resnet_fe[3](x)
        feature_maps["1"] = x

        # sequential blocks, build from BasicBlock or Bottleneck blocks
        for i in range(4, self.fh_num_bottom_up + 3):
            x = self.resnet_fe[i](x)
            feature_maps[str(i - 2)] = x

        assert len(feature_maps) == self.fh_num_bottom_up
        # x is (batch_size, 512, H=20, W=15) for 640x480 input image

        # FEATURE HEAD TOP-DOWN PASS
        xf = self.fh_conv1x1[str(self.fh_num_bottom_up)](feature_maps[str(self.fh_num_bottom_up)])
        for i in range(self.fh_num_bottom_up, self.fh_num_bottom_up - self.fh_num_top_down, -1):
            xf = self.fh_tconvs[str(i)](xf)  # Upsample using transposed convolution
            xf = xf + self.fh_conv1x1[str(i - 1)](feature_maps[str(i - 1)])

        return xf
