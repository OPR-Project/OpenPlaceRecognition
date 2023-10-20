"""ResNet-based image feature extractors."""
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18FPNFeatureExtractor(nn.Module):
    """ResNet18 image feature extractor with FPN block.

    The code is adopted from the repository: https://github.com/jac99/MinkLocMultimodal, MIT License
    """

    def __init__(
        self,
        in_channels: int = 3,
        lateral_dim: int = 256,
        fh_num_bottom_up: int = 4,
        fh_num_top_down: int = 0,
        pretrained: bool = True,
    ) -> None:
        """ResNet18 image feature extractor with FPN block.

        Args:
            in_channels (int): Number of input channels. Defaults to 3.
            lateral_dim (int): Output dimension for lateral connections. Defaults to 256.
            fh_num_bottom_up (int): Number of bottom-up steps. Defaults to 4.
            fh_num_top_down (int): Number of top-down steps. Defaults to 0.
            pretrained (bool): Whether to load ImageNet-pretrained model. Defaults to True.

        Raises:
            ValueError: If `in_channels` is not 3 and `pretrained` is True.
        """
        super().__init__()

        # Number of channels in each layer of ResNet18
        layers = (64, 64, 128, 256, 512)

        if not (0 < fh_num_bottom_up <= 5):
            raise ValueError("Number of bottom-up steps must be in range [1, 5]")
        if not (0 <= fh_num_top_down < fh_num_bottom_up):
            raise ValueError("Number of top-down steps must be in range [0, fh_num_bottom_up)")
        if in_channels != 3 and pretrained:
            raise ValueError("Pretrained models are only available for 3-channel images")

        self.lateral_dim = lateral_dim
        self.fh_num_bottom_up = fh_num_bottom_up
        self.fh_num_top_down = fh_num_top_down

        model = resnet18(weights=(ResNet18_Weights.IMAGENET1K_V1 if pretrained else None))
        # Last 2 blocks are AdaptiveAvgPool2d and Linear (get rid of them)
        self.resnet_fe = nn.ModuleList(list(model.children())[: 3 + self.fh_num_bottom_up])

        # change input conv to accept n-channel images
        if in_channels != 3:
            self.resnet_fe[0] = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.resnet_fe[0].out_channels,
                kernel_size=self.resnet_fe[0].kernel_size,
                stride=self.resnet_fe[0].stride,
                padding=self.resnet_fe[0].padding,
                dilation=self.resnet_fe[0].dilation,
                groups=self.resnet_fe[0].groups,
                bias=self.resnet_fe[0].bias,
                padding_mode=self.resnet_fe[0].padding_mode,
                device=self.resnet_fe[0].device,
                dtype=self.resnet_fe[0].dtype,
            )

        # Lateral connections and top-down pass for the feature extraction head
        self.fh_tconvs = nn.ModuleDict()  # Top-down transposed convolutions in feature head
        self.fh_conv1x1 = nn.ModuleDict()  # 1x1 convolutions in lateral connections to the feature head
        for i in range(self.fh_num_bottom_up - self.fh_num_top_down, self.fh_num_bottom_up):
            self.fh_conv1x1[str(i + 1)] = nn.Conv2d(
                in_channels=layers[i], out_channels=self.lateral_dim, kernel_size=1
            )
            self.fh_tconvs[str(i + 1)] = nn.ConvTranspose2d(
                in_channels=self.lateral_dim, out_channels=self.lateral_dim, kernel_size=2, stride=2
            )

        # One more lateral connection
        temp = self.fh_num_bottom_up - self.fh_num_top_down
        self.fh_conv1x1[str(temp)] = nn.Conv2d(
            in_channels=layers[temp - 1], out_channels=self.lateral_dim, kernel_size=1
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

        if len(feature_maps) != self.fh_num_bottom_up:
            raise RuntimeError(
                f"Number of feature maps ({len(feature_maps)}) does not match fh_num_bottom_up"
            )
        # x is (batch_size, 512, H=20, W=15) for 640x480 input image

        # FEATURE HEAD TOP-DOWN PASS
        xf = self.fh_conv1x1[str(self.fh_num_bottom_up)](feature_maps[str(self.fh_num_bottom_up)])
        for i in range(self.fh_num_bottom_up, self.fh_num_bottom_up - self.fh_num_top_down, -1):
            xf = self.fh_tconvs[str(i)](xf)  # Upsample using transposed convolution
            xf = xf + self.fh_conv1x1[str(i - 1)](feature_maps[str(i - 1)])

        return xf
