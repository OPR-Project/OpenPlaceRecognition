"""VGG-based image feature extractors."""
from torch import Tensor, nn
from torchvision.models import VGG16_Weights, vgg16


class VGGFeatureExtractor(nn.Module):
    """VGG-based image feature extractor."""

    def __init__(
        self,
        model: nn.Module,
        in_channels: int = 3,
        pretrained: bool = True,
    ) -> None:
        """VGG-based image feature extractor.

        Args:
            model (nn.Module): VGG model to use as feature extractor.
            in_channels (int): Number of input channels. Defaults to 3.
            pretrained (bool): Whether to use pretrained weights. Defaults to True.

        Raises:
            ValueError: If `pretrained` is True and `in_channels` is not 3.
        """
        super().__init__()

        if in_channels != 3 and pretrained:
            raise ValueError("Pretrained models are only available for 3-channel images")

        self.vgg_fe = model.features

        # change input conv to accept n-channel images
        if in_channels != 3:
            self.vgg_fe[0] = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.vgg_fe[0].out_channels,
                kernel_size=self.vgg_fe[0].kernel_size,
                stride=self.vgg_fe[0].stride,
                padding=self.vgg_fe[0].padding,
                dilation=self.vgg_fe[0].dilation,
                groups=self.vgg_fe[0].groups,
                bias=self.vgg_fe[0].bias,
                padding_mode=self.vgg_fe[0].padding_mode,
                device=next(self.vgg_fe[0].parameters()).device,
                dtype=next(self.vgg_fe[0].parameters()).dtype,
            )

    def forward(self, image: Tensor) -> Tensor:  # noqa: D102
        return self.vgg_fe(image)


class VGG16FeatureExtractor(VGGFeatureExtractor):
    """VGG-based image feature extractor."""

    def __init__(self, in_channels: int = 3, pretrained: bool = True) -> None:
        """VGG-based image feature extractor.

        Args:
            in_channels (int): Number of input channels. Defaults to 3.
            pretrained (bool): Whether to use pretrained weights. Defaults to True.
        """
        model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        super().__init__(model=model, in_channels=in_channels, pretrained=pretrained)
