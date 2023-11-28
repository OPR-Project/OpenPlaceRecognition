"""ConvNeXt-based image feature extractors."""
from torch import Tensor, nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


class ConvNeXtTinyFeatureExtractor(nn.Module):
    """ConvNeXt-Tiny image feature extractor."""

    def __init__(
        self,
        in_channels: int = 3,
        pretrained: bool = True,
    ) -> None:
        """ConvNeXt-Tiny image feature extractor.

        Args:
            in_channels (int): Number of input channels. Defaults to 3.
            pretrained (bool): Whether to load ImageNet-pretrained model. Defaults to True.

        Raises:
            ValueError: If `in_channels` is not 3 and `pretrained` is True.
        """
        super().__init__()

        if in_channels != 3 and pretrained:
            raise ValueError("Pretrained models are only available for 3-channel images")

        model = convnext_tiny(weights=(ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None))
        self.feature_extractor = model.features

        # change input conv to accept n-channel images
        if in_channels != 3:
            self.feature_extractor[0][0] = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.feature_extractor[0][0].out_channels,
                kernel_size=self.feature_extractor[0][0].kernel_size,
                stride=self.feature_extractor[0][0].stride,
                padding=self.feature_extractor[0][0].padding,
                dilation=self.feature_extractor[0][0].dilation,
                groups=self.feature_extractor[0][0].groups,
                bias=True,
                padding_mode=self.feature_extractor[0][0].padding_mode,
                device=next(self.feature_extractor[0][0].parameters()).device,
                dtype=next(self.feature_extractor[0][0].parameters()).dtype,
            )

    def forward(self, image: Tensor) -> Tensor:  # noqa: D102
        return self.feature_extractor(image)
