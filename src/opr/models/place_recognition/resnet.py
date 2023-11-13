"""ResNet image models for Place Recognition."""
from opr.modules import GeM
from opr.modules.feature_extractors import ResNet18FPNFeatureExtractor

from .base import ImageModel


class ResNet18(ImageModel):
    """ResNet18 image model for Place Recognition."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 256,
        num_top_down: int = 0,
        pooling: str = "gem",
        pretrained: bool = True,
    ) -> None:
        """ResNet18 image model for Place Recognition.

        Args:
            in_channels (int): Number of input channels. Defaults to 3.
            out_channels (int): Number of output channels. Defaults to 256.
            num_top_down (int): Number of top-down layers. Defaults to 0.
            pooling (str): Pooling method to use. Currently only "gem" is supported. Defaults to "gem".
            pretrained (bool): Whether to use pretrained weights. Defaults to True.

        Raises:
            NotImplementedError: If given pooling method is unknown.
        """
        feature_extractor = ResNet18FPNFeatureExtractor(
            in_channels=in_channels,
            lateral_dim=out_channels,
            fh_num_bottom_up=4,
            fh_num_top_down=num_top_down,
            pretrained=pretrained,
        )
        if pooling == "gem":
            pooling = GeM()
        else:
            raise NotImplementedError("Unknown pooling method: {}".format(pooling))
        super().__init__(
            backbone=feature_extractor,
            head=pooling,
            fusion=None,
        )
