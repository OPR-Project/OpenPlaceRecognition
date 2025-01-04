"""Implementation of CosPlace model."""
from typing import Literal

from opr.modules.cosplace import CosPlace
from opr.modules.feature_extractors import (
    ResNet18FPNFeatureExtractor,
    ResNet50FPNFeatureExtractor,
    VGG16FeatureExtractor,
)

from .base import ImageModel


class CosPlaceModel(ImageModel):
    """CosPlace: Rethinking Visual Geo-localization for Large-Scale Applications.

    Paper: https://arxiv.org/abs/2204.02287
    """

    def __init__(
        self, backbone: Literal["resnet18", "resnet50", "vgg16"] = "resnet50", out_dim: int = 256
    ) -> None:
        """Initialize CosPlace Image Model.

        Args:
            backbone (str): Backbone architecture. Defaults to "resnet50".
            out_dim (int): Output dimension of the model. Defaults to 256.

        Raises:
            NotImplementedError: If given backbone is unknown.
        """
        if backbone == "resnet18":
            backbone = ResNet18FPNFeatureExtractor()
            dim = 256
        elif backbone == "resnet50":
            backbone = ResNet50FPNFeatureExtractor()
            dim = 256
        elif backbone == "vgg16":
            backbone = VGG16FeatureExtractor()
            dim = 512
        else:
            raise NotImplementedError(f"Backbone {backbone} is not supported.")
        head = CosPlace(in_dim=dim, out_dim=out_dim)
        super().__init__(
            backbone=backbone,
            head=head,
        )
