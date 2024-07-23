"""Implementation of APGeM Image Model."""
from typing import Literal

from opr.modules import GeM
from opr.modules.feature_extractors import (
    ResNet18FPNFeatureExtractor,
    ResNet50FPNFeatureExtractor,
    VGG16FeatureExtractor,
)

from .base import ImageModel


class APGeMModel(ImageModel):
    """APGeM: 'Learning with Average Precision: Training Image Retrieval with a Listwise Loss'.

    Paper: https://arxiv.org/abs/1906.07589
    """

    def __init__(self, backbone: Literal["resnet18", "resnet50", "vgg16"] = "resnet50") -> None:
        """Initialize APGeM Image Model.

        Args:
            backbone (str): Backbone architecture. Defaults to "resnet50".

        Raises:
            NotImplementedError: If given backbone is unknown.
        """
        if backbone == "resnet18":
            backbone = ResNet18FPNFeatureExtractor()
        elif backbone == "resnet50":
            backbone = ResNet50FPNFeatureExtractor()
        elif backbone == "vgg16":
            backbone = VGG16FeatureExtractor()
        else:
            raise NotImplementedError(f"Backbone {backbone} is not supported.")
        head = GeM()
        super().__init__(
            backbone=backbone,
            head=head,
        )
