"""Implementation of NetVLAD model."""
from typing import Literal

from opr.modules.feature_extractors import (
    ResNet18FPNFeatureExtractor,
    ResNet50FPNFeatureExtractor,
    VGG16FeatureExtractor,
)
from opr.modules.netvlad import NetVLAD

from .base import ImageModel


class NetVLADModel(ImageModel):
    """NetVLAD: CNN architecture for weakly supervised place recognition.

    Paper: https://arxiv.org/abs/1511.07247v3
    Code is adopted from the repository: https://github.com/Nanne/pytorch-NetVlad
    """

    def __init__(
        self,
        backbone: Literal["resnet18", "resnet50", "vgg16"] = "resnet18",
        num_clusters: int = 64,
        normalize_input: bool = True,
        vladv2: bool = False,
    ) -> None:
        """Initialize NetVLAD Image Model.

        Args:
            backbone (str): Backbone architecture. Defaults to "resnet18".
            num_clusters (int): Number of VLAD clusters. Defaults to 64.
            normalize_input (bool): Whether to normalize input data or not. Defaults to True.
            vladv2 (bool): Use vladv2 init params method. Defaults to False.

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

        head = NetVLAD(num_clusters, dim, normalize_input, vladv2)
        super().__init__(
            backbone=backbone,
            head=head,
        )
