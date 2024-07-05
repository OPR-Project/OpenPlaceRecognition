from torch.nn.modules import Module
from opr.modules.cosplace import CosPlace
from opr.modules.feature_extractors import ResNet50FPNFeatureExtractor

from .base import ImageModel


class CosPlaceModel(ImageModel):
    """CosPlace: Rethinking Visual Geo-localization for Large-Scale Applications

    Paper: https://arxiv.org/abs/2204.02287
    """

    def __init__(self, backbone: str = "resnet50"):
        """Initialize CosPlace Image Model.

        Args:
            backbone (str): Backbone architecture. Defaults to "resnet50".

        Raises:
            NotImplementedError: If given backbone is unknown.
        """
        if backbone == "resnet50":
            backbone = ResNet50FPNFeatureExtractor()
        else:
            raise NotImplementedError(f"Backbone {backbone} is not supported.")
        head = CosPlace()
        super().__init__(
            backbone=backbone,
            head=head,
        )
