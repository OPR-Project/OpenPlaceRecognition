"""Implementation of APGeM Image Model."""
from opr.modules import GeM
from opr.modules.feature_extractors import ResNet50FPNFeatureExtractor

from .base import ImageModel


class APGeMImageModel(ImageModel):
    """APGeM: 'Learning with Average Precision: Training Image Retrieval with a Listwise Loss'.

    Paper: https://arxiv.org/abs/1906.07589
    """

    def __init__(self, backbone: str = "resnet50") -> None:
        """Initialize APGeM Image Model.

        Args:
            backbone (str): Backbone architecture. Defaults to "resnet50".

        Raises:
            NotImplementedError: If given backbone is unknown.
        """
        if backbone == "resnet50":
            backbone = ResNet50FPNFeatureExtractor()
        else:
            raise NotImplementedError(f"Backbone {backbone} is not supported.")
        head = GeM()
        super().__init__(
            backbone=backbone,
            head=head,
        )
