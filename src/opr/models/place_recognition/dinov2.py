import torch
from torch import nn

from .base import ImageModel
from opr.modules.feature_extractors.dinov2 import DINO_V2_MODELS, DINO_FACETS, BOQ_MODELS
from opr.modules.gem import GlobalAvgPooling, GlobalMaxPooling
from opr.modules.feature_extractors.dinov2 import ViTBaseFeatureExtractor


class DinoV2(ImageModel):
    def __init__(
            self,
            dino_model: DINO_V2_MODELS,
            layer: int,
            facet: DINO_FACETS="token",
            use_cls=False,
            norm_descs=True,
            device: str = "cpu",
            pooling: str | None = "gap",
            weights_path = None,
        ) -> None:
        feature_extractor = ViTBaseFeatureExtractor(
            vit_type=dino_model,
            layer=layer,
            facet=facet,
            use_cls=use_cls,
            norm_descs=norm_descs,
            device=device,
        )

        if pooling == "gap":
            pooling = GlobalAvgPooling()
        elif pooling == "gmp":
            pooling = GlobalMaxPooling()
        elif pooling is None:
            pooling = nn.Identity()
        else:
            raise NotImplementedError("Unknown pooling method: {}".format(pooling))

        super().__init__(
            backbone=feature_extractor,
            head=pooling,
            fusion=None,
        )
