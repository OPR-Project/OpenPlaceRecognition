import torch
from .base import ImageModel
from opr.modules.gem import GlobalAvgPooling, GlobalMaxPooling
from opr.modules.feature_extractors.theia import TheiaFeatureExtractor

class Theia(ImageModel):
    def __init__(
            self,
            feat_type: str="theia", 
            device: str="cpu",
            pooling: str="gap",
            weights_path = None,
        ) -> None:

        feature_extractor = TheiaFeatureExtractor(
            feat_type=feat_type,
            device=device
        )

        if pooling == "gap":
            pooling = GlobalAvgPooling()
        elif pooling == "gmp":
            pooling = GlobalMaxPooling()
        else:
            raise NotImplementedError("Unknown pooling method: {}".format(pooling))

        super().__init__(
            backbone=feature_extractor,
            head=pooling,
            fusion=None
        )
        

