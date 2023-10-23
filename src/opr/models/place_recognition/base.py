"""Base meta-models for Place Recognition."""
from typing import Dict, Optional

import MinkowskiEngine as ME  # noqa: N817
from torch import Tensor, nn

from opr.modules import Concat


class ImageModel(nn.Module):
    """Meta-model for image-based Place Recognition. Combines feature extraction backbone and head modules."""

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        fusion: Optional[nn.Module] = None,
    ) -> None:
        """Meta-model for image-based Place Recognition.

        Args:
            backbone (ImageFeatureExtractor): Image feature extraction backbone.
            head (ImageHead): Image head module.
            fusion (FusionModule, optional): Module to fuse descriptors for multiple images in batch.
                Defaults to None.
        """
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.fusion = fusion

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:  # noqa: D102
        img_descriptors = {}
        for key, value in batch.items():
            if key.startswith("images_"):
                img_descriptors[key] = self.head(self.backbone(value))
        if len(img_descriptors) > 1:
            if self.fusion is None:
                raise ValueError("Fusion module is not defined but multiple images are provided")
            descriptor = self.fusion(img_descriptors)
        else:
            if self.fusion is not None:
                raise ValueError("Fusion module is defined but only one image is provided")
            descriptor = list(img_descriptors.values())[0]
        out_dict: Dict[str, Tensor] = {"final_descriptor": descriptor}
        return out_dict


class CloudModel(nn.Module):
    """Meta-model for lidar-based Place Recognition. Combines feature extraction backbone and head modules."""

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
    ) -> None:
        """Meta-model for lidar-based Place Recognition.

        Args:
            backbone (CloudFeatureExtractor): Cloud feature extraction backbone.
            head (CloudHead): Cloud head module.
        """
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:  # noqa: D102
        x = ME.SparseTensor(
            features=batch["pointclouds_lidar_feats"], coordinates=batch["pointclouds_lidar_coords"]
        )
        x = self.backbone(x)
        x = self.head(x)
        out_dict: Dict[str, Tensor] = {"final_descriptor": x}
        return out_dict


class LateFusionModel(nn.Module):
    """Meta-model for multimodal Place Recognition architectures with late fusion."""

    def __init__(
        self,
        image_module: Optional[ImageModel] = None,
        semantic_module: Optional[ImageModel] = None,
        cloud_module: Optional[CloudModel] = None,
        # text_module: Optional[Union[FusionTextModel, MultiTextModule]] = None,  # TODO: re-think
        fusion_module: Optional[nn.Module] = None,
    ) -> None:
        """Meta-model for multimodal Place Recognition architectures with late fusion.

        Args:
            image_module (ImageModule, optional): Image modality branch. Defaults to None.
            semantic_module (ImageModule, optional): Semantic modality branch. Defaults to None.
            cloud_module (CloudModule, optional): Cloud modality branch. Defaults to None.
            fusion_module (FusionModule, optional): Module to fuse different modalities.
                If None, will be set to opr.modules.Concat(). Defaults to None.
        """
        super().__init__()

        self.image_module = image_module
        self.semantic_module = semantic_module
        self.cloud_module = cloud_module
        # self.text_module = text_module
        if fusion_module:
            self.fusion_module = fusion_module
        else:
            self.fusion_module = Concat()

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:  # noqa: D102
        out_dict: Dict[str, Tensor] = {}

        if self.image_module is not None:
            out_dict["image"] = self.image_module(batch)

        if self.semantic_module is not None:
            out_dict["semantic"] = self.semantic_module(batch)

        if self.cloud_module is not None:
            out_dict["cloud"] = self.cloud_module(batch)

        # if self.text_module is not None and isinstance(self.text_module, FusionTextModel):
        #     out_dict["text"] = self.text_module(batch["back_embs"], batch["front_embs"])
        # elif self.text_module is not None and isinstance(self.text_module, MultiTextModule):
        #     out_dict["text"] = self.text_module(batch)

        out_dict["final_descriptor"] = self.fusion_module(out_dict)

        return out_dict
