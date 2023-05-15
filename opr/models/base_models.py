"""Interfaces and meta-models definitions."""
from typing import Dict, Optional, Union, Tuple

import MinkowskiEngine as ME  # noqa: N817
from torch import Tensor, nn
import torch


class ImageFeatureExtractor(nn.Module):
    """Interface class for image feature extractor module."""

    def __init__(self):
        """Interface class for image feature extractor module."""
        super().__init__()

    def forward(self, image: Tensor) -> Tensor:  # noqa: D102
        raise NotImplementedError()
    
class DoubleImageFeatureExtractor(nn.Module):
    """Interface class for image feature extractor module."""

    def __init__(self):
        """Interface class for image feature extractor module."""
        super().__init__()

    def forward(self, image_left: Tensor, image_right: Tensor) -> Tuple[Tensor, Tensor]:  # noqa: D102
        raise NotImplementedError()


class ImageHead(nn.Module):
    """Interface class for image head module."""

    def __init__(self):
        """Interface class for image head module."""
        super().__init__()

    def forward(self, feature_map: Tensor) -> Tensor:  # noqa: D102
        raise NotImplementedError()

class DoubleImageHead(nn.Module):
    """Interface class for double images head module."""

    def __init__(self):
        """Interface class for image head module."""
        super().__init__()

    def forward(self, feature_map_left: Tensor, feature_map_right: Tensor) -> Tuple[Tensor, Tensor]:  # noqa: D102
        raise NotImplementedError()


class ImageModule(nn.Module):
    """Meta-module for image branch. Combines feature extraction backbone and head modules."""

    def __init__(
        self,
        backbone: ImageFeatureExtractor,
        head: ImageHead,
    ):
        """Meta-module for image branch.

        Args:
            backbone (ImageFeatureExtractor): Image feature extraction backbone.
            head (ImageHead): Image head module.
        """
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        x = self.backbone(x)
        x = self.head(x)
        return x


class DoubleImageModule(nn.Module):
    """Meta-module for double images branch. Combines feature extraction backbone and head modules."""

    def __init__(
        self,
        backbone: DoubleImageFeatureExtractor,
        head: DoubleImageHead,
    ):
        """Meta-module for image branch.

        Args:
            backbone (ImageFeatureExtractor): Image feature extraction backbone.
            head (ImageHead): Image head module.
        """
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x_left: Tensor, x_right: Tensor) -> Tuple[Tensor, Tensor]:  # noqa: D102
        x_left, x_right = self.backbone(x_left, x_right)
        x_left, x_right = self.head(x_left, x_right)
        return x_left, x_right

class CloudFeatureExtractor(nn.Module):
    """Interface class for cloud feature extractor module."""

    def __init__(self):
        """Interface class for cloud feature extractor module."""
        super().__init__()

    def forward(self, cloud: ME.SparseTensor) -> ME.SparseTensor:  # noqa: D102
        raise NotImplementedError()


class CloudHead(nn.Module):
    """Interface class for cloud head module."""

    def __init__(self):
        """Interface class for cloud head module."""
        super().__init__()

    def forward(self, feature_map: ME.SparseTensor) -> Tensor:  # noqa: D102
        raise NotImplementedError()


class CloudModule(nn.Module):
    """Meta-module for cloud branch. Combines feature extraction backbone and head modules."""

    def __init__(
        self,
        backbone: CloudFeatureExtractor,
        head: CloudHead,
    ):
        """Meta-module for cloud branch.

        Args:
            backbone (CloudFeatureExtractor): Cloud feature extraction backbone.
            head (CloudHead): Cloud head module.

        Raises:
            ValueError: If incompatible cloud backbone and head are given.
        """
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: ME.SparseTensor) -> Tensor:  # noqa: D102
        x = self.backbone(x)
        x = self.head(x)
        return x


class FusionModule(nn.Module):
    """Interface class for fusion module."""

    def __init__(self):
        """Interface class for fusion module."""
        super().__init__()

    def forward(self, data: Dict[str, Union[Tensor, ME.SparseTensor]]) -> Tensor:  # noqa: D102
        raise NotImplementedError()


class MultiImageModule(nn.Module):
    """Module to work with multiple images with late fusion."""

    def __init__(self, image_module: ImageModule, fusion_module: FusionModule) -> None:
        """Module to work with multiple images with late fusion.

        Args:
            image_module (ImageModule): Module to process each image.
            fusion_module (FusionModule): Module to fuse descriptors of each image.
        """
        super().__init__()
        self.image_module = image_module
        self.fusion_module = fusion_module

    def forward(self, data: Dict[str, Tensor]) -> Tensor:  # noqa: D102
        x_dict = {}
        for key in data:
            if key.startswith("images_"):
                x_dict[key] = self.image_module(data[key])
        x = self.fusion_module(x_dict)
        return x
    
    
#####################################
#### Stupid Text Model for ITLP #####
#####################################
class BasicTextNN(nn.Module):
    def __init__(self, pca_size=100, hidden_size=100):
        super().__init__()
        self.fc1 = nn.Linear(pca_size, hidden_size)
        self.relu = nn.ReLU() 
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, pca_vector):
        x = self.fc1(pca_vector)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class FusionTextModel(nn.Module):
    def __init__(self, pca_size=100, hidden_size=100):
        super().__init__()
        self.back_nn = BasicTextNN(pca_size=pca_size, hidden_size=hidden_size)
        self.front_nn = BasicTextNN(pca_size=pca_size, hidden_size=hidden_size)
        self.fuse_fc1 = nn.Linear(hidden_size*2, hidden_size)
        
    def forward(self, back_vector, front_vector):
        back_feat = self.back_nn(back_vector)
        front_feat = self.front_nn(front_vector)
        fusion_feat = torch.cat((back_feat, front_feat), dim=1)
        embed = self.fuse_fc1(fusion_feat)
        return embed
    
#####################################
#### One text model for all cam #####
#####################################
class TextModule(nn.Module):
    def __init__(self, text_emb_size=100, hidden_size=100):
        super().__init__()
        self.fc1 = nn.Linear(text_emb_size, hidden_size)
        self.relu = nn.ReLU() 
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, embedding):
        x = self.fc1(embedding)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
class EmptyTextModule(nn.Module):
    def __init__(self, text_emb_size=None, hidden_size=None):
        super().__init__()

    def forward(self, embedding):
        embedding.requires_grad = True
        return embedding


class MultiTextModule(nn.Module):
    """Module to work with multiple text embeddings with late fusion."""

    def __init__(self, text_module: TextModule, fusion_module: FusionModule) -> None:
        """Module to work with multiple text embeddings with late fusion.

        Args:
            text_module (TextModule): Module to process each text embedding.
            fusion_module (FusionModule): Module to fuse descriptors of each text embedding.
        """
        super().__init__()
        self.text_module = text_module
        self.fusion_module = fusion_module

    def forward(self, data: Dict[str, Tensor]) -> Tensor:  # noqa: D102
        x_dict = {}
        for key in data:
            if key.startswith("text_emb_"):
                x_dict[key] = self.text_module(data[key])
        x = self.fusion_module(x_dict)
        return x    


class ComposedModel(nn.Module):
    """Composition model for multimodal architectures."""

    def __init__(
        self,
        image_module: Optional[Union[ImageModule, MultiImageModule]] = None,
        semantic_module: Optional[ImageModule] = None,
        chonky_module: Optional[DoubleImageModule]= None,
        cloud_module: Optional[CloudModule] = None,
        text_module: Optional[Union[FusionTextModel, MultiTextModule]] = None,
        fusion_module: Optional[FusionModule] = None,
    ) -> None:
        """Composition model for multimodal architectures.

        Args:
            image_module (ImageModule, optional): Image modality branch. Defaults to None.
            semantic_module (ImageModule, optional): Semantic modality branch. Defaults to None.
            chonky_module (DoubleImageModule, optional): Image+Semantic modalities branch. Defaults to None.
            cloud_module (CloudModule, optional): Cloud modality branch. Defaults to None.
            fusion_module (FusionModule, optional): Module to fuse different modalities. Defaults to None.
        """
        super().__init__()

        self.image_module = image_module
        self.semantic_module = semantic_module
        self.chonky_module = chonky_module
        self.cloud_module = cloud_module
        self.text_module = text_module
        self.fusion_module = fusion_module

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Optional[Tensor]]:  # noqa: D102
        out_dict: Dict[str, Optional[Tensor]] = {
            "image": None,
            "cloud": None,
            # "fusion": None,
        }

        if self.chonky_module is None: #! It's a bit tricky but idk how to do it better now
            if self.image_module is not None and isinstance(self.image_module, ImageModule):
                out_dict["image"] = self.image_module(batch["images"])
            elif self.image_module is not None and isinstance(self.image_module, MultiImageModule):
                out_dict["image"] = self.image_module(batch)

            if self.semantic_module is not None and isinstance(self.semantic_module, ImageModule):
                out_dict["semantic"] = self.semantic_module(batch["semantics"])
        
        if self.chonky_module is not None and isinstance(self.chonky_module, DoubleImageModule):
            out_dict["image"], out_dict["semantic"] = self.chonky_module(batch['images'], batch['semantics'])

        if self.cloud_module is not None:
            cloud = ME.SparseTensor(features=batch["features"], coordinates=batch["coordinates"])
            out_dict["cloud"] = self.cloud_module(cloud)
            
        if self.text_module is not None and isinstance(self.text_module, FusionTextModel):
            out_dict["text"] = self.text_module(batch["back_embs"], batch["front_embs"])
        elif self.text_module is not None and isinstance(self.text_module, MultiTextModule):
            out_dict["text"] = self.text_module(batch)

        if self.fusion_module is not None:
            out_dict["fusion"] = self.fusion_module(out_dict)

        return out_dict
