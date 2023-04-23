"""Multimodal Place Recognition models."""
from pathlib import Path
from typing import Optional, Union

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from .base_models import ComposedModel


def minkloc_multimodal(weights: Optional[Union[str, Path]] = None) -> ComposedModel:
    """Default MinkLoc++ model configuration.

    Paper: https://arxiv.org/abs/2104.05327

    Komorowski, Jacek, Monika Wysoczańska, and Tomasz Trzcinski. "MinkLoc++: lidar
    and monocularimage fusion for place recognition." 2021 International Joint
    Conference on Neural Networks (IJCNN). IEEE, 2021.

    Args:
        weights (Union[str, Path], optional): The path to the weights 'pth' file. Defaults to None.

    Raises:
        ValueError: If given weights file does not exist.

    Returns:
        ComposedModel: MinkLoc++ model with default configuration from the original paper.
    """
    model_config = DictConfig(
        {
            "_target_": "opr.models.base_models.ComposedModel",
            "image_module": {
                "_target_": "opr.models.base_models.ImageModule",
                "backbone": {
                    "_target_": "opr.models.resnet.ResNet18FPNExtractor",
                    "lateral_dim": 128,
                    "fh_num_bottom_up": 4,
                    "fh_num_top_down": 0,
                    "pretrained": True,
                },
                "head": {"_target_": "opr.models.layers.gem.GeM", "p": 3, "eps": 1e-06},
            },
            "cloud_module": {
                "_target_": "opr.models.base_models.CloudModule",
                "backbone": {
                    "_target_": "opr.models.minkloc.MinkResNetFPNExtractor",
                    "out_channels": 128,
                    "num_top_down": 1,
                    "conv0_kernel_size": 5,
                    "block": "ECABasicBlock",
                    "layers": [1, 1, 1],
                    "planes": [32, 64, 64],
                },
                "head": {"_target_": "opr.models.layers.gem.MinkGeM", "p": 3, "eps": 1e-06},
            },
            "fusion_module": {"_target_": "opr.models.fusion.Concat"},
        }
    )
    model = instantiate(model_config)
    if weights is not None:
        if not Path(weights).exists():
            raise ValueError(f"Given weights file does not exist: {weights}")
        ckpt = torch.load(weights)["model_state_dict"]
        model.load_state_dict(ckpt)
    return model
