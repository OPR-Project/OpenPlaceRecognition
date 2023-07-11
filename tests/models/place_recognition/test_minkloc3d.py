"""Test cases for opr.models.place_recognition.minkloc3d module."""
from pathlib import Path
from typing import Union

from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from opr.models.place_recognition.minkloc3d import MinkLoc3D


def load_config(config_path: Union[str, Path]) -> Union[DictConfig, ListConfig]:
    """Load config from path.

    Args:
        config_path (Union[str, Path]): Path to config file.

    Returns:
        Union[DictConfig, ListConfig]: Config object.
    """
    config = OmegaConf.load(config_path)
    return config


def test_minkloc3d_instantiate() -> None:
    """Should instantiate MinkLoc3D object."""
    config = load_config("configs/model/place_recognition/minkloc3d.yaml")
    model = instantiate(config)
    assert isinstance(model, MinkLoc3D)
