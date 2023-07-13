"""Utility functions for tests."""
from pathlib import Path
from typing import Union

from omegaconf import DictConfig, ListConfig, OmegaConf


def load_config(config_path: Union[str, Path]) -> Union[DictConfig, ListConfig]:
    """Load config from path.

    Args:
        config_path (Union[str, Path]): Path to config file.

    Returns:
        Union[DictConfig, ListConfig]: Config object.
    """
    config = OmegaConf.load(config_path)
    return config
