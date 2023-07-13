"""Test cases for opr.datasets.oxford module."""
from pathlib import Path

import pytest
from hydra.utils import instantiate

from opr.datasets.oxford import OxfordDataset
from tests.utils import load_config


@pytest.mark.e2e
def test_oxford_dataset_instantiate_with_real_data() -> None:
    """Should instantiate OxfordDataset object."""
    config = load_config("configs/dataset/oxford.yaml")
    config.dataset_root = "/home/docker_opr/Datasets/pnvlad_oxford_robotcar"
    assert Path(config.dataset_root).exists()
    dataset = instantiate(config)
    assert isinstance(dataset, OxfordDataset)
