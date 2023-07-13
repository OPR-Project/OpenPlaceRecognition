"""Test cases for opr.models.place_recognition.minkloc3d module."""
import pytest
from hydra.utils import instantiate

from opr.models.place_recognition.minkloc3d import MinkLoc3D
from tests.utils import load_config


@pytest.mark.e2e
def test_minkloc3d_instantiate() -> None:
    """Should instantiate MinkLoc3D object."""
    config = load_config("configs/model/place_recognition/minkloc3d.yaml")
    model = instantiate(config)
    assert isinstance(model, MinkLoc3D)
