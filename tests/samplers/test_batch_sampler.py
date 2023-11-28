"""Test cases for opr.samplers.batch_sampler module."""
import pytest
from hydra.utils import instantiate

from opr.samplers.batch_sampler import BatchSampler
from tests.utils import load_config


@pytest.mark.e2e
def test_batch_sampler_instantiate_with_real_data() -> None:
    """Should instantiate BatchSampler object."""
    dataset_config = load_config("configs/dataset/oxford.yaml")
    dataset_config.dataset_root = "/home/docker_opr/Datasets/pnvlad_oxford_robotcar"
    dataset = instantiate(dataset_config)
    sampler_config = load_config("configs/sampler/batch_sampler.yaml")
    sampler_config.dataset = dataset_config
    sampler = instantiate(sampler_config, dataset=dataset)
    assert isinstance(sampler, BatchSampler)
