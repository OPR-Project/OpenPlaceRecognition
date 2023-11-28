"""Test cases for opr.datasets.nclt module."""
# TODO: it is not DRY at all -> copy-pasted from test_oxford.py
from pathlib import Path

import pytest
from hydra.utils import instantiate
from torch.utils.data import DataLoader

from opr.datasets.nclt import NCLTDataset
from tests.utils import load_config


@pytest.mark.e2e
def test_oxford_dataset_instantiate_with_real_data() -> None:
    """Should instantiate OxfordDataset object."""
    config = load_config("configs/dataset/nclt.yaml")
    config.dataset_root = "/home/docker_opr/Datasets/NCLT_preprocessed_v2"
    assert Path(config.dataset_root).exists()
    dataset = instantiate(config)
    assert isinstance(dataset, NCLTDataset)


@pytest.mark.e2e
def test_oxford_dataset_collate_fn_with_real_data() -> None:
    """Should return correct batch, positives and negatives masks."""
    config = load_config("configs/dataset/nclt.yaml")
    config.dataset_root = "/home/docker_opr/Datasets/NCLT_preprocessed_v2"
    dataset = instantiate(config)
    dataloader = DataLoader(
        dataset=dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=dataset.collate_fn
    )
    batch, positives_mask, negatives_mask = next(iter(dataloader))
    assert list(batch.keys()) == ["utms", "pointclouds_lidar_coords", "pointclouds_lidar_feats"]
    assert positives_mask.shape == (2, 2)
    assert negatives_mask.shape == (2, 2)
