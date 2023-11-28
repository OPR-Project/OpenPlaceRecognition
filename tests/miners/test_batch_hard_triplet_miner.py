"""Test cases for opr.miners.batch_hard_triplet_miner module."""
import torch
from hydra.utils import instantiate
from pytorch_metric_learning.distances import LpDistance

from opr.miners.batch_hard_triplet_miner import BatchHardTripletMiner
from tests.utils import load_config


def test_batch_hard_triplet_miner_instantiate_from_config() -> None:
    """Should instantiate BatchHardTripletMiner object."""
    config = load_config("configs/miner/batch_hard_triplet_miner.yaml")
    miner = instantiate(config)
    assert isinstance(miner, BatchHardTripletMiner)


def test_batch_hard_triplet_miner_stats_have_required_keys() -> None:
    """Should have required keys in stats."""
    miner = BatchHardTripletMiner(distance=LpDistance())
    assert set(miner.stats.keys()) == {
        "max_pos_pair_dist",
        "max_neg_pair_dist",
        "mean_pos_pair_dist",
        "mean_neg_pair_dist",
        "min_pos_pair_dist",
        "min_neg_pair_dist",
    }


def test_batch_hard_triplet_miner_stats_init_with_zero_values() -> None:
    """Should have zero values in stats."""
    miner = BatchHardTripletMiner(distance=LpDistance())
    assert set(miner.stats.values()) == {0.0}


def test_batch_hard_triplet_miner_saves_statistics() -> None:
    """Should update statistics after forward pass."""
    miner = BatchHardTripletMiner(distance=LpDistance())
    torch.manual_seed(42)

    embeddings = torch.randn(4, 512)
    positives_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]).bool()
    negatives_mask = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]]).bool()

    _, _, _ = miner(embeddings, positives_mask, negatives_mask)

    for _, value in miner.stats.items():
        assert value > 0.0
