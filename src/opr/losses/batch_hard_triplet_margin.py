"""Multimodal triplet margin loss implementation.

Code adopted from repository: https://github.com/jac99/MinkLocMultimodal, MIT License
"""

from typing import Dict, Tuple

from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.reducers import AvgNonZeroReducer
from torch import Tensor, nn

from opr.miners import BatchHardTripletMiner


class BatchHardTripletMarginLoss(nn.Module):
    """Triplet margin loss with batch hard triplet miner.

    Code adopted from repository: https://github.com/jac99/MinkLocMultimodal, MIT License
    """

    def __init__(self, margin: float = 0.2) -> None:
        """Triplet margin loss with batch hard triplet miner.

        Args:
            margin (float): Margin value for TripletMarginLoss. Defaults to 0.2.
        """
        super().__init__()
        self.margin = margin
        distance = LpDistance(normalize_embeddings=False, collect_stats=True)
        reducer_fn = AvgNonZeroReducer(collect_stats=True)
        self.miner_fn = BatchHardTripletMiner(distance=distance)
        self.loss_fn = TripletMarginLoss(
            margin=self.margin, swap=True, distance=distance, reducer=reducer_fn, collect_stats=True
        )

    def forward(  # noqa: D102
        self, embeddings: Tensor, positives_mask: Tensor, negatives_mask: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        hard_triplets = self.miner_fn(embeddings, positives_mask, negatives_mask)
        miner_stats = self.miner_fn.stats
        loss = self.loss_fn(embeddings, indices_tuple=hard_triplets)
        stats = {
            "loss": loss.item(),
            "avg_embedding_norm": self.loss_fn.distance.final_avg_query_norm,
            "num_triplets": len(hard_triplets[0]),
            "num_non_zero_triplets": float(self.loss_fn.reducer.num_past_filter),
        }
        try:
            stats["non_zero_rate"] = stats["num_non_zero_triplets"] / stats["num_triplets"]
        except ZeroDivisionError:
            print("WARNING: encoutered batch with 'num_triplets' == 0.")
            stats["non_zero_rate"] = 1.0
        stats.update(miner_stats)
        return loss, stats
