"""Multimodal contrastive loss implementation.

Code adopted from repository: https://github.com/jac99/MinkLocMultimodal, MIT License
"""

from typing import Dict, Tuple

from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.losses import ContrastiveLoss
from pytorch_metric_learning.reducers import AvgNonZeroReducer
from torch import Tensor, nn

from opr.miners import BatchHardTripletMiner


class BatchHardContrastiveLoss(nn.Module):
    """Contrastive loss with batch hard triplet miner.

    Code adopted from repository: https://github.com/jac99/MinkLocMultimodal, MIT License
    """

    def __init__(self, pos_margin: float = 0.2, neg_margin: float = 0.2):
        """
        Initializes the BatchHardContrastiveLoss module.

        Args:
            pos_margin (float): Margin value for positive pairs in ContrastiveLoss. Defaults to 0.2.
            neg_margin (float): Margin value for negative pairs in ContrastiveLoss. Defaults to 0.2.
        """
        super().__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.distance = LpDistance(normalize_embeddings=False, collect_stats=True)
        self.miner_fn = BatchHardTripletMiner(distance=self.distance)
        reducer_fn = AvgNonZeroReducer(collect_stats=True)
        self.loss_fn = ContrastiveLoss(
            pos_margin=self.pos_margin,
            neg_margin=self.neg_margin,
            distance=self.distance,
            reducer=reducer_fn,
            collect_stats=True,
        )

    def forward(  # noqa: D102
        self, embeddings: Tensor, positives_mask: Tensor, negatives_mask: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        hard_triplets = self.miner_fn(embeddings, positives_mask, negatives_mask)
        miner_stats = self.miner_fn.stats

        # dummy_labels = torch.arange(embeddings.shape[0]).to(embeddings.device)
        # loss = self.loss_fn(embeddings, dummy_labels, hard_triplets)

        loss = self.loss_fn(embeddings, indices_tuple=hard_triplets)

        # print(self.loss_fn.reducer.reducers["pos_loss"])
        # print(type(self.loss_fn.reducer.reducers["pos_loss"]))
        # print(self.loss_fn.reducer.__dict__)
        # print(self.loss_fn.reducer.reducers["pos_loss"].__dict__)

        # print(self.loss_fn.reducer.reducers)

        # print(self.loss_fn.reducer.reducers["pos_loss"])
        # print("=========================")
        # print(self.loss_fn.reducer.reducers["pos_loss"].__dict__)
        # print("=========================")
        # print(self.loss_fn.reducer.reducers["pos_loss"].pos_loss.item())

        # print(self.loss_fn.reducer.reducers["pos_loss"].num_past_filter)

        stats = {
            "loss": loss.item(),
            "avg_embedding_norm": self.loss_fn.distance.final_avg_query_norm,
            "pos_pairs_above_threshold": self.loss_fn.reducer.reducers["pos_loss"].num_past_filter,
            "neg_pairs_above_threshold": self.loss_fn.reducer.reducers["neg_loss"].num_past_filter,
            "num_pairs": 2 * len(hard_triplets[0]),
        }

        try:
            stats["non_zero_rate"] = (
                stats["pos_pairs_above_threshold"] + stats["neg_pairs_above_threshold"]
            ) / stats["num_pairs"]
        except ZeroDivisionError:
            print("WARNING: encoutered batch with 'num_pairs' == 0.")
            stats["non_zero_rate"] = 1.0

        stats.update(miner_stats)

        return loss, stats
