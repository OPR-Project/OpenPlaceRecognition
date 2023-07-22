"""Batch hard triplet miner implementation.

Code adopted from repository: https://github.com/jac99/MinkLocMultimodal, MIT License
"""

from typing import Dict, Tuple

import torch
from pytorch_metric_learning.distances import BaseDistance
from torch import Tensor, nn


class BatchHardTripletMiner(nn.Module):
    """Batch hard triplet miner.

    Original idea is taken from repository: https://github.com/jac99/MinkLocMultimodal, MIT License
    """

    _stats: Dict[str, float]

    def __init__(
        self,
        distance: BaseDistance,
    ) -> None:
        """Batch hard triplet miner.

        Args:
            distance (BaseDistance): Distance function to use.
        """
        super().__init__()
        self.distance = distance
        self._init_stats()

    def forward(
        self, embeddings: Tensor, positives_mask: Tensor, negatives_mask: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Mine hard triplets from given batch of embeddings. For each element in batch triplet will be mined.

        Args:
            embeddings (Tensor): Dictionary with model output embeddings.
            positives_mask (Tensor): Binary mask of positive elements in batch.
            negatives_mask (Tensor): Binary mask of negative elements in batch.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Hard triplets tuple (a, p, n).
        """
        d_embeddings = embeddings.detach()
        with torch.no_grad():
            hard_triplets, self._stats = self._mine(d_embeddings, positives_mask, negatives_mask)
        return hard_triplets

    @property
    def stats(self) -> Dict[str, float]:
        """Return statistics of last forward pass."""
        return self._stats

    def _init_stats(self) -> None:
        self._stats = {}
        keys = (
            "max_pos_pair_dist",
            "max_neg_pair_dist",
            "mean_pos_pair_dist",
            "mean_neg_pair_dist",
            "min_pos_pair_dist",
            "min_neg_pair_dist",
        )
        for key in keys:
            self._stats[key] = 0.0

    def _mine(
        self, embeddings: Tensor, positives_mask: Tensor, negatives_mask: Tensor
    ) -> Tuple[Tuple[Tensor, Tensor, Tensor], Dict[str, float]]:
        """Mine hard triplets from given batch of embeddings."""
        dist_mat = self.distance(embeddings)
        (hardest_positive_dist, hardest_positive_indices), a1p_keep = self._get_max_per_row(
            dist_mat, positives_mask
        )
        (hardest_negative_dist, hardest_negative_indices), a2n_keep = self._get_min_per_row(
            dist_mat, negatives_mask
        )
        a_keep_idx = torch.where(a1p_keep & a2n_keep)
        a = torch.arange(dist_mat.size(0)).to(hardest_positive_indices.device)[a_keep_idx]
        p = hardest_positive_indices[a_keep_idx]
        n = hardest_negative_indices[a_keep_idx]
        stats = {}
        stats["max_pos_pair_dist"] = torch.max(hardest_positive_dist).item()
        stats["max_neg_pair_dist"] = torch.max(hardest_negative_dist).item()
        stats["mean_pos_pair_dist"] = torch.mean(hardest_positive_dist).item()
        stats["mean_neg_pair_dist"] = torch.mean(hardest_negative_dist).item()
        stats["min_pos_pair_dist"] = torch.min(hardest_positive_dist).item()
        stats["min_neg_pair_dist"] = torch.min(hardest_negative_dist).item()
        return (a, p, n), stats

    def _get_max_per_row(self, mat: Tensor, mask: Tensor) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
        """Max per row with mask."""
        non_zero_rows = torch.any(mask, dim=1)
        mat_masked = mat.clone()
        mat_masked[~mask] = 0
        return torch.max(mat_masked, dim=1), non_zero_rows

    def _get_min_per_row(self, mat: Tensor, mask: Tensor) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
        """Min per row with mask."""
        non_inf_rows = torch.any(mask, dim=1)
        mat_masked = mat.clone()
        mat_masked[~mask] = float("inf")
        return torch.min(mat_masked, dim=1), non_inf_rows
