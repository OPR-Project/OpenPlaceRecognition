"""Batch sampler from MinkLoc method.

Code adopted from repository: https://github.com/jac99/MinkLocMultimodal, MIT License
"""
from typing import List, Optional

import numpy as np
from numpy.random import default_rng
from torch.utils.data import Sampler

from opr.datasets.base import BaseDataset


class BatchSampler(Sampler):
    """Sampler returning list of indices to form a mini-batch.

    Samples elements in groups consisting of k=2 similar elements (positives)
    Batch has the following structure: item1_1, ..., item1_k, item2_1, ... item2_k, itemn_1, ..., itemn_k
    """

    is_batches_generated: bool = False

    def __init__(
        self,
        dataset: BaseDataset,
        batch_size: int,
        batch_size_limit: Optional[int] = None,
        batch_expansion_rate: Optional[float] = None,
        max_batches: Optional[int] = None,
        positives_per_group: int = 2,
        seed: Optional[int] = None,
    ) -> None:
        """Sampler returning list of indices to form a mini-batch.

        Note:
            The dynamic batch size option is implemented. You can read more about it
            in the MinkLoc paper: https://arxiv.org/abs/2011.04530

        Args:
            dataset (BaseDataset): Dataset from which to sample.
            batch_size (int): Initial batch size.
            batch_size_limit (int, optional): Maximum batch size if dynamic batch sizing
                is enabled (see MinkLoc paper for details). Defaults to None.
            batch_expansion_rate (float, optional): Batch expansion rate if dynamic batch sizing
                is enabled (see MinkLoc paper for details). Defaults to None.
            max_batches (int, optional): Maximum number of batches to generate in epoch. Defaults to None.
            positives_per_group (int): Number of positive elements to sample in group. Defaults to 2.
            seed (int, optional): Random seed. Defaults to None.
        """
        if batch_expansion_rate is not None:
            assert batch_size_limit is not None
            assert batch_expansion_rate > 1.0, "batch_expansion_rate must be greater than 1"
            assert batch_size <= batch_size_limit, "batch_size_limit must be greater or equal to batch_size"

        self.batch_size = batch_size
        self.batch_size_limit = batch_size_limit
        self.batch_expansion_rate = batch_expansion_rate
        self.max_batches = max_batches
        self.dataset = dataset

        assert positives_per_group >= 2, "Number of positive examples per group must be >= 2"
        self.positives_per_group = positives_per_group

        if self.batch_size < 2 * self.positives_per_group:
            self.batch_size = 2 * self.positives_per_group
            print("WARNING: Batch too small. Batch size increased to {}.".format(self.batch_size))
        elif self.batch_size % self.positives_per_group != 0:
            self.batch_size = self.batch_size - (self.batch_size % self.positives_per_group)
            print(
                "WARNING: Batch size must be divisible by number of positives per group.",
                f"Batch size decreased to {self.batch_size}",
                f"(positives_per_group={self.positives_per_group}).",
            )

        if self.batch_size_limit is not None and (self.batch_size_limit % self.positives_per_group != 0):
            self.batch_size_limit = self.batch_size_limit - (self.batch_size_limit % self.positives_per_group)
            print(
                "WARNING: Batch size limit must be divisible by number of positives per group.",
                f"Batch size limit decreased to {self.batch_size_limit}",
                f"(positives_per_group={self.positives_per_group}).",
            )
            assert self.batch_size <= self.batch_size_limit

        self.batch_idx: List[List[int]] = []  # Index of elements in each batch (re-generated every epoch)
        self.elems_ndx = np.arange(len(self.dataset))  # array of indexes

        self.rng = default_rng(seed=seed)
        self.generate_batches()  # generate initial batches list (to make __len__ work properly)

    def __iter__(self):  # noqa: D105
        if not self.is_batches_generated:
            self.generate_batches()  # re-generate batches on every epoch
        for batch in self.batch_idx:
            yield batch
        self.is_batches_generated = False

    def __len__(self) -> int:  # noqa: D105
        return len(self.batch_idx)

    def expand_batch(self) -> None:
        """Batch expansion method. See MinkLoc paper for details about dynamic batch sizing."""
        if self.batch_expansion_rate is None or self.batch_size_limit is None:
            print("WARNING: dynamic batch sizing is disabled but 'expand_batch' method was called.")
            return

        if self.batch_size >= self.batch_size_limit:
            return

        old_batch_size = self.batch_size
        self.batch_size = int(self.batch_size * self.batch_expansion_rate)
        # ensure that it is still divisible by number of positives per group:
        self.batch_size = self.batch_size - (self.batch_size % self.positives_per_group)
        # but if batch_expansion_rate is small - we may decrease it back to previous batch size:
        if self.batch_size == old_batch_size:
            self.batch_size += self.positives_per_group  # smallest possible step
        # then check if it is smaller than the limit
        self.batch_size = min(self.batch_size, self.batch_size_limit)
        print(f"=> Batch size increased from: {old_batch_size} to {self.batch_size}")
        self.generate_batches()

    def generate_batches(self) -> None:
        """Generate training/evaluation batches."""
        # batch_idx holds indexes of elements in each batch as a list of lists
        self.batch_idx = []

        unused_elements_ndx = np.copy(self.elems_ndx)
        current_batch: List[int] = []

        while True:
            if len(current_batch) >= self.batch_size or len(unused_elements_ndx) == 0:
                # Flush out batch, when it has a desired size, or a smaller batch, when there's no more
                # elements to process
                if len(current_batch) >= 2 * self.positives_per_group:
                    # Ensure there're at least two groups of similar elements, otherwise, it would not be possible
                    # to find negative examples in the batch
                    assert (
                        len(current_batch) % self.positives_per_group == 0
                    ), "Incorrect bach size: {}".format(len(current_batch))
                    self.batch_idx.append(current_batch)
                    current_batch = []
                    if (self.max_batches is not None) and (len(self.batch_idx) >= self.max_batches):
                        break
                if len(unused_elements_ndx) == 0:
                    break

            # Add k similar elements to the batch
            selected_element = self.rng.choice(unused_elements_ndx)
            unused_elements_ndx = np.delete(
                unused_elements_ndx, np.argwhere(unused_elements_ndx == selected_element)
            )

            positives = self.dataset.get_positives(selected_element)
            if len(positives) < (self.positives_per_group - 1):
                # we need at least k-1 positive examples
                continue

            unused_positives = [e for e in positives if e in unused_elements_ndx]
            used_positives = [e for e in positives if e not in unused_elements_ndx]
            # If there're unused elements similar to selected_element, sample from them
            # otherwise sample from all similar elements
            current_batch += [selected_element]
            for _ in range(self.positives_per_group - 1):
                if len(unused_positives) > 0:
                    pos_i = self.rng.choice(len(unused_positives))
                    another_positive = unused_positives.pop(pos_i)
                    unused_elements_ndx = np.delete(
                        unused_elements_ndx, np.argwhere(unused_elements_ndx == another_positive)
                    )
                else:
                    pos_i = self.rng.choice(len(used_positives))
                    another_positive = used_positives.pop(pos_i)
                current_batch += [another_positive]

        for batch in self.batch_idx:
            assert len(batch) % self.positives_per_group == 0, "Incorrect bach size: {}".format(len(batch))
        self.is_batches_generated = True
