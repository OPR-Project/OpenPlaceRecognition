"""Batch sampler from MinkLoc method.

Code adopted from repository: https://github.com/jac99/MinkLocMultimodal, MIT License
"""
from typing import Iterator, List, Optional

import numpy as np
import torch
from numpy.random import default_rng
from torch.utils.data import Sampler

from opr.datasets.base import BasePlaceRecognitionDataset


class BatchSampler(Sampler):
    """Sampler returning list of indices to form a mini-batch.

    Samples elements in groups consisting of k=2 similar elements (positives)
    Batch has the following structure: item1_1, ..., item1_k, item2_1, ... item2_k, itemn_1, ..., itemn_k
    """

    # TODO: refactor this class to be more readable
    # TODO: separate private members from public members
    is_batches_generated: bool = False

    def __init__(
        self,
        dataset: BasePlaceRecognitionDataset,
        batch_size: int,
        batch_size_limit: Optional[int] = None,
        batch_expansion_rate: Optional[float] = None,
        max_batches: Optional[int] = None,
        positives_per_group: int = 2,
        seed: Optional[int] = None,
        drop_last: bool = True,
    ) -> None:
        """Sampler returning list of indices to form a mini-batch.

        Note:
            The dynamic batch size option is implemented. You can read more about it
                in the MinkLoc paper: https://arxiv.org/abs/2011.04530

        Args:
            dataset (BasePlaceRecognitionDataset): Dataset from which to sample.
            batch_size (int): Initial batch size.
            batch_size_limit (int, optional): Maximum batch size if dynamic batch sizing
                is enabled (see MinkLoc paper for details). Defaults to None.
            batch_expansion_rate (float, optional): Batch expansion rate if dynamic batch sizing
                is enabled (see MinkLoc paper for details). Defaults to None.
            max_batches (int, optional): Maximum number of batches to generate in epoch. If None, then
                no limit will be applied. Defaults to None.
            positives_per_group (int): Number of positive elements to sample in group. Defaults to 2.
            seed (int, optional): Random seed. Defaults to None.
            drop_last (bool): If True, the sampler will drop the last batch if its size would be less
                than batch_size. Defaults to True.

        Raises:
            ValueError: If batch_size_limit is not specified when batch_expansion_rate is specified.
            ValueError: If batch_expansion_rate is less or equal to 1.0.
            ValueError: If batch_size_limit is less or equal to batch_size.
            ValueError: If positives_per_group is less than 2.
        """
        if batch_expansion_rate is not None:
            if batch_size_limit is None:
                raise ValueError("batch_size_limit must be specified if batch_expansion_rate is specified")
            if batch_expansion_rate <= 1.0:
                raise ValueError("batch_expansion_rate must be greater than 1.0")
            if batch_size_limit <= batch_size:
                raise ValueError("batch_size_limit must be greater than batch_size")

        self.batch_size = batch_size
        self.batch_size_limit = batch_size_limit
        self.batch_expansion_rate = batch_expansion_rate
        self.max_batches = max_batches
        self.drop_last = drop_last
        self.dataset = dataset

        if positives_per_group < 2:
            raise ValueError("positives_per_group must be greater or equal to 2")
        self.positives_per_group = positives_per_group

        if self.batch_size < 2 * self.positives_per_group:
            self.batch_size = 2 * self.positives_per_group
            print("WARNING: Batch too small. Batch size increased to {}.".format(self.batch_size))
            # TODO: change print to logger
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
            if self.batch_size > self.batch_size_limit:
                raise ValueError("batch_size must be less or equal to batch_size_limit")

        self.batch_idx: List[List[int]] = []  # Index of elements in each batch (re-generated every epoch)
        self.elems_ndx = np.arange(len(self.dataset))  # array of indexes

        self.rng = default_rng(seed=seed)
        self.generate_batches()  # generate initial batches list (to make __len__ work properly)

    def __iter__(self) -> Iterator[List[int]]:  # noqa: D105
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

    def generate_batches(self) -> None:  # noqa: C901 # TODO: refactor to reduce complexity
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
                    if len(current_batch) % self.positives_per_group != 0:
                        raise ValueError("Batch size must be divisible by number of positives per group.")
                    if self.drop_last and len(current_batch) < self.batch_size:
                        # Drop last batch if it is smaller than batch_size
                        break
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

            positives = self.dataset.positives_index[selected_element].numpy()
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
            if len(batch) % self.positives_per_group != 0:
                raise ValueError(f"Incorrect bach size: {len(batch)}")
        self.is_batches_generated = True


class DistributedBatchSamplerWrapper(Sampler):
    """Wrapper for BatchSampler that supports distributed batch sampling."""

    def __init__(
        self, sampler: BatchSampler, num_replicas: Optional[int] = None, rank: Optional[int] = None
    ) -> None:
        """Wrapper for BatchSampler that supports distributed batch sampling.

        Args:
            sampler (BatchSampler): BatchSampler instance to wrap.
            num_replicas (int, optional): Number of processes participating in distributed training.
                If None, then torch.distributed.get_world_size() will be used. Defaults to None.
            rank (int, optional): Process rank. If None, then torch.distributed.get_rank() will be used.
                Defaults to None.

        Raises:
            ValueError: If sampler has drop_last=False.
            RuntimeError: If distributed package is not available.
            ValueError: If rank is out of range [0, num_replicas-1].
            ValueError: If batch size is not divisible by the number of replicas.
        """
        self.sampler = sampler
        if not self.sampler.drop_last:
            raise ValueError(
                "DistributedBatchSamplerWrapper currently requires sampler to have drop_last=True"
            )
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")
        self.num_replicas = num_replicas
        self.rank = rank
        self.global_batch_size = self.sampler.batch_size
        if self.global_batch_size % self.num_replicas != 0:
            raise ValueError("Batch size should be divisible by the number of replicas")
        self.local_batch_size = self.global_batch_size // self.num_replicas
        self.start_end_indices = self.local_batch_size * self.rank, self.local_batch_size * (self.rank + 1)

    def __iter__(self) -> Iterator[List[int]]:  # noqa: D105
        start_idx, end_idx = self.start_end_indices
        if not self.sampler.is_batches_generated:
            self.sampler.generate_batches()  # re-generate batches on every epoch
        for batch in self.sampler.batch_idx:
            yield batch[start_idx:end_idx]
        self.is_batches_generated = False

    def __len__(self) -> int:  # noqa: D105
        return len(self.sampler)
