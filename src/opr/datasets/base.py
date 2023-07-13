"""Base dataset implementation."""
from pathlib import Path
from typing import Dict, List, Literal, Tuple, Union

import numpy as np
import pandas as pd
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import Dataset


class BasePlaceRecognitionDataset(Dataset):
    """Base class for track-based Place Recognition dataset."""

    dataset_root: Path
    subset: Literal["train", "val", "test"]
    dataset_df: DataFrame
    data_to_load: Tuple[str, ...]

    def __init__(
        self,
        dataset_root: Union[str, Path],
        subset: Literal["train", "val", "test"],
        data_to_load: Union[str, Tuple[str, ...]],
        positive_threshold: float = 10.0,
        negative_threshold: float = 50.0,
    ) -> None:
        """Base class for track-based Place Recognition dataset.

        Args:
            dataset_root (Union[str, Path]): The path to the root directory of the dataset.
            subset (Literal["train", "val", "test"]): The subset of the dataset to load.
            data_to_load (Union[str, Tuple[str, ...]]): The list of data sources to load.
            positive_threshold (float): The maximum distance between two elements
                for them to be considered positive. Defaults to 10.0.
            negative_threshold (float): The maximum distance between two elements
                for them to be considered non-negative. Defaults to 50.0.

        Raises:
            FileNotFoundError: If the dataset_root directory does not exist.
            ValueError: If an invalid subset is given.
            FileNotFoundError: If the csv file for the given subset does not exist.
            ValueError: If positive_threshold or negative_threshold is a negative number.
        """
        self.dataset_root = Path(dataset_root)
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Given dataset_root={self.dataset_root!r} doesn't exist")

        valid_subsets = ("train", "val", "test")
        if subset not in valid_subsets:
            raise ValueError(f"Invalid subset argument: {subset!r} not in {valid_subsets!r}")
        self.subset = subset

        subset_csv_path = self.dataset_root / f"{subset}.csv"
        if not subset_csv_path.exists():
            raise FileNotFoundError(
                f"There is no {subset}.csv file in given dataset_root={self.dataset_root!r}."
                "Consider checking documentation on how to preprocess the dataset."
            )
        self.dataset_df = pd.read_csv(subset_csv_path, index_col=0)

        if isinstance(data_to_load, str):
            data_to_load = tuple([data_to_load])
        else:
            data_to_load = tuple(data_to_load)
        self.data_to_load = data_to_load

        if positive_threshold < 0.0:
            raise ValueError(f"positive_threshold must be non-negative, but {positive_threshold!r} given.")
        if negative_threshold < 0.0:
            raise ValueError(f"negative_threshold must be non-negative, but {negative_threshold!r} given.")

        self._positives_index = self._build_index_by_distance_threshold(positive_threshold)
        self._nonnegative_index = self._build_index_by_distance_threshold(negative_threshold)

    def __len__(self) -> int:  # noqa: D105
        return len(self.dataset_df)

    def __getitem__(self, idx: int) -> Dict[str, Union[int, Tensor]]:  # noqa: D105
        raise NotImplementedError()

    def _build_index_by_distance_threshold(self, threshold_value: float) -> List[np.ndarray]:
        """Build index of elements that satisfy a UTM distance threshold condition.

        Args:
            threshold_value (float): The maximum UTM distance between two elements
                for them to be considered positive.

        Returns:
            List[np.ndarray]: List of element indexes that satisfy the UTM distance threshold condition
                for each element in the dataset.
        """
        distances = np.linalg.norm(
            self.dataset_df[["northing", "easting"]].to_numpy(dtype=np.float64)[:, None, :]
            - self.dataset_df[["northing", "easting"]].to_numpy(dtype=np.float64)[None, :, :],
            axis=-1,
        )
        mask = distances < threshold_value
        result = [np.where(row)[0] for row in mask]
        return result

    @property
    def positives_index(self) -> List[np.ndarray]:
        """List of indexes of positive samples for each element in the dataset."""
        return self._positives_index

    @property
    def nonnegative_index(self) -> List[np.ndarray]:
        """List of indexes of non-negatives samples for each element in the dataset."""
        return self._nonnegative_index

    def collate_fn(self) -> Tuple[Dict[str, Tensor], Tensor, Tensor]:
        """Collate function for torch.utils.data.DataLoader."""
        raise NotImplementedError()
