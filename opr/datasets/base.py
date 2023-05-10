"""Base dataset implementation."""
import pickle
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import Dataset

from opr.datasets.augmentations import (
    DefaultCloudSetTransform,
    DefaultCloudTransform,
    DefaultImageTransform,
)


class BaseDataset(Dataset):
    """Base class for track-based Place Recognition dataset."""

    valid_subsets: Tuple[str, ...] = ("train", "val", "test")
    # valid_modalities: Tuple[str, ...] = ("image", "cloud")
    dataset_root: Path
    subset: Literal["train", "val", "test"]
    dataset_df: DataFrame
    modalities: Tuple[str, ...]
    images_subdir: Optional[Path] = None
    clouds_subdir: Optional[Path] = None
    positives_index: Dict[int, List[int]]
    nonnegatives_index: Dict[int, List[int]]
    image_transform: DefaultImageTransform
    cloud_transform: DefaultCloudTransform
    cloud_set_transform: DefaultCloudSetTransform
    mink_quantization_size: Optional[float]

    def __init__(
        self,
        dataset_root: Union[str, Path],
        subset: Literal["train", "val", "test"] = "train",
        modalities: Union[str, Tuple[str, ...]] = ("image", "cloud"),
    ) -> None:
        """Base class for track-based Place Recognition dataset.

        Args:
            dataset_root (Union[str, Path]): Path to the dataset root directory.
            subset (Literal["train", "val", "test"]): Current subset to load. Defaults to "train".
            modalities (Union[str, Tuple[str, ...]]): List of modalities for which the data should be loaded.
                Defaults to ( "image", "cloud").

        Raises:
            FileNotFoundError: If dataset_root doesn't exist.
            ValueError: If invalid subset given.
            FileNotFoundError: If there is no csv file for given subset.
            ValueError: If invalid modalities given.
            ValueError: If "subset_positives_index.pkl" file is missing.
            ValueError: If "subset_nonnegatives_index.pkl" file is missing.
        """
        self.dataset_root = Path(dataset_root)
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Given dataset_root={self.dataset_root} doesn't exist")

        if subset not in self.valid_subsets:
            raise ValueError(f"Invalid subset argument: '{subset}' not in {self.valid_subsets}")
        self.subset = subset
        subset_csv = self.dataset_root / f"{subset}.csv"
        if not subset_csv.exists():
            raise FileNotFoundError(
                f"There is no {subset}.csv file in given dataset_root={self.dataset_root}."
                "Consider checking documentation on how to preprocess the dataset."
            )
        self.dataset_df = pd.read_csv(subset_csv, index_col=0)

        if isinstance(modalities, str):
            modalities = tuple([modalities])
        # if not set(modalities).issubset(self.valid_modalities):
        #     raise ValueError(f"Invalid modalities argument: '{modalities}' not in {self.valid_modalities}")
        self.modalities = modalities

        positives_index_pkl = self.dataset_root / f"{subset}_positives_index.pkl"
        if not positives_index_pkl.exists():
            raise ValueError(f"Missing '{subset}_positives_index.pkl' file.")
        with open(positives_index_pkl, "rb") as f:
            self.positives_index = pickle.load(f)
        nonnegatives_index_pkl = self.dataset_root / f"{subset}_nonnegatives_index.pkl"
        if not nonnegatives_index_pkl.exists():
            raise ValueError(f"Missing '{subset}_nonnegatives_index.pkl' file.")
        with open(nonnegatives_index_pkl, "rb") as f:
            self.nonnegatives_index = pickle.load(f)

    def __len__(self) -> int:  # noqa: D105
        return len(self.dataset_df)

    def __getitem__(self, idx: int) -> Dict[str, Union[int, Tensor]]:  # noqa: D105
        raise NotImplementedError()

    def get_positives(self, idx: int) -> np.ndarray:
        """Return the indices of positive elements.

        Args:
            idx (int): Index of element for which the positives needs to be found.

        Returns:
            ndarray: The array of indices of positive elements.
        """
        return np.array(self.positives_index[idx])

    def get_nonnegatives(self, idx: int) -> np.ndarray:
        """Return the indices of non-negatives elements.

        Args:
            idx (int): Index of element for which the non-negatives needs to be found.

        Returns:
            ndarray: The array of indices of non-negatives elements.
        """
        return np.array(self.nonnegatives_index[idx])
