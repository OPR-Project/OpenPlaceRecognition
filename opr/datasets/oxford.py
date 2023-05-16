"""PointNetVLAD Oxford RobotCar dataset implementation."""
import pickle
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch import Tensor

from opr.datasets.augmentations import (
    DefaultCloudSetTransform,
    DefaultCloudTransform,
    DefaultImageTransform,
    DefaultSemanticTransform,
    OneHotSemanticTransform
)
from opr.datasets.base import BaseDataset


class OxfordDataset(BaseDataset):
    """PointNetVLAD Oxford RobotCar dataset implementation."""

    images_subdir: Optional[Path] = None
    clouds_subdir: Optional[Path] = None
    semantic_subdir: Optional[Path] = None

    def __init__(
        self,
        dataset_root: Union[str, Path],
        subset: Literal["train", "val", "test"] = "train",
        modalities: Union[str, Tuple[str, ...]] = ("image", "cloud"),
        images_subdir: Optional[Union[str, Path]] = "stereo_centre",
        semantic_subdir: Optional[Union[str, Path]] = 'stereo_centre_segmentation',
        random_select_nearest_images: bool = False,
        mink_quantization_size: Optional[float] = 0.01,
    ) -> None:
        """Oxford RobotCar dataset implementation.

        Original dataset site: https://robotcar-dataset.robots.ox.ac.uk/

        We use the preprocessed version of the dataset that was introduced
        in PointNetVLAD paper: https://arxiv.org/abs/1804.03492

        Args:
            dataset_root (Union[str, Path]): Path to the dataset root directory.
            subset (Literal["train", "val", "test"]): Current subset to load. Defaults to "train".
            modalities (Union[str, Tuple[str, ...]]): List of modalities for which the data should be loaded.
                Defaults to ( "image", "cloud").
            images_subdir (Union[str, Path], optional): Images subdirectory path. Defaults to "stereo_centre".
            random_select_nearest_images (bool): Whether to select random nearest top-20 images
                as described in "MinkLoc++" paper. Defaults to False.
            mink_quantization_size (float, optional): The quantization size for point clouds.
                Defaults to 0.01.

        Raises:
            FileNotFoundError: If there is no "subset_lidar2image_index.pickle" file.
            ValueError: If images_subdir is undefined when "images" in modalities.
        """
        super().__init__(dataset_root, subset, modalities)

        # lidar2image_index_pickle = self.dataset_root / f"{subset}_lidar2image_index.pickle"
        # if not lidar2image_index_pickle.exists():
        #     raise FileNotFoundError(
        #         f"There is no {subset}_lidar2image_index.pickle file in given dataset_root={self.dataset_root}."
        #         "Consider checking documentation on how to preprocess the dataset."
        #     )
        # with open(lidar2image_index_pickle, "rb") as f:
        #     self.lidar2image_index = pickle.load(f)
        # self.random_select_nearest_images = random_select_nearest_images

        if "image" in self.modalities:
            if images_subdir:
                self.images_subdir = Path(images_subdir)
            else:
                raise ValueError(
                    "Given 'images' in 'modalities' argument, but 'images_subdir' is set to None"
                )

        if "semantic" in self.modalities:
            if semantic_subdir:
                self.semantic_subdir = Path(semantic_subdir)
            else:
                raise ValueError(
                    "Given 'semantic' in 'modalities' argument, but 'semantic_subdir' is set to None"
                )

        if "cloud" in self.modalities:
            if subset in ("train", "val"):
                self.clouds_subdir = Path("pointcloud_20m_10overlap")
            else:
                self.clouds_subdir = Path("pointcloud_20m")

        self.image_transform = DefaultImageTransform(train=(self.subset == "train"))
        self.semantic_transform = DefaultSemanticTransform(train=(self.subset == "train"))
        self.cloud_transform = DefaultCloudTransform(train=(self.subset == "train"))
        self.cloud_set_transform = DefaultCloudSetTransform(train=(self.subset == "train"))

        self.mink_quantization_size = mink_quantization_size

    def __getitem__(self, idx: int) -> Dict[str, Union[int, Tensor]]:  # noqa: D105
        data: Dict[str, Union[int, Tensor]] = {"idx": idx}
        row = self.dataset_df.iloc[idx]
        data["utm"] = torch.tensor(row[["northing", "easting"]].to_numpy(dtype=np.float64))
        track_dir = self.dataset_root / str(row["track"])
        if "image" in self.modalities and self.images_subdir is not None:
            # if self.subset == "test" or not self.random_select_nearest_images:
            #     image_ts = self.lidar2image_index[idx][0]
            #     im_filepath = track_dir / self.images_subdir / f"{image_ts}.png"
            # else:
            #     image_ts = np.random.choice(self.lidar2image_index[idx])
            #     im_filepath = track_dir / self.images_subdir / f"{image_ts}.png"
            image_ts = int(row["stereo_centre"])
            im_filepath = track_dir / self.images_subdir / f"{image_ts}.png"
            im = cv2.imread(str(im_filepath))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = self.image_transform(im)
            data["image"] = im

        # TODO: implement multi-camera setup better?
        for cam_name in ("stereo_centre", "mono_rear", "mono_left", "mono_right"):
            if f"image_{cam_name}" in self.modalities:
                image_ts = int(row[cam_name])
                im_filepath = track_dir / f"{cam_name}_small" / f"{image_ts}.png"
                im = cv2.imread(str(im_filepath))
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = self.image_transform(im)
                data[f"image_{cam_name}"] = im

            if f"semantic_{cam_name}" in self.modalities:
                image_ts = int(row[cam_name])
                im_filepath = track_dir / f"{cam_name}_segmentation_small" / f"{image_ts}.png" # image id is equal to semantic mask id~
                im = cv2.imread(str(im_filepath), cv2.IMREAD_UNCHANGED)
                im = self.semantic_transform(im)
                data[f"semantic_{cam_name}"] = im

        if "semantic" in self.modalities and self.semantic_subdir is not None:
            image_ts = int(row["stereo_centre"])
            im_filepath = track_dir / self.semantic_subdir / f"{image_ts}.png" # image id is equal to semantic mask id~
            im = cv2.imread(str(im_filepath), cv2.IMREAD_UNCHANGED)
            # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = self.semantic_transform(im)
            data["semantic"] = im

        if "cloud" in self.modalities and self.clouds_subdir is not None:
            pc_filepath = track_dir / self.clouds_subdir / f"{row['pointcloud']}.bin"
            pc = self._load_pc(pc_filepath)
            pc = self.cloud_transform(pc)
            data["cloud"] = pc
        return data

    def _load_pc(self, filepath: Union[str, Path]) -> Tensor:
        pc = np.fromfile(filepath, dtype=np.float64)
        pc = np.reshape(pc, (pc.shape[0] // 3, 3))
        pc_tensor = torch.tensor(pc, dtype=torch.float)
        return pc_tensor

class OxfordDataset_OneHotSemantic(BaseDataset):
    """PointNetVLAD Oxford RobotCar dataset implementation."""

    images_subdir: Optional[Path] = None
    clouds_subdir: Optional[Path] = None

    def __init__(
        self,
        dataset_root: Union[str, Path],
        subset: Literal["train", "val", "test"] = "train",
        modalities: Union[str, Tuple[str, ...]] = ("image", "cloud"),
        images_subdir: Optional[Union[str, Path]] = "stereo_centre",
         semantic_subdir: Optional[Union[str, Path]] = 'stereo_centre_segmentation',
        random_select_nearest_images: bool = False,
        mink_quantization_size: Optional[float] = 0.01,
    ) -> None:
        """Oxford RobotCar dataset implementation.

        Original dataset site: https://robotcar-dataset.robots.ox.ac.uk/

        We use the preprocessed version of the dataset that was introduced
        in PointNetVLAD paper: https://arxiv.org/abs/1804.03492

        Args:
            dataset_root (Union[str, Path]): Path to the dataset root directory.
            subset (Literal["train", "val", "test"]): Current subset to load. Defaults to "train".
            modalities (Union[str, Tuple[str, ...]]): List of modalities for which the data should be loaded.
                Defaults to ( "image", "cloud").
            images_subdir (Union[str, Path], optional): Images subdirectory path. Defaults to "stereo_centre".
            random_select_nearest_images (bool): Whether to select random nearest top-20 images
                as described in "MinkLoc++" paper. Defaults to False.
            mink_quantization_size (float, optional): The quantization size for point clouds.
                Defaults to 0.01.

        Raises:
            FileNotFoundError: If there is no "subset_lidar2image_index.pickle" file.
            ValueError: If images_subdir is undefined when "images" in modalities.
        """
        super().__init__(dataset_root, subset, modalities)

        # lidar2image_index_pickle = self.dataset_root / f"{subset}_lidar2image_index.pickle"
        # if not lidar2image_index_pickle.exists():
        #     raise FileNotFoundError(
        #         f"There is no {subset}_lidar2image_index.pickle file in given dataset_root={self.dataset_root}."
        #         "Consider checking documentation on how to preprocess the dataset."
        #     )
        # with open(lidar2image_index_pickle, "rb") as f:
        #     self.lidar2image_index = pickle.load(f)
        # self.random_select_nearest_images = random_select_nearest_images

        if "image" in self.modalities:
            if images_subdir:
                self.images_subdir = Path(images_subdir)
            else:
                raise ValueError(
                    "Given 'images' in 'modalities' argument, but 'images_subdir' is set to None"
                )

        if "semantic" in self.modalities:
            if semantic_subdir:
                self.semantic_subdir = Path(semantic_subdir)
            else:
                raise ValueError(
                    "Given 'semantic' in 'modalities' argument, but 'semantic_subdir' is set to None"
                )

        if "cloud" in self.modalities:
            if subset in ("train", "val"):
                self.clouds_subdir = Path("pointcloud_20m_10overlap")
            else:
                self.clouds_subdir = Path("pointcloud_20m")

        self.image_transform = DefaultImageTransform(train=(self.subset == "train"))
        self.semantic_transform = OneHotSemanticTransform(train=(self.subset == "train"))
        self.cloud_transform = DefaultCloudTransform(train=(self.subset == "train"))
        self.cloud_set_transform = DefaultCloudSetTransform(train=(self.subset == "train"))

        self.mink_quantization_size = mink_quantization_size

    def __getitem__(self, idx: int) -> Dict[str, Union[int, Tensor]]:  # noqa: D105
        data: Dict[str, Union[int, Tensor]] = {"idx": idx}
        row = self.dataset_df.iloc[idx]
        data["utm"] = torch.tensor(row[["northing", "easting"]].to_numpy(dtype=np.float64))
        track_dir = self.dataset_root / str(row["track"])
        if "image" in self.modalities and self.images_subdir is not None:
            # if self.subset == "test" or not self.random_select_nearest_images:
            #     image_ts = self.lidar2image_index[idx][0]
            #     im_filepath = track_dir / self.images_subdir / f"{image_ts}.png"
            # else:
            #     image_ts = np.random.choice(self.lidar2image_index[idx])
            #     im_filepath = track_dir / self.images_subdir / f"{image_ts}.png"
            image_ts = int(row["stereo_centre"])
            im_filepath = track_dir / self.images_subdir / f"{image_ts}.png"
            im = cv2.imread(str(im_filepath))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = self.image_transform(im)
            data["image"] = im

        # TODO: implement multi-camera setup better?
        for cam_name in ("stereo_centre", "mono_rear", "mono_left", "mono_right"):
            if f"image_{cam_name}" in self.modalities:
                image_ts = int(row[cam_name])
                im_filepath = track_dir / f"{cam_name}_small" / f"{image_ts}.png"
                im = cv2.imread(str(im_filepath))
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = self.image_transform(im)
                data[f"image_{cam_name}"] = im

        if "semantic" in self.modalities and self.semantic_subdir is not None:
            image_ts = int(row["stereo_centre"])
            im_filepath = track_dir / self.semantic_subdir / f"{image_ts}.png" # image id is equal to semantic mask id~
            im = cv2.imread(str(im_filepath), cv2.IMREAD_UNCHANGED)
            # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = self.semantic_transform(im)
            data["semantic"] = im

        if "cloud" in self.modalities and self.clouds_subdir is not None:
            pc_filepath = track_dir / self.clouds_subdir / f"{row['pointcloud']}.bin"
            pc = self._load_pc(pc_filepath)
            pc = self.cloud_transform(pc)
            data["cloud"] = pc
        return data

    def _load_pc(self, filepath: Union[str, Path]) -> Tensor:
        pc = np.fromfile(filepath, dtype=np.float64)
        pc = np.reshape(pc, (pc.shape[0] // 3, 3))
        pc_tensor = torch.tensor(pc, dtype=torch.float)
        return pc_tensor
