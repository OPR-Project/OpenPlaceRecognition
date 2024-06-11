"""HM3D dataset implementation."""
import gc
import pickle
from pathlib import Path
from typing import Any, Literal

import cv2
import MinkowskiEngine as ME  # type: ignore
import numpy as np
import torch
from loguru import logger
from scipy.spatial.transform import Rotation as R
from torch import Tensor

from opr.datasets.augmentations import DefaultHM3DImageTransform
from opr.datasets.base import BasePlaceRecognitionDataset


class HM3DDataset(BasePlaceRecognitionDataset):
    """HM3D dataset implementation."""

    # TODO: the current implementation inherits from BasePlaceRecognitionDataset, which is a track-based
    # dataset. HM3D is a scene-based dataset, so the current implementation is not optimal. It would be
    # better to inherit from a scene-based dataset class, but we don't have one yet. We should refactor
    # the dataset classes to have a common base class for scene-based datasets and another for track-based
    # datasets, and then inherit from the appropriate base class.

    _valid_data: tuple[str, ...] = (
        "image_front",
        "image_back",
        "depth_front",
        "depth_back",
        "pointcloud_lidar",
    )

    def __init__(
        self,
        dataset_root: str | Path,
        subset: Literal["train", "val", "test"],
        data_to_load: str | tuple[str, ...],
        positive_threshold: float = 5.0,
        negative_threshold: float = 10.0,
        positive_iou_threshold: float = 0.1,
        pointcloud_quantization_size: float = 0.1,
        max_point_distance: float = 20.0,
        image_transform: Any | None = None,
        pointcloud_transform: Any | None = None,
        pointcloud_set_transform: Any | None = None,
    ) -> None:
        """Initialize HM3D dataset.

        Args:
            dataset_root (str | Path): Path to the root directory of the dataset.
            subset (Literal["train", "val", "test"]): Dataset subset to load.
            data_to_load (str | tuple[str, ...]): Data to load from the dataset.
            positive_threshold (float): The maximum UTM distance between two elements
                for them to be considered positive. Defaults to 2.0.
            negative_threshold (float): The maximum UTM distance between two elements
                for them to be considered non-negative. Defaults to 10.0.
            positive_iou_threshold (float): The minimum IoU between two elements. Defaults to 0.1.
            pointcloud_quantization_size (float): Pointcloud quantization size. Defaults to 0.1.
            max_point_distance (float): Maximum point distance. Defaults to 20.0.
            image_transform (Any, optional): Image transformation to apply. Defaults to None.
            pointcloud_transform (Any, optional): Pointcloud transformation to apply. Defaults to None.
            pointcloud_set_transform (Any, optional): Pointcloud set transformation to apply. Defaults to None.

        Raises:
            ValueError: If an invalid data_to_load argument is provided.
        """
        logger.warning("HM3D dataset is in research phase. The API is subject to change.")
        self._positive_iou_threshold = positive_iou_threshold
        super().__init__(
            dataset_root,
            subset,
            data_to_load,
            positive_threshold,
            negative_threshold,
            image_transform=image_transform,
            pointcloud_transform=pointcloud_transform,
            pointcloud_set_transform=pointcloud_set_transform,
        )

        if subset == "test":
            self.dataset_df["in_query"] = True  # for compatibility with Oxford Dataset

        if any(elem not in self._valid_data for elem in self.data_to_load):
            raise ValueError(f"Invalid data_to_load argument. Valid data list: {self._valid_data!r}")

        self._pointcloud_quantization_size = pointcloud_quantization_size
        self._max_point_distance = max_point_distance

        self.image_transform = image_transform or DefaultHM3DImageTransform(train=(self.subset == "train"))
        self.pointcloud_transform = None
        self.pointcloud_set_transform = None

    def __len__(self) -> int:  # noqa: D105
        return len(self.dataset_df)

    def _load_image(self, idx: int) -> Tensor:
        scene_id = str(self.dataset_df.iloc[idx]["scene_id"])
        frame_id = int(self.dataset_df.iloc[idx]["frame_id"])
        dataset = str(self.dataset_df.iloc[idx]["dataset"])
        subset = "train" if self.subset == "train" else "val"
        image_filepath = self.dataset_root / f"{dataset}_{subset}" / f"{scene_id}" / f"{frame_id+1}_rgb.png"
        image = cv2.cvtColor(cv2.imread(str(image_filepath)), cv2.COLOR_BGR2RGB)
        if self.image_transform:
            image = self.image_transform(image)
        return image

    def _load_pointcloud(
        self, idx: int, position: Literal["front", "left", "back", "right"] = "front"
    ) -> Tensor:
        scene_id = str(self.dataset_df.iloc[idx]["scene_id"])
        frame_id = int(self.dataset_df.iloc[idx]["frame_id"])
        dataset = str(self.dataset_df.iloc[idx]["dataset"])
        subset = "train" if self.subset == "train" else "val"
        pointcloud_filepath = (
            self.dataset_root / f"{dataset}_{subset}" / f"{scene_id}" / f"{frame_id+1}_cloud_downsampled.npz"
        )
        pointcloud = np.load(pointcloud_filepath)["arr_0"]
        if position == "left":
            rotation = R.from_euler("z", 90, degrees=True)
            pointcloud = rotation.apply(pointcloud)
        elif position == "back":
            rotation = R.from_euler("z", 180, degrees=True)  # rotate 180 degrees around last axis
            pointcloud = rotation.apply(pointcloud)
        elif position == "right":
            rotation = R.from_euler("z", -90, degrees=True)
            pointcloud = rotation.apply(pointcloud)
        pointcloud = torch.tensor(pointcloud, dtype=torch.float32)
        if self.pointcloud_transform:
            pointcloud = self.pointcloud_transform(pointcloud)
        if self._max_point_distance is not None:
            pointcloud = pointcloud[np.linalg.norm(pointcloud, axis=1) < self._max_point_distance]
        return pointcloud

    def _load_and_concat_pointcloud(
        self,
        front_idx: int,
        left_idx: int | None = None,
        back_idx: int | None = None,
        right_idx: int | None = None,
    ) -> Tensor:
        all_pc = torch.tensor([], dtype=torch.float32)
        for idx, position in zip(
            [front_idx, left_idx, back_idx, right_idx], ["front", "left", "back", "right"]
        ):
            if idx is not None:
                pointcloud = self._load_pointcloud(idx, position)
                all_pc = torch.cat([all_pc, pointcloud], dim=0)
        return all_pc

    def _get_left_idx(self, idx: int) -> int:
        return idx + 1 if idx % 4 in [0, 1, 2] else idx - 3

    def _get_back_idx(self, idx: int) -> int:
        return idx + 2 if idx % 4 in [0, 1] else idx - 2

    def _get_right_idx(self, idx: int) -> int:
        return idx - 1 if idx % 4 in [1, 2, 3] else idx + 3

    def __getitem__(self, idx: int) -> dict[str, Any]:  # noqa: D105
        data = {"idx": torch.tensor(idx, dtype=int)}
        data["utm"] = torch.tensor(self.dataset_df.iloc[idx][["x", "y"]].to_numpy(dtype=np.float64))
        theta = R.from_quat(
            self.dataset_df.iloc[idx][["qw", "qx", "qz", "qy"]].to_numpy(dtype=np.float64)
        ).as_euler("xzy", degrees=True)[-1]
        data["theta"] = torch.tensor(theta, dtype=torch.float64)
        left_idx = self._get_left_idx(idx)
        back_idx = self._get_back_idx(idx)
        right_idx = self._get_right_idx(idx)

        for data_type in self.data_to_load:
            if data_type == "image_front":
                data[data_type] = self._load_image(idx)
            elif data_type == "image_back":
                data[data_type] = self._load_image(back_idx)
            elif data_type == "pointcloud_lidar":
                data["pointcloud_lidar_coords"] = self._load_and_concat_pointcloud(
                    front_idx=idx, left_idx=left_idx, back_idx=back_idx, right_idx=right_idx
                )
                data["pointcloud_lidar_feats"] = torch.ones_like(
                    data["pointcloud_lidar_coords"][:, 0], dtype=torch.float32
                ).unsqueeze(1)
            elif data_type in ["depth_front", "depth_back"]:
                raise NotImplementedError

        return data

    def _build_masks(self, positive_threshold: float, negative_threshold: float) -> tuple[Tensor, Tensor]:
        """Build boolean masks for dataset elements that satisfy a UTM distance threshold condition.

        Args:
            positive_threshold (float): The maximum UTM distance between two elements
                for them to be considered positive.
            negative_threshold (float): The maximum UTM distance between two elements
                for them to be considered non-negative.

        Returns:
            Tuple[Tensor, Tensor]: A tuple of two boolean masks that satisfy the UTM distance threshold
                condition for each element in the dataset. The first mask contains the indices of elements
                that satisfy the positive threshold, while the second mask contains the indices of elements
                that satisfy the negative threshold.
        """
        if positive_threshold > 5:
            logger.warning("Positive threshold is too high. Recommended maximum value is 5.")
        _pos_mask_filepath = (
            self.dataset_root
            / f"{'train' if self.subset == 'train' else 'val'}_positives_mask_threshold{int(positive_threshold)}.pt"
        )
        _neg_mask_filepath = (
            self.dataset_root
            / f"{'train' if self.subset == 'train' else 'val'}_negatives_mask_threshold{int(negative_threshold)}.pt"
        )
        if _pos_mask_filepath.exists() and _neg_mask_filepath.exists():
            logger.debug("Loading masks from file")
            positives_mask = torch.load(_pos_mask_filepath)
            negatives_mask = torch.load(_neg_mask_filepath)
        else:
            logger.debug("Files with masks not found, calculating from scratch")
            xy = self.dataset_df[["x", "y"]].to_numpy(dtype=np.float64)
            distances = torch.cdist(torch.tensor(xy), torch.tensor(xy), p=2)
            logger.debug("Calculating positives_mask")
            positives_mask = (distances > 0) & (distances < positive_threshold)
            logger.debug("Calculating negatives_mask")
            negatives_mask = distances > negative_threshold
            del xy, distances
            gc.collect()

        positives_iou_values = torch.load(
            f"{self.dataset_root}/{'train' if self.subset == 'train' else 'val'}_positives_iou.pt"
        )
        logger.debug(f"Positives IoU values shape: {positives_iou_values.shape}")
        logger.debug(f"Positives IoU values dtype: {positives_iou_values.dtype}")
        _positives_iou_values_mem = (positives_iou_values.element_size() * positives_iou_values.numel()) // (
            1024**2
        )
        logger.debug(f"positives_iou_values memory: {_positives_iou_values_mem} MB")
        positives_iou_mask = positives_iou_values > self._positive_iou_threshold
        _positives_iou_mask_mem = (positives_iou_mask.element_size() * positives_iou_mask.numel()) // (
            1024**2
        )
        logger.debug(f"positives_iou_mask memory: {_positives_iou_mask_mem} MB")
        logger.debug("Calculating positives_mask with respect to IoU mask")
        positives_mask = positives_mask & positives_iou_mask
        logger.debug(f"Number of positive pairs: {positives_mask.sum().item()}")
        logger.debug(
            f"Number of non-zero rows in positives_mask: {(positives_mask.sum(dim=1) > 0).sum().item()}"
        )

        logger.debug("Returning masks")
        return positives_mask, negatives_mask

    def _build_indexes(
        self, positive_threshold: float, negative_threshold: float
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Build index of elements that satisfy a UTM distance threshold condition.

        Args:
            positive_threshold (float): The maximum UTM distance between two elements
                for them to be considered positive.
            negative_threshold (float): The maximum UTM distance between two elements
                for them to be considered non-negative.

        Returns:
            Tuple[List[Tensor], List[Tensor]]: Tuple (positive_indices, nonnegative_indices)
                of two lists of element indexes that satisfy the UTM distance threshold condition
                for each element in the dataset.
        """
        positives_mask = self._positives_mask
        positive_indices = [torch.nonzero(row).squeeze(dim=-1) for row in positives_mask]

        _nonneg_index_filepath = (
            self.dataset_root / f"{'train' if self.subset == 'train' else 'val'}_nonnegative_index.pkl"
        )
        if _nonneg_index_filepath.exists():
            logger.debug("Loading nonnegative index from file")
            with open(_nonneg_index_filepath, "rb") as file:
                # !!! LOADING PICKLES IS DANGEOURS, USE WITH CAUTION
                nonnegative_indices = pickle.load(file)  # noqa: S301
        else:
            xy = self.dataset_df[["x", "y"]].values.astype("float32")
            distances = torch.cdist(torch.tensor(xy), torch.tensor(xy), p=2)
            nonnegatives_mask = distances < negative_threshold
            nonnegative_indices = [torch.nonzero(row).squeeze(dim=-1) for row in nonnegatives_mask]

        return positive_indices, nonnegative_indices

    # TODO: this is almost the same collate_fn as in Oxford -> refactor to DRY principle
    def _collate_data_dict(self, data_list: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        result: dict[str, Tensor] = {}
        result["idxs"] = torch.stack([e["idx"] for e in data_list], dim=0)
        for data_key in data_list[0].keys():
            if data_key == "idx" or data_key == "theta":
                continue
            elif data_key == "utm":
                result["utms"] = torch.stack([e["utm"] for e in data_list], dim=0)
            elif data_key.startswith("image_"):
                result[f"images_{data_key[6:]}"] = torch.stack([e[data_key] for e in data_list])
            # elif data_key.startswith("mask_"):
            #     result[f"masks_{data_key[5:]}"] = torch.stack([e[data_key] for e in data_list])
            elif data_key == "pointcloud_lidar_coords":
                coords_list = [e["pointcloud_lidar_coords"] for e in data_list]
                feats_list = [e["pointcloud_lidar_feats"] for e in data_list]
                n_points = [int(e.shape[0]) for e in coords_list]
                coords_tensor = torch.cat(coords_list, dim=0).unsqueeze(0)  # (1,batch_size*n_points,3)
                if self.pointcloud_set_transform is not None:
                    # Apply the same transformation on all dataset elements
                    coords_tensor = self.pointcloud_set_transform(coords_tensor)
                coords_list = torch.split(coords_tensor.squeeze(0), split_size_or_sections=n_points, dim=0)
                quantized_coords_list = []
                quantized_feats_list = []
                for coords, feats in zip(coords_list, feats_list):
                    quantized_coords, quantized_feats = ME.utils.sparse_quantize(
                        coordinates=coords,
                        features=feats,
                        quantization_size=self._pointcloud_quantization_size,
                    )
                    quantized_coords_list.append(quantized_coords)
                    quantized_feats_list.append(quantized_feats)

                result["pointclouds_lidar_coords"] = ME.utils.batched_coordinates(quantized_coords_list)
                result["pointclouds_lidar_feats"] = torch.cat(quantized_feats_list)
            elif data_key == "pointcloud_lidar_feats":
                continue
            else:
                raise ValueError(f"Unknown data key: {data_key!r}")
        return result

    def collate_fn(self, data_list: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        """Pack input data list into batch.

        Args:
            data_list (List[Dict[str, Tensor]]): batch data list generated by DataLoader.

        Returns:
            Dict[str, Tensor]: dictionary of batched data.
        """
        return self._collate_data_dict(data_list)
