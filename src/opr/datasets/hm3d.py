"""HM3D dataset implementation."""
from pathlib import Path
from typing import Any, Literal

import cv2

# import MinkowskiEngine as ME  # type: ignore
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
        positive_threshold: float = 2.0,
        negative_threshold: float = 10.0,
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
            image_transform (Any, optional): Image transformation to apply. Defaults to None.
            pointcloud_transform (Any, optional): Pointcloud transformation to apply. Defaults to None.
            pointcloud_set_transform (Any, optional): Pointcloud set transformation to apply. Defaults to None.

        Raises:
            ValueError: If an invalid data_to_load argument is provided.
        """
        logger.warning("HM3D dataset is in research phase. The API is subject to change.")
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

        self.data_path = (
            self.dataset_root / "hm3d_train" if subset == "train" else self.dataset_root / "hm3d_val"
        )

        self.image_transform = image_transform or DefaultHM3DImageTransform(train=(self.subset == "train"))

    def __len__(self) -> int:  # noqa: D105
        return len(self.dataset_df)

    def __getitem__(self, idx: int) -> dict[str, Any]:  # noqa: D105
        data = {"idx": torch.tensor(idx, dtype=int)}
        data["utm"] = torch.tensor(self.dataset_df.iloc[idx][["x", "y"]].to_numpy(dtype=np.float64))
        if "image_front" in self.data_to_load:
            scene_id = int(self.dataset_df.iloc[idx]["scene_id"])
            frame_id = int(self.dataset_df.iloc[idx]["frame_id"])
            image_filepath = self.data_path / f"{scene_id}" / f"{frame_id+1}_rgb.png"
            image = cv2.cvtColor(cv2.imread(str(image_filepath)), cv2.COLOR_BGR2RGB)
            if self.image_transform:
                image = self.image_transform(image)
            data["image_front"] = image
        if "image_back" in self.data_to_load:
            if idx % 4 == 0 or idx % 4 == 1:
                back_idx = idx + 2
            else:
                back_idx = idx - 2
            scene_id = int(self.dataset_df.iloc[back_idx]["scene_id"])
            frame_id = int(self.dataset_df.iloc[back_idx]["frame_id"])
            image_filepath = self.data_path / f"{scene_id}" / f"{frame_id+1}_rgb.png"
            image = cv2.cvtColor(cv2.imread(str(image_filepath)), cv2.COLOR_BGR2RGB)
            if self.image_transform:
                image = self.image_transform(image)
            data["image_back"] = image
        if "depth_front" in self.data_to_load:
            raise NotImplementedError
        if "depth_back" in self.data_to_load:
            raise NotImplementedError
        if "pointcloud_lidar" in self.data_to_load:
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
        xy = self.dataset_df[["x", "y"]].values.astype("float32")
        quats = self.dataset_df[["qw", "qx", "qy", "qz"]].values.astype("float32")
        distances = torch.cdist(torch.tensor(xy), torch.tensor(xy), p=2)
        x_angles = []
        for quat in quats:
            x_angles.append(R.from_quat(quat).as_euler("xyz", degrees=True)[0])
        x_angles = np.array(x_angles)
        angle_dists = torch.cdist(
            torch.tensor(x_angles).reshape(-1, 1), torch.tensor(x_angles).reshape(-1, 1), p=2
        )
        angle_masks = (angle_dists == 0) | (angle_dists == 180)
        positives_mask = (distances > 0) & (distances < positive_threshold) & angle_masks
        negatives_mask = distances > negative_threshold

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
        xy = self.dataset_df[["x", "y"]].values.astype("float32")
        quats = self.dataset_df[["qw", "qx", "qy", "qz"]].values.astype("float32")
        distances = torch.cdist(torch.tensor(xy), torch.tensor(xy), p=2)
        x_angles = []
        for quat in quats:
            x_angles.append(R.from_quat(quat).as_euler("xyz", degrees=True)[0])
        x_angles = np.array(x_angles)
        angle_dists = torch.cdist(
            torch.tensor(x_angles).reshape(-1, 1), torch.tensor(x_angles).reshape(-1, 1), p=2
        )

        angle_masks = (angle_dists == 0) | (angle_dists == 180)
        positives_mask = (distances > 0) & (distances < positive_threshold) & angle_masks
        nonnegatives_mask = distances < negative_threshold

        # Convert the boolean masks to index tensors
        positive_indices = [torch.nonzero(row).squeeze(dim=-1) for row in positives_mask]
        nonnegative_indices = [torch.nonzero(row).squeeze(dim=-1) for row in nonnegatives_mask]

        return positive_indices, nonnegative_indices

    # TODO: this is the same collate_fn as in Oxford -> refactor to DRY principle
    def _collate_data_dict(self, data_list: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        result: dict[str, Tensor] = {}
        result["idxs"] = torch.stack([e["idx"] for e in data_list], dim=0)
        for data_key in data_list[0].keys():
            if data_key == "idx":
                continue
            elif data_key == "utm":
                result["utms"] = torch.stack([e["utm"] for e in data_list], dim=0)
            elif data_key.startswith("image_"):
                result[f"images_{data_key[6:]}"] = torch.stack([e[data_key] for e in data_list])
            # elif data_key.startswith("mask_"):
            #     result[f"masks_{data_key[5:]}"] = torch.stack([e[data_key] for e in data_list])
            # elif data_key == "pointcloud_lidar_coords":
            #     coords_list = [e["pointcloud_lidar_coords"] for e in data_list]
            #     feats_list = [e["pointcloud_lidar_feats"] for e in data_list]
            #     n_points = [int(e.shape[0]) for e in coords_list]
            #     coords_tensor = torch.cat(coords_list, dim=0).unsqueeze(0)  # (1,batch_size*n_points,3)
            #     if self.pointcloud_set_transform is not None:
            #         # Apply the same transformation on all dataset elements
            #         coords_tensor = self.pointcloud_set_transform(coords_tensor)
            #     coords_list = torch.split(coords_tensor.squeeze(0), split_size_or_sections=n_points, dim=0)
            #     quantized_coords_list = []
            #     quantized_feats_list = []
            #     for coords, feats in zip(coords_list, feats_list):
            #         quantized_coords, quantized_feats = ME.utils.sparse_quantize(
            #             coordinates=coords,
            #             features=feats,
            #             quantization_size=self._pointcloud_quantization_size,
            #         )
            #         quantized_coords_list.append(quantized_coords)
            #         quantized_feats_list.append(quantized_feats)

            #     result["pointclouds_lidar_coords"] = ME.utils.batched_coordinates(quantized_coords_list)
            #     result["pointclouds_lidar_feats"] = torch.cat(quantized_feats_list)
            # elif data_key == "pointcloud_lidar_feats":
            #     continue
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
