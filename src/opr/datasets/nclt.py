"""NCLT dataset implementation."""
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import cv2
import MinkowskiEngine as ME  # type: ignore
import numpy as np
import torch
from torch import Tensor

from opr.datasets.base import BasePlaceRecognitionDataset
from opr.utils import cartesian_to_spherical


class NCLTDataset(BasePlaceRecognitionDataset):
    """NCLT dataset implementation."""

    _images_dirname: str
    _masks_dirname: str
    _pointclouds_dirname: str
    _pointcloud_quantization_size: Optional[Union[float, Tuple[float, float, float]]]
    _max_point_distance: Optional[float]
    _spherical_coords: bool
    _use_intensity_values: bool
    _valid_data: Tuple[str, ...] = (
        "image_Cam0",
        "image_Cam1",
        "image_Cam2",
        "image_Cam3",
        "image_Cam4",
        "image_Cam5",
        "pointcloud_lidar",
        "mask_Cam0",
        "mask_Cam1",
        "mask_Cam2",
        "mask_Cam3",
        "mask_Cam4",
        "mask_Cam5",
        # TODO: add text embeddings data
    )

    def __init__(
        self,
        dataset_root: Union[str, Path],
        subset: Literal["train", "val", "test"],
        data_to_load: Union[str, Tuple[str, ...]],
        positive_threshold: float = 10.0,
        negative_threshold: float = 50.0,
        images_dirname: str = "images_small",
        masks_dirname: str = "segmentation_masks_small",
        pointclouds_dirname: str = "velodyne_data",
        pointcloud_quantization_size: Optional[Union[float, Tuple[float, float, float]]] = 0.5,
        max_point_distance: Optional[float] = None,
        spherical_coords: bool = False,
        use_intensity_values: bool = False,
        image_transform: Optional[Any] = None,
        semantic_transform: Optional[Any] = None,
        pointcloud_transform: Optional[Any] = None,
        pointcloud_set_transform: Optional[Any] = None,
    ) -> None:
        """NCLT dataset implementation.

        Args:
            dataset_root (Union[str, Path]): Path to the dataset root directory.
            subset (Literal["train", "val", "test"]): Current subset to load. Defaults to "train".
            data_to_load (Union[str, Tuple[str, ...]]): The list of data to load.
                Check the documentation for the list of available data.
            positive_threshold (float): The UTM distance threshold value for positive samples.
                Defaults to 10.0.
            negative_threshold (float): The UTM distance threshold value for negative samples.
                Defaults to 50.0.
            images_dirname (str): Images directory name. It should be specified explicitly
                if custom preprocessing was done. Defaults to "images".
            masks_dirname (str): Masks directory name. It should be specified explicitly
                if custom preprocessing was done. Defaults to "segmentation_masks".
            pointclouds_dirname (str): Point clouds directory name. It should be specified
                explicitly if custom preprocessing was done. Defaults to "velodyne_data".
            pointcloud_quantization_size (float, optional): The quantization size for point clouds.
                Defaults to 0.01.
            max_point_distance (float, optional): The maximum distance of points from the origin.
                Defaults to None.
            spherical_coords (bool): Whether to use spherical coordinates for point clouds.
                Defaults to False.
            use_intensity_values (bool): Whether to use intensity values for point clouds. Defaults to False.
            image_transform (Any, optional): Images transform. If None, DefaultImageTransform will be used.
                Defaults to None.
            semantic_transform (Any, optional): Semantic masks transform. If None, DefaultSemanticTransform
                will be used. Defaults to None.
            pointcloud_transform (Any, optional): Point clouds transform. If None, DefaultCloudTransform
                will be used. Defaults to None.
            pointcloud_set_transform (Any, optional): Point clouds set transform. If None,
                DefaultCloudSetTransform will be used. Defaults to None.

        Raises:
            ValueError: If data_to_load contains invalid data source names.
            FileNotFoundError: If images, masks or pointclouds directory does not exist.
        """
        # TODO: ^ docstring is also not DRY -> it is almost the same as in Oxford dataset
        super().__init__(
            dataset_root,
            subset,
            data_to_load,
            positive_threshold,
            negative_threshold,
            image_transform,
            semantic_transform,
            pointcloud_transform,
            pointcloud_set_transform,
        )

        if subset == "test":
            self.dataset_df["in_query"] = True  # for compatibility with Oxford Dataset

        if any(elem not in self._valid_data for elem in self.data_to_load):
            raise ValueError(f"Invalid data_to_load argument. Valid data list: {self._valid_data!r}")

        _track_name = self.dataset_df.iloc[0]["track"]

        if any(elem.startswith("image") for elem in self.data_to_load):
            self._images_dirname = images_dirname
            if not (self.dataset_root / _track_name / self._images_dirname).exists():
                raise FileNotFoundError(f"Images directory {self._images_dirname!r} does not exist.")

        if any(elem.startswith("mask") for elem in self.data_to_load):
            self._masks_dirname = masks_dirname
            if not (self.dataset_root / _track_name / self._masks_dirname).exists():
                raise FileNotFoundError(f"Masks directory {self._masks_dirname!r} does not exist.")

        if "pointcloud_lidar" in self.data_to_load:
            self._pointclouds_dirname = pointclouds_dirname
            if not (self.dataset_root / _track_name / self._pointclouds_dirname).exists():
                raise FileNotFoundError(
                    f"Pointclouds directory {self._pointclouds_dirname!r} does not exist."
                )

        self._pointcloud_quantization_size = pointcloud_quantization_size
        self._max_point_distance = max_point_distance
        self._spherical_coords = spherical_coords
        self._use_intensity_values = use_intensity_values

    # TODO: apply DRY principle -> this is almost the same as in Oxford dataset
    def __getitem__(self, idx: int) -> Dict[str, Tensor]:  # noqa: D105
        row = self.dataset_df.iloc[idx]
        data = {"idx": torch.tensor(idx, dtype=int)}
        data["utm"] = torch.tensor(row[["northing", "easting"]].to_numpy(dtype=np.float64))
        track_dir = self.dataset_root / str(row["track"])

        for data_source in self.data_to_load:
            if data_source.startswith("image_"):
                cam_name = data_source[6:]  # remove "image_" prefix
                image_ts = int(row["image"])
                im_filepath = track_dir / self._images_dirname / f"{cam_name}" / f"{image_ts}.png"
                im = cv2.imread(str(im_filepath))
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = self.image_transform(im)
                data[data_source] = im
            elif data_source.startswith("mask_"):
                cam_name = data_source[5:]  # remove "mask_" prefix
                image_ts = int(row["image"])
                mask_filepath = track_dir / self._masks_dirname / f"{cam_name}" / f"{image_ts}.png"
                mask = cv2.imread(str(mask_filepath), cv2.IMREAD_UNCHANGED)
                mask = self.semantic_transform(mask)
                data[data_source] = mask
            elif data_source == "pointcloud_lidar":
                pc_filepath = track_dir / self._pointclouds_dirname / f"{row['pointcloud']}.bin"
                pointcloud = self._load_pc(pc_filepath)
                data[f"{data_source}_coords"] = self.pointcloud_transform(pointcloud[:, :3])
                if self._use_intensity_values:
                    data[f"{data_source}_feats"] = pointcloud[:, 3].unsqueeze(1)
                else:
                    data[f"{data_source}_feats"] = torch.ones_like(pointcloud[:, :1])

        return data

    def _load_pc(self, filepath: Union[str, Path]) -> Tensor:
        if self._use_intensity_values:
            raise NotImplementedError("Intensity values are not supported yet.")
        pc = np.fromfile(filepath, dtype=np.float32).reshape(-1, 3)  # TODO: preprocess pointclouds properly
        if self._spherical_coords:
            pc = cartesian_to_spherical(pc, dataset_name="nclt")
        if self._max_point_distance is not None:
            pc = pc[np.linalg.norm(pc, axis=1) < self._max_point_distance]
        pc_tensor = torch.tensor(pc, dtype=torch.float)
        return pc_tensor

    # TODO: this is the same collate_fn as in Oxford -> refactor to DRY principle
    def _collate_data_dict(self, data_list: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        result: Dict[str, Tensor] = {}
        result["idxs"] = torch.stack([e["idx"] for e in data_list], dim=0)
        for data_key in data_list[0].keys():
            if data_key == "idx":
                continue
            elif data_key == "utm":
                result["utms"] = torch.stack([e["utm"] for e in data_list], dim=0)
            elif data_key.startswith("image_"):
                result[f"images_{data_key[6:]}"] = torch.stack([e[data_key] for e in data_list])
            elif data_key.startswith("mask_"):
                result[f"masks_{data_key[5:]}"] = torch.stack([e[data_key] for e in data_list])
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

    def collate_fn(self, data_list: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        """Pack input data list into batch.

        Args:
            data_list (List[Dict[str, Tensor]]): batch data list generated by DataLoader.

        Returns:
            Dict[str, Tensor]: dictionary of batched data.
        """
        return self._collate_data_dict(data_list)
