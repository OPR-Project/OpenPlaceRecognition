"""NCLT dataset implementation."""
from pathlib import Path
from time import time
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import cv2
import numpy as np
import open3d as o3d
import torch
from loguru import logger
from torch import Tensor
from omegaconf import OmegaConf

from opr.datasets.base import BasePlaceRecognitionDataset
from opr.datasets.projection import NCLTProjector
from opr.datasets.soc_utils import (
    get_points_labels_by_mask,
    instance_masks_to_objects,
    pack_objects,
    semantic_mask_to_instances,
)
from opr.utils import cartesian_to_spherical

try:
    import MinkowskiEngine as ME  # type: ignore

    minkowski_available = True
except ImportError:
    logger.warning("MinkowskiEngine is not installed. Some features may not be available.")
    minkowski_available = False


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
        use_minkowski: bool = True,
        pointcloud_quantization_size: Optional[Union[float, Tuple[float, float, float]]] = 0.5,
        max_point_distance: Optional[float] = None,
        normalize_point_cloud: bool = False,
        num_points_sample: int | None = None,
        spherical_coords: bool = False,
        use_intensity_values: bool = False,
        image_transform: Optional[Any] = None,
        semantic_transform: Optional[Any] = None,
        pointcloud_transform: Optional[Any] = None,
        pointcloud_set_transform: Optional[Any] = None,
        load_soc: bool = False,
        top_k_soc: int = 10,
        soc_coords_type: Literal["cylindrical_3d", "cylindrical_2d", "euclidean", "spherical"] = "euclidean",
        max_distance_soc: float = 50.0,
        anno: OmegaConf = None,
        exclude_dynamic: bool = False,
        dynamic_labels: Optional[list] = None,
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
            use_minkowski (bool): Whether to use MinkowskiEngine to collate point clouds in batches.
                Defaults to True.
            pointcloud_quantization_size (float, optional): The quantization size for point clouds.
                Defaults to 0.01.
            max_point_distance (float, optional): The maximum distance of points from the origin.
                Defaults to None.
            normalize_point_cloud (bool): Whether to normalize point clouds by max_point_distance.
                Defaults to False.
            num_points_sample (int, optional): The number of points to sample from the point cloud.
                Defaults to None, which means no sampling.
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
            load_soc (bool): Whether to load SOC (Semantic Objects in Context) data. Defaults to False.
            top_k_soc (int): The number of objects to keep in SOC data. Defaults to 10.
            soc_coords_type (Literal["cylindrical_3d", "cylindrical_2d", "euclidean", "spherical"]): The type
                of coordinates to use in SOC data. Defaults to "euclidean".
            max_distance_soc (float): The maximum distance of objects in SOC data. Defaults to 50.0.
            anno (OmegaConf): The annotation configuration. Defaults to None.
            exclude_dynamic (bool): Whether to exclude dynamic objects from the point cloud. Defaults to False.
            dynamic_labels (Optional[list]): The list of dynamic labels. Defaults to None.

        Raises:
            ValueError: If data_to_load contains invalid data source names.
            FileNotFoundError: If images, masks or pointclouds directory does not exist.
            ValueError: If num_points_sample is not specified and MinkowskiEngine is not used.
            ValueError: If max_point_distance is not specified and normalize_point_cloud is set to True.
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

        self._use_minkowski = use_minkowski
        self._num_points_sample = num_points_sample
        if self._num_points_sample is None and not self._use_minkowski:
            raise ValueError(
                "num_points_sample must be specified if MinkowskiEngine is not used to collate data in batch."
            )

        self._pointcloud_quantization_size = pointcloud_quantization_size
        self._max_point_distance = max_point_distance
        self._normalize_point_cloud = normalize_point_cloud
        if not self._max_point_distance and self._normalize_point_cloud:
            raise ValueError("max_point_distance must be specified if normalize_point_cloud is set to True.")
        self._spherical_coords = spherical_coords
        self._use_intensity_values = use_intensity_values

        self.load_soc = load_soc
        self.front_cam_proj = NCLTProjector(front=True)
        self.back_cam_proj = NCLTProjector(front=False)
        self.top_k_soc = top_k_soc
        self.max_distance_soc = max_distance_soc
        self.soc_coords_type = soc_coords_type
        if self.soc_coords_type not in ("cylindrical_3d", "cylindrical_2d", "euclidean", "spherical"):
            raise ValueError(f"Unknown soc_coords_type: {soc_coords_type!r}")
        self.anno = anno
        if anno:
            self.special_classes = [
                self.anno.staff_classes.index(special) for special in self.anno.special_classes
            ]
        self.exclude_dynamic = exclude_dynamic
        self.dynamic_labels = dynamic_labels

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
                if self.exclude_dynamic:
                    im = self._mask_dynamic_pixels(im, cam_name, self.dynamic_labels, idx)
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
                if self.exclude_dynamic:
                    pointcloud = self._mask_dynamic_points(pointcloud, self.dynamic_labels, idx)
                data[f"{data_source}_coords"] = self.pointcloud_transform(pointcloud[:, :3])
                if self._use_intensity_values:
                    data[f"{data_source}_feats"] = pointcloud[:, 3].unsqueeze(1)
                else:
                    data[f"{data_source}_feats"] = torch.ones_like(pointcloud[:, :1])

        if self.load_soc:
            soc = self._get_soc(idx)
            data["soc"] = soc

        return data

    def _load_pc(self, filepath: Union[str, Path], torch_tensor: bool = True) -> Tensor:
        if self._use_intensity_values:
            raise NotImplementedError("Intensity values are not supported yet.")
        pc = np.fromfile(filepath, dtype=np.float32).reshape(-1, 3)  # TODO: preprocess pointclouds properly
        if self._max_point_distance is not None:
            pc = pc[np.linalg.norm(pc, axis=1) < self._max_point_distance]
        if self._normalize_point_cloud:
            pc = pc / self._max_point_distance
        if self._spherical_coords:
            pc = cartesian_to_spherical(pc, dataset_name="nclt")
        if torch_tensor:
            pc_tensor = torch.tensor(pc, dtype=torch.float)
            return pc_tensor
        else:
            return pc

    def _mask_dynamic_points(self, pointcloud: Tensor, dynamic_labels: list, idx: int) -> Tensor:
        row = self.dataset_df.iloc[idx]
        image_ts = int(row["image"])
        track_dir = self.dataset_root / str(row["track"])

        mask_front_filepath = track_dir / self._masks_dirname / "Cam5" / f"{image_ts}.png"
        mask_front = cv2.imread(str(mask_front_filepath), cv2.IMREAD_UNCHANGED).transpose(1, 0)
        mask_back_filepath = track_dir / self._masks_dirname / "Cam2" / f"{image_ts}.png"
        mask_back = cv2.imread(str(mask_back_filepath), cv2.IMREAD_UNCHANGED).transpose(1, 0)

        coords_front, _, in_image_front = self.front_cam_proj(pointcloud)
        coords_back, _, in_image_back = self.back_cam_proj(pointcloud)

        point_labels = np.zeros(len(pointcloud), dtype=np.uint8)
        point_labels[in_image_front] = get_points_labels_by_mask(coords_front, mask_front)
        point_labels[in_image_back] = get_points_labels_by_mask(coords_back, mask_back)
        return pointcloud[np.isin(point_labels, dynamic_labels, invert=True)]

    def _mask_dynamic_pixels(self, im: np.array, cam_name: str, dynamic_labels: list, idx: int) -> np.array:
        row = self.dataset_df.iloc[idx]
        image_ts = int(row["image"])
        track_dir = self.dataset_root / str(row["track"])

        mask_filepath = track_dir / self._masks_dirname / cam_name / f"{image_ts}.png"
        mask = cv2.imread(str(mask_filepath), cv2.IMREAD_UNCHANGED)
        im[np.isin(mask, dynamic_labels)] = 0
        return im

    def _collate_pc_minkowski(self, data_list: List[Dict[str, Tensor]]) -> tuple[Tensor, Tensor]:
        if not minkowski_available:
            raise RuntimeError("MinkowskiEngine is not installed. Cannot process point clouds.")
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
        return ME.utils.batched_coordinates(quantized_coords_list), torch.cat(quantized_feats_list)

    def _collate_pc(self, data_list: List[Dict[str, Tensor]]) -> Tensor:
        coords_list = [e["pointcloud_lidar_coords"] for e in data_list]
        coords_list = [self._random_point_sample(coords, self._num_points_sample) for coords in coords_list]
        # TODO: add support for features tensor
        return torch.stack(coords_list)  # B x NUM_POINTS_FPS x 3

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
            elif data_key == "soc":
                result["soc"] = torch.stack([e["soc"] for e in data_list], dim=0)
            elif data_key == "pointcloud_lidar_coords":
                if self._use_minkowski:
                    (
                        result["pointclouds_lidar_coords"],
                        result["pointclouds_lidar_feats"],
                    ) = self._collate_pc_minkowski(data_list)
                else:
                    result["pointclouds_lidar_coords"] = self._collate_pc(data_list)
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

    def _custom_fps(self, point: Tensor, num_points: int) -> Tensor:
        N, _ = point.shape
        xyz = point[:, :3]
        centroids = torch.zeros((num_points,))
        distance = torch.ones((N,)) * 1e10
        farthest = torch.randint(0, N, (1,))
        for i in range(num_points):
            centroids[i] = farthest
            centroid = xyz[farthest, :]
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.argmax(distance, -1)
        point = point[centroids.int()]
        return point

    def _o3d_fps(self, input: Tensor, num_points: int) -> Tensor:
        pc_o3d = o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(input))
        pcd = o3d.t.geometry.PointCloud(pc_o3d)
        pcd = pcd.farthest_point_down_sample(num_points)
        pc = torch.utils.dlpack.from_dlpack(pcd.point.positions.to_dlpack())
        return pc

    def _random_point_sample(self, point: Tensor, num_points: int) -> Tensor:
        N = point.shape[0]
        if N >= num_points:
            sample_idx = torch.randperm(N)[:num_points]
        else:
            sample_idx = torch.cat((torch.arange(N), torch.randint(0, N, (num_points - N,))), dim=0)
        point = point[sample_idx]
        return point

    def _get_soc(self, idx: int) -> Tensor:
        row = self.dataset_df.iloc[idx]
        image_ts = int(row["image"])
        track_dir = self.dataset_root / str(row["track"])

        mask_front_filepath = track_dir / self._masks_dirname / "Cam5" / f"{image_ts}.png"
        mask_front = cv2.imread(str(mask_front_filepath), cv2.IMREAD_UNCHANGED).transpose(1, 0)
        mask_back_filepath = track_dir / self._masks_dirname / "Cam2" / f"{image_ts}.png"
        mask_back = cv2.imread(str(mask_back_filepath), cv2.IMREAD_UNCHANGED).transpose(1, 0)

        pc_filepath = track_dir / self._pointclouds_dirname / f"{row['pointcloud']}.bin"
        lidar_scan = self._load_pc(pc_filepath, torch_tensor=False)

        coords_front, _, in_image_front = self.front_cam_proj(lidar_scan)
        coords_back, _, in_image_back = self.back_cam_proj(lidar_scan)

        point_labels = np.zeros(len(lidar_scan), dtype=np.uint8)
        point_labels[in_image_front] = get_points_labels_by_mask(coords_front, mask_front)
        point_labels[in_image_back] = get_points_labels_by_mask(coords_back, mask_back)

        instances_front = semantic_mask_to_instances(
            mask_front,
            area_threshold=10,
            labels_whitelist=self.special_classes,
        )
        instances_back = semantic_mask_to_instances(
            mask_back,
            area_threshold=10,
            labels_whitelist=self.special_classes,
        )

        objects_front = instance_masks_to_objects(
            instances_front,
            coords_front,
            point_labels[in_image_front],
            lidar_scan[in_image_front],
        )
        objects_back = instance_masks_to_objects(
            instances_back,
            coords_back,
            point_labels[in_image_back],
            lidar_scan[in_image_back],
        )

        objects = {**objects_front, **objects_back}
        packed_objects = pack_objects(objects, self.top_k_soc, self.max_distance_soc, self.special_classes)

        if self.soc_coords_type == "cylindrical_3d":
            packed_objects = np.concatenate(
                (
                    np.linalg.norm(packed_objects, axis=-1, keepdims=True),
                    np.arctan2(packed_objects[..., 1], packed_objects[..., 0])[..., None],
                    packed_objects[..., 2:],
                ),
                axis=-1,
            )
            if self.subset == "train":
                packed_objects = self.augment_coords_with_rotation(
                    packed_objects, angle_range=(-np.pi, np.pi)
                )
                packed_objects = self.augment_coords_with_normal(packed_objects, std=(0.2, 0.2, 0.2))
        elif self.soc_coords_type == "cylindrical_2d":
            packed_objects = np.concatenate(
                (
                    np.linalg.norm(packed_objects[..., :2], axis=-1, keepdims=True),
                    np.arctan2(packed_objects[..., 1], packed_objects[..., 0])[..., None],
                    packed_objects[..., 2:],
                ),
                axis=-1,
            )
        elif self.soc_coords_type == "euclidean":
            pass
        elif self.soc_coords_type == "spherical":
            packed_objects = np.concatenate(
                (
                    np.linalg.norm(packed_objects, axis=-1, keepdims=True),
                    np.arccos(
                        packed_objects[..., 2] / np.linalg.norm(packed_objects, axis=-1, keepdims=True)
                    ),
                    np.arctan2(packed_objects[..., 1], packed_objects[..., 0])[..., None],
                ),
                axis=-1,
            )
        else:
            raise ValueError(f"Unknown soc_coords_type: {self.soc_coords_type!r}")

        objects_tensor = torch.from_numpy(packed_objects).float()

        return objects_tensor

    def augment_coords_with_rotation(
        self, coords: np.ndarray, angle_range: Tuple = (-np.pi, np.pi)
    ) -> np.ndarray:
        """Augment the coordinates with a random rotation - all objects are rotated by the same, random uniformly distributed angle.

        Args:
            coords (np.ndarray): The coordinates to be augmented.
            angle_range (Tuple, optional): The range of the random rotation angle. Defaults to (-np.pi, np.pi).

        Returns:
            np.ndarray: The augmented coordinates.
        """
        # Generate a random angle for rotation within the specified range
        random_angle = np.random.uniform(low=angle_range[0], high=angle_range[1])

        # Add the random angle to the θ coordinate of each triplet
        coords[:, :, 1] = (coords[:, :, 1] + random_angle) % (2 * np.pi)

        # Adjust angles to be in the range (-pi, pi)
        coords[:, :, 1] = (coords[:, :, 1] + np.pi) % (2 * np.pi) - np.pi

        return coords

    def augment_coords_with_normal(
        self,
        coords: np.ndarray,
        mean: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        std: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> np.ndarray:
        """Augment the coordinates with a random normal distribution.

        Args:
            coords (np.ndarray): The coordinates to be augmented.
            mean (Tuple[float, float, float], optional): The mean of the normal distribution. Defaults to (0.0, 0.0, 0.0).
            std (Tuple[float, float, float], optional): The standard deviation of the normal distribution. Defaults to (1.0, 1.0, 1.0).

        Returns:
            np.ndarray: The augmented coordinates.
        """
        # Generate random values from a normal distribution
        N, K = coords.shape[:2]
        for i, (m, s) in enumerate(zip(mean, std)):
            random_deltas = np.random.normal(m, s, size=(N, K, 1))
            coords[:, :, i] += random_deltas[:, :, 0]

        coords[:, :, 0] = np.maximum(coords[:, :, 0], 0)
        coords[:, :, 1] = (coords[:, :, 1] + np.pi) % (2 * np.pi) - np.pi

        return coords
