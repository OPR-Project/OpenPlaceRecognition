"""NCLT dataset implementation."""
from pathlib import Path
from time import time
from typing import Any, Literal

import pandas as pd
import cv2
import numpy as np
import open3d as o3d
import torch
from loguru import logger
from torch import Tensor
from torch.utils.data import Dataset
from omegaconf import OmegaConf
import albumentations as A  # noqa: N812
from albumentations.pytorch import ToTensorV2

from opr.utils import cartesian_to_spherical

try:
    import MinkowskiEngine as ME  # type: ignore
    minkowski_available = True
except ImportError:
    logger.warning("MinkowskiEngine is not installed. Some features may not be available.")
    minkowski_available = False


class NCLTDatasetV2(Dataset):
    """NCLT dataset implementation version 2, refined."""

    _valid_data: tuple[str, ...] = (
        "image_Cam0",
        "image_Cam1",
        "image_Cam2",
        "image_Cam3",
        "image_Cam4",
        "image_Cam5",
        "pointcloud_lidar",
    )

    def __init__(
        self,
        dataset_root: str | Path,
        subset: Literal["train", "val", "test"],
        data_to_load: str | tuple[str, ...],
        positive_threshold: float = 10.0,
        negative_threshold: float = 50.0,
        images_dirname: str = "images",
        pointclouds_dirname: str = "velodyne_data",
        sequence_len: int = 1,
        max_interframe_distance: float = 2.0,
        window_step: int = 1,
        use_minkowski: bool = False,
        pointcloud_quantization_size: float | tuple[float, float, float] | None = None,
        max_point_distance: float | None = None,
        normalize_point_cloud: bool = False,
        num_points_sample: int | None = None,
        sampling_strategy: Literal["fps_custom", "fps_o3d", "random"] | None = None,
        spherical_coords: bool = False,
        use_intensity_values: bool = True,
        image_transform: A.Compose | None = None,
        pointcloud_transform: Any | None = None,
    ) -> None:
        self._dataset_root = Path(dataset_root)
        if not self._dataset_root.exists():
            raise FileNotFoundError(f"Dataset root directory {self._dataset_root} does not exist.")

        self._subset = subset
        csv_filepath = self._dataset_root / f"{subset}.csv"
        if not csv_filepath.exists():
            raise FileNotFoundError(f"CSV file {csv_filepath} does not exist.")

        self._dataset_df = self._read_dataset_df(csv_filepath)

        self._data_to_load = (data_to_load,) if isinstance(data_to_load, str) else data_to_load
        if not set(self._data_to_load).issubset(self._valid_data):
            raise ValueError(f"Invalid data_to_load: {self._data_to_load}")

        self._positive_threshold = positive_threshold
        self._negative_threshold = negative_threshold

        self._images_dirname = images_dirname
        self._pointclouds_dirname = pointclouds_dirname

        self._sequence_len = sequence_len
        if not isinstance(self._sequence_len, int) or self._sequence_len <= 0:
            raise ValueError(f"sequence_len must be a positive integer, got {self._sequence_len}.")

        self._max_interframe_distance = max_interframe_distance
        if self._max_interframe_distance <= 0:
            raise ValueError("max_interframe_distance must be positive.")

        self._window_step = window_step
        if self._window_step <= 0:
            raise ValueError("window_step must be a positive integer.")

        self._max_point_distance = max_point_distance
        self._normalize_point_cloud = normalize_point_cloud
        if self._normalize_point_cloud:
            if self._max_point_distance is None or self._max_point_distance <= 0:
                raise ValueError("max_point_distance must be provided and positive when normalize_point_cloud is True.")
        self._num_points_sample = num_points_sample
        if self._num_points_sample is not None and self._num_points_sample > 0:
            if sampling_strategy is None:
                raise ValueError("sampling_strategy must be set when num_points_sample is provided.")
            elif sampling_strategy not in {"fps_custom", "fps_o3d", "random"}:
                raise ValueError(f"Invalid sampling_strategy: {sampling_strategy}.")
        self._sampling_strategy = sampling_strategy

        self._spherical_coords = spherical_coords
        self._use_intensity_values = use_intensity_values

        self._use_minkowski = use_minkowski
        if self._use_minkowski and not minkowski_available:
            raise ValueError("MinkowskiEngine is not installed. Set use_minkowski=False.")
        self._pointcloud_quantization_size = pointcloud_quantization_size
        if self._pointcloud_quantization_size is not None and not self._use_minkowski:
            logger.warning(
                "Pointcloud quantization is set, but MinkowskiEngine is not used. "
                "Quantization will be ignored."
            )

        self._image_transform = image_transform or A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

        self._pointcloud_transform = pointcloud_transform

        # Precompute sequence indices if sequence_len > 1.
        if self._sequence_len > 1:
            self._sequence_indices = self._build_sequence_indices()
        else:
            self._sequence_indices = None

        # Initialize masks and indexes.
        self._positives_mask, self._negatives_mask = self._build_masks(self._positive_threshold, self._negative_threshold)
        self._positives_index, self._nonnegative_index = self._build_indexes(self._positive_threshold, self._negative_threshold)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        if self._sequence_len == 1:
            return self._get_single_frame(idx)
        else:
            return self._get_sequence(idx)

    def __len__(self) -> int:
        if self._sequence_len == 1:
            return len(self._dataset_df)
        else:
            return len(self._sequence_indices)

    def _build_sequence_indices(self) -> list[list[int]]:
        """Build valid sequence indices using a sliding window.

        A valid sequence has:
          - All rows with the same value in the "track" column.
          - For each consecutive pair, the Euclidean distance (from "tx", "ty", "tz")
            is no more than self._max_interframe_distance.
        The window moves with a step of self._window_step.
        """
        sequences = []
        df = self._dataset_df
        n = len(df)
        for i in range(0, n - self._sequence_len + 1, self._window_step):
            window = df.iloc[i : i + self._sequence_len]
            if window["track"].nunique() != 1:
                continue
            valid = True
            for j in range(len(window) - 1):
                pt1 = window.iloc[j][["tx", "ty", "tz"]].values.astype(float)
                pt2 = window.iloc[j + 1][["tx", "ty", "tz"]].values.astype(float)
                if np.linalg.norm(pt2 - pt1) > self._max_interframe_distance:
                    valid = False
                    break
            if valid:
                sequences.append(list(window.index))
        return sequences

    def _get_single_frame(self, idx: int) -> dict[str, Tensor]:
        """Return a dictionary with data for a single frame at index idx."""
        row = self._dataset_df.iloc[idx]
        data = {"idx": torch.tensor(idx, dtype=torch.int)}
        data["utm"] = torch.tensor(row[["tx", "ty"]].to_numpy(dtype=np.float32))
        track_dir = self._dataset_root / str(row["track"]) if "track" in row else self._dataset_root

        for data_source in self._data_to_load:
            if data_source.startswith("image_"):
                cam_name = data_source[6:]  # Remove "image_" prefix.
                image_ts = int(row["frame_timestamp"])
                im_filepath = track_dir / self._images_dirname / f"{cam_name}" / f"{image_ts}.png"
                data[data_source] = self._load_image(im_filepath)
            elif data_source == "pointcloud_lidar":
                pc_filepath = track_dir / self._pointclouds_dirname / f"{row['frame_timestamp']}.bin"
                pointcloud = self._load_pc(pc_filepath)
                if self._pointcloud_transform is not None:
                    data[f"{data_source}_coords"] = self._pointcloud_transform(pointcloud[:, :3])
                else:
                    data[f"{data_source}_coords"] = pointcloud[:, :3]
                if self._use_intensity_values:
                    data[f"{data_source}_feats"] = pointcloud[:, 3].unsqueeze(1)
                else:
                    data[f"{data_source}_feats"] = torch.ones_like(pointcloud[:, :1])
        return data

    def _get_sequence(self, idx: int) -> dict[str, Tensor]:
        """Return a dictionary with data for a sequence of frames.

        For all keys except "utm", stack the values from each frame.
        For the "utm" key, return only the last value in the sequence.
        """
        seq_indices = self._sequence_indices[idx]
        frames = [self._get_single_frame(i) for i in seq_indices]
        data = {}
        for key in frames[0]:
            if key == "utm":
                data[key] = frames[-1]["utm"]
            else:
                data[key] = torch.stack([frame[key] for frame in frames])
        return data

    def _read_dataset_df(self, csv_filepath: str | Path) -> pd.DataFrame:
        expected_columns = {
            "timestamp": np.int64,
            "frame_timestamp": np.int64,
            "tx": np.float32,
            "ty": np.float32,
            "tz": np.float32,
            "qw": np.float32,
            "qx": np.float32,
            "qy": np.float32,
            "qz": np.float32,
        }
        df = pd.read_csv(csv_filepath, dtype=expected_columns)
        if not set(expected_columns.keys()).issubset(df.columns):
            raise ValueError(f"CSV file {csv_filepath} does not contain the required columns.")
        return df

    def _load_pc(self, filepath: str | Path) -> Tensor:
        pc = np.fromfile(str(filepath), dtype=np.float32).reshape(-1, 4)
        if not self._use_intensity_values:
            pc = pc[:, :3]
        if self._max_point_distance is not None:
            distances = np.linalg.norm(pc[:, :3], axis=1)
            pc = pc[distances <= self._max_point_distance]
            if self._normalize_point_cloud:
                pc[:, :3] = pc[:, :3] / self._max_point_distance
        pc_tensor = torch.tensor(pc, dtype=torch.float)
        if self._num_points_sample is not None and self._num_points_sample > 0:
            if self._sampling_strategy == "fps_custom":
                pc_tensor = self._fps_custom(pc_tensor, self._num_points_sample)
            elif self._sampling_strategy == "fps_o3d":
                pc_tensor = self._fps_o3d(pc_tensor, self._num_points_sample)
            elif self._sampling_strategy == "random":
                pc_tensor = self._random_point_sample(pc_tensor, self._num_points_sample)
            else:
                raise ValueError(f"Unsupported sampling strategy: {self._sampling_strategy}")
        return pc_tensor

    def _load_image(self, filepath: str | Path) -> Tensor:
        img = cv2.imread(str(filepath))
        if img is None:
            raise FileNotFoundError(f"Image file {filepath} could not be loaded.")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transformed = self._image_transform(image=img)
        return transformed["image"]

    def _collate_pc_minkowski(self, data_list: list[dict[str, Tensor]]) -> tuple[Tensor, Tensor]:
        raise NotImplementedError("MinkowskiEngine collate is not implemented yet.")

    def _collate_pc(self, data_list: list[dict[str, Tensor]]) -> Tensor:
        coords_list = [e["pointcloud_lidar_coords"] for e in data_list]
        feats_list = [e["pointcloud_lidar_feats"] for e in data_list]
        return torch.stack(coords_list), torch.stack(feats_list)

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
            elif data_key == "pointcloud_lidar_coords":
                if self._use_minkowski:
                    (
                        result["pointclouds_lidar_coords"],
                        result["pointclouds_lidar_feats"],
                    ) = self._collate_pc_minkowski(data_list)
                else:
                    (
                        result["pointclouds_lidar_coords"],
                        result["pointclouds_lidar_feats"],
                    ) = self._collate_pc(data_list)
            elif data_key == "pointcloud_lidar_feats":
                continue
            else:
                raise ValueError(f"Unknown data key: {data_key!r}")
        return result

    def _build_masks(self, positive_threshold: float, negative_threshold: float) -> tuple[Tensor, Tensor]:
        if self._sequence_len == 1:
            utm = torch.tensor(self._dataset_df[["tx", "ty"]].to_numpy(dtype=np.float32), dtype=torch.float32)
        else:
            seq_utm = [self._dataset_df.iloc[seq[-1]][["tx", "ty"]].to_numpy(dtype=np.float32) for seq in self._sequence_indices]
            utm = torch.tensor(seq_utm, dtype=torch.float32)
        distances = torch.cdist(utm, utm)
        positives_mask = (distances > 0) & (distances < positive_threshold)
        negatives_mask = distances > negative_threshold
        return positives_mask, negatives_mask

    def _build_indexes(self, positive_threshold: float, negative_threshold: float) -> tuple[list[Tensor], list[Tensor]]:
        if self._sequence_len == 1:
            utm = torch.tensor(self._dataset_df[["tx", "ty"]].to_numpy(dtype=np.float32), dtype=torch.float32)
        else:
            seq_utm = [self._dataset_df.iloc[seq[-1]][["tx", "ty"]].to_numpy(dtype=np.float32) for seq in self._sequence_indices]
            utm = torch.tensor(seq_utm, dtype=torch.float32)
        distances = torch.cdist(utm, utm)
        positives_mask = (distances > 0) & (distances < positive_threshold)
        nonnegatives_mask = distances < negative_threshold
        positive_indices = [torch.nonzero(row).squeeze(dim=-1) for row in positives_mask]
        nonnegative_indices = [torch.nonzero(row).squeeze(dim=-1) for row in nonnegatives_mask]
        return positive_indices, nonnegative_indices

    @property
    def positives_index(self) -> list[Tensor]:
        """List of indexes of positive samples for each element in the dataset."""
        return self._positives_index

    @property
    def nonnegative_index(self) -> list[Tensor]:
        """List of indexes of non-negative samples for each element in the dataset."""
        return self._nonnegative_index

    @property
    def positives_mask(self) -> Tensor:
        """Boolean mask of positive samples for each element in the dataset."""
        return self._positives_mask

    @property
    def negatives_mask(self) -> Tensor:
        """Boolean mask of negative samples for each element in the dataset."""
        return self._negatives_mask

    def collate_fn(self, data_list: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        """Pack input data list into batch.

        Args:
            data_list (List[Dict[str, Tensor]]): batch data list generated by DataLoader.

        Returns:
            Dict[str, Tensor]: dictionary of batched data.
        """
        return self._collate_data_dict(data_list)

    def _fps_custom(self, points: Tensor, num_points: int) -> Tensor:
        N, _ = points.shape
        xyz = points[:, :3]
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
        return points[centroids.int()]

    def _fps_o3d(self, points: Tensor, num_points: int) -> Tensor:
        # TODO: make it work correctly with intensity column
        pc_o3d = o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(points))
        pcd = o3d.t.geometry.PointCloud(pc_o3d)
        pcd = pcd.farthest_point_down_sample(num_points)
        return torch.utils.dlpack.from_dlpack(pcd.point.positions.to_dlpack())

    def _random_point_sample(self, points: Tensor, num_points: int) -> Tensor:
        N = points.shape[0]
        if N >= num_points:
            sample_idx = torch.randperm(N)[:num_points]
        else:
            sample_idx = torch.cat((torch.arange(N), torch.randint(0, N, (num_points - N,))), dim=0)
        return points[sample_idx]
