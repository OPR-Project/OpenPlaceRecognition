"""Hierarchical Localization Pipeline."""
from os import PathLike
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from geotransformer.utils.pointcloud import (
    get_rotation_translation_from_transform,
    get_transform_from_rotation_translation,
)
from loguru import logger
from scipy.spatial.transform import Rotation
from torch import Tensor
from tqdm import tqdm

from opr.pipelines.place_recognition import PlaceRecognitionPipeline
from opr.pipelines.registration import PointcloudRegistrationPipeline


class LocalizationPipeline:
    """Hierarchical Localization Pipeline.

    The task of localiation is solved in two steps:
    1. Find the best match for the query in the database (Place Recognition).
    2. Refine the pose estimate using the query and the database match (Registration).
    """

    def __init__(
        self,
        place_recognition_pipeline: PlaceRecognitionPipeline,
        registration_pipeline: PointcloudRegistrationPipeline,
        precomputed_reg_feats: bool = False,
        pointclouds_subdir: str | PathLike | None = None,
    ) -> None:
        """Hierarchical Localization Pipeline.

        The task of localiation is solved in two steps:
        1. Find the best match for the query in the database (Place Recognition).
        2. Refine the pose estimate using the query and the database match (Registration).

        Args:
            place_recognition_pipeline (PlaceRecognitionPipeline): Place Recognition pipeline.
            registration_pipeline (PointcloudRegistrationPipeline): Registration pipeline.
            precomputed_reg_feats (bool): Whether to use precomputed registration features. Defaults to False.
            pointclouds_subdir (str | PathLike, optional): Sub-directory with pointclouds. Will be used
                for computing registration feats, if they are not exist; or for loading pointclouds
                if `precomputed_reg_feats=False`. Defaults to None.

        Raises:
            ValueError: Pointclouds sub-directory must be provided if precomputed registration
                features are not used.
            ValueError: Precomputed registration features are only supported for HRegNet.
        """
        self.pr_pipe = place_recognition_pipeline
        self.reg_pipe = registration_pipeline
        self.database_df = self.pr_pipe.database_df
        self.database_dir = self.pr_pipe.database_dir

        if not precomputed_reg_feats and pointclouds_subdir is None:
            raise ValueError(
                "Pointclouds sub-directory must be provided if precomputed registration features are not used."
            )

        self.pointclouds_dir = (self.database_dir / pointclouds_subdir) if pointclouds_subdir else None
        if self.pointclouds_dir is not None and not self.pointclouds_dir.exists():
            raise ValueError(f"Pointclouds directory not found: {self.pointclouds_dir}")

        self.precomputed_reg_feats = precomputed_reg_feats
        self.precomputed_reg_feats_dir = None
        reg_model_name = registration_pipeline.model.__class__.__name__
        if self.precomputed_reg_feats:
            if reg_model_name != "HRegNet":
                raise ValueError("Precomputed registration features are only supported for HRegNet.")
            self.precomputed_reg_feats_dir = self.database_dir / f"{reg_model_name}_features"
            if not self.precomputed_reg_feats_dir.exists():
                logger.warning(
                    f"Precomputed registration features directory not found: {self.precomputed_reg_feats_dir}. "
                    "It will be created and features will be computed."
                )
                self.precomputed_reg_feats_dir.mkdir()
                if self.pointclouds_dir is None or not Path(self.pointclouds_dir).exists():
                    raise ValueError(
                        "Pointclouds directory must be provided to compute registration features."
                    )
            if not any(self.precomputed_reg_feats_dir.iterdir()):
                logger.warning("Precomputed registration features directory is empty. Computing features.")
                self.compute_reg_features(
                    save_dir=self.precomputed_reg_feats_dir, pointclouds_dir=self.pointclouds_dir
                )

    def infer(self, input_data: Dict[str, Tensor]) -> Dict[str, np.ndarray]:
        """Single sample inference.

        Args:
            input_data (Dict[str, Tensor]): Input data. Dictionary with keys in the following format:

                "image_{camera_name}" for images from cameras,

                "mask_{camera_name}" for semantic segmentation masks,

                "pointcloud_lidar_coords" for pointcloud coordinates from lidar,

                "pointcloud_lidar_feats" for pointcloud features from lidar.

        Returns:
            Dict[str, np.ndarray]: Inference results. Dictionary with keys:

                "db_match_pose" for database match pose in the format [tx, ty, tz, qx, qy, qz, qw],

                "estimated_pose" for estimated pose in the format [tx, ty, tz, qx, qy, qz, qw].
        """
        out_dict = {}

        pr_output = self.pr_pipe.infer(input_data)
        query_pc = input_data["pointcloud_lidar_coords"]
        db_pose = pr_output["pose"]
        out_dict["db_match_pose"] = db_pose
        db_pose = get_transform_from_rotation_translation(
            Rotation.from_quat(db_pose[3:]).as_matrix(), db_pose[:3]
        )

        db_idx = pr_output["idx"]
        if not self.precomputed_reg_feats:
            db_pc_filename = f"{int(self.database_df.iloc[db_idx]['pointcloud'])}.bin"
            db_pc = self._load_pc(self.pointclouds_dir / db_pc_filename)
            estimated_transform = self.reg_pipe.infer(query_pc=query_pc, db_pc=db_pc)
        else:
            db_pc_feats = self._load_feats(
                self.precomputed_reg_feats_dir / f"{int(self.database_df.iloc[db_idx]['pointcloud'])}.pt"
            )
            estimated_transform = self.reg_pipe.infer(query_pc=query_pc, db_pc_feats=db_pc_feats)
        estimated_pose = db_pose @ self._invert_rigid_transformation_matrix(estimated_transform)
        rot, trans = get_rotation_translation_from_transform(estimated_pose)
        rot = Rotation.from_matrix(rot).as_quat()
        pose = np.concatenate([trans, rot])
        out_dict["estimated_pose"] = pose

        return out_dict

    def compute_reg_features(self, save_dir: str | PathLike, pointclouds_dir: str | PathLike) -> None:
        """Compute registration features for the database.

        Args:
            save_dir (str | PathLike): Directory to save the features.
            pointclouds_dir (str | PathLike): Directory where pointclouds will be saved.
        """
        for _, row in tqdm(self.database_df.iterrows(), total=len(self.database_df), leave=False):
            pc_ts = int(row["pointcloud"])
            pointcloud_path = Path(pointclouds_dir) / f"{pc_ts}.bin"
            pointcloud = self._load_pc(pointcloud_path)
            pointcloud = pointcloud.to(self.reg_pipe.device)
            pointcloud = pointcloud[:, :3]
            pointcloud = self.reg_pipe._downsample_pointcloud(pointcloud)
            features = self.reg_pipe.model.extract_features(pointcloud)
            feature_save_path = Path(save_dir) / f"{pc_ts}.pt"
            torch.save(features, feature_save_path)

    def _invert_rigid_transformation_matrix(self, T: np.ndarray) -> np.ndarray:
        """Inverts a 4x4 rigid body transformation matrix.

        Args:
            T (np.ndarray): A 4x4 rigid body transformation matrix.

        Returns:
            np.ndarray: The inverted 4x4 rigid body transformation matrix.

        Raises:
            ValueError: Input matrix must be 4x4.
        """
        if T.shape != (4, 4):
            raise ValueError("Input matrix must be 4x4.")

        R = T[:3, :3]
        t = T[:3, 3]

        R_inv = R.T
        t_inv = -R.T @ t

        T_inv = np.eye(4)
        T_inv[:3, :3] = R_inv
        T_inv[:3, 3] = t_inv

        return T_inv

    def _load_pc(
        self, filepath: str | PathLike, num_point_properties: int = 3, dtype: np.dtype = np.float32
    ) -> Tensor:
        """Load pointcloud from file.

        Args:
            filepath (str | PathLike): Filepath to the pointcloud bin file.
            num_point_properties (int): Number of properties for each point in the point cloud. Defaults to 3.
            dtype (np.dtype): Data type of the point cloud. Defaults to np.float32.

        Returns:
            Tensor: Pointcloud.
        """
        pc = np.fromfile(str(filepath), dtype=dtype).reshape(-1, num_point_properties)
        pc = torch.from_numpy(pc)
        return pc

    def _load_feats(self, filepath: str | PathLike) -> Dict[str, Tensor]:
        """Load features from file.

        Args:
            filepath (str | PathLike): Filepath to the features file.

        Returns:
            Dict[str, Tensor]: Features.
        """
        feats = torch.load(filepath, map_location=self.reg_pipe.device, weights_only=True)
        return feats
