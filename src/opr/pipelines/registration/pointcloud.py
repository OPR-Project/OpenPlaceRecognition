"""Pointcloud registration pipeline."""
from os import PathLike
from typing import Optional, Tuple, Union

import numpy as np
import open3d as o3d
import torch
from torch import Tensor, nn

from opr.utils import init_model, parse_device


class PointcloudRegistrationPipeline:
    """Pointcloud registration pipeline."""

    def __init__(
        self,
        model: nn.Module,
        model_weights_path: Optional[Union[str, PathLike]] = None,
        device: Union[str, int, torch.device] = "cuda",
        voxel_downsample_size: Optional[float] = 0.3,
    ) -> None:
        """Pointcloud registration pipeline.

        Args:
            model (nn.Module): Model.
            model_weights_path (Union[str, PathLike], optional): Path to the model weights.
                If None, the weights are not loaded. Defaults to None.
            device (Union[str, int, torch.device]): Device to use. Defaults to "cuda".
            voxel_downsample_size (Optional[float]): Voxel downsample size. Defaults to 0.3.
        """
        self.device = parse_device(device)
        self.model = init_model(model, model_weights_path, self.device)
        self.voxel_downsample_size = voxel_downsample_size

    def _downsample_pointcloud(self, pc: Tensor) -> Tensor:
        """Downsample the pointcloud.

        Args:
            pc (Tensor): Pointcloud. Coordinates array of shape (N, 3).

        Returns:
            Tensor: Downsampled pointcloud. Coordinates array of shape (M, 3), where M <= N.
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc.cpu().numpy())
        pcd = pcd.voxel_down_sample(self.voxel_downsample_size)
        pc = torch.from_numpy(np.array(pcd.points).astype(np.float32)).float()
        return pc

    def infer(self, query_pc: Tensor, db_pc: Tensor) -> np.ndarray:
        """Infer the transformation between the query and the database pointclouds.

        Args:
            query_pc (Tensor): Query pointcloud. Coordinates array of shape (N, 3).
            db_pc (Tensor): Database pointcloud. Coordinates array of shape (M, 3).

        Returns:
            np.ndarray: Transformation matrix.
        """
        query_pc = self._downsample_pointcloud(query_pc)
        db_pc = self._downsample_pointcloud(db_pc)
        with torch.no_grad():
            transform = self.model(query_pc, db_pc)["estimated_transform"]
        return transform.cpu().numpy()


class RansacGlobalRegistrationPipeline:
    """Pointcloud registration pipeline using RANSAC."""

    def __init__(self, voxel_downsample_size: float = 0.5) -> None:
        """Pointcloud registration pipeline using RANSAC.

        Args:
            voxel_downsample_size (float): Voxel downsample size. Defaults to 0.5.
        """
        self.voxel_downsample_size = voxel_downsample_size

    def _preprocess_point_cloud(
        self, points: Tensor
    ) -> Tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd_down = pcd.voxel_down_sample(self.voxel_downsample_size)
        radius_normal = self.voxel_downsample_size * 2
        pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        radius_feature = self.voxel_downsample_size * 5
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
        )
        return pcd_down, pcd_fpfh

    def _execute_global_registration(
        self,
        source_down: o3d.geometry.PointCloud,
        target_down: o3d.geometry.PointCloud,
        source_fpfh: o3d.pipelines.registration.Feature,
        target_fpfh: o3d.pipelines.registration.Feature,
    ) -> o3d.pipelines.registration.RegistrationResult:
        distance_threshold = self.voxel_downsample_size * 1.5
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down,
            target_down,
            source_fpfh,
            target_fpfh,
            True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3,
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
        )
        return result

    def infer(self, query_pc: Tensor, db_pc: Tensor) -> np.ndarray:
        """Infer the transformation between the query and the database pointclouds.

        Args:
            query_pc (Tensor): Query pointcloud. Coordinates array of shape (N, 3).
            db_pc (Tensor): Database pointcloud. Coordinates array of shape (M, 3).

        Returns:
            np.ndarray: Transformation matrix.
        """
        query_pc = query_pc.cpu().numpy()
        db_pc = db_pc.cpu().numpy()
        source_down, source_fpfh = self._preprocess_point_cloud(query_pc)
        target_down, target_fpfh = self._preprocess_point_cloud(db_pc)
        result = self._execute_global_registration(source_down, target_down, source_fpfh, target_fpfh)
        return result.transformation
