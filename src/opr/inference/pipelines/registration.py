"""Registration pipelines for inference.

This module defines interfaces and lightweight implementations for estimating
rigid 4×4 transformations between two point clouds. It is backend-agnostic:
methods can be geometric (e.g., feature matching) or learning-based. The module
establishes clear transform-direction semantics for safe composition with world
poses.

Direction and composition:
- Inputs: `query_pc` (source), `db_pc` (target)
- Output: `T_db<-q` such that `x_db = T_db<-q * x_q` (homogeneous 4×4, column
  vectors)

Example: given a known database world pose `T_w<-db` and an estimated
`T_db<-q`, the query world pose is `T_w<-q = T_w<-db * T_db<-q`.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import open3d as o3d
from torch import Tensor


class RansacPointCloudRegistrationPipeline:
    """Point cloud registration pipeline using Open3D RANSAC.

    The pipeline performs the following steps:
      1) Voxel downsample both clouds and estimate normals
      2) Compute FPFH features
      3) Run RANSAC-based global registration

    Returned transform semantics:
    - We feed Open3D with `source=query`, `target=database`.
    - Open3D returns a transform mapping `source→target`, i.e. `T_db<-q`.
    This can be composed with a known database world pose `T_w<-db` to get
    `T_w<-q = T_w<-db * T_db<-q`.
    """

    def __init__(self, voxel_downsample_size: float = 0.5) -> None:
        """Initialize the RANSAC registration pipeline.

        Args:
            voxel_downsample_size (float): Voxel size used for downsampling and
                for computing normal/feature radii. Larger values are faster but
                may reduce accuracy. Defaults to 0.5.
        """
        self.voxel_downsample_size = voxel_downsample_size

    def _preprocess_point_cloud(
        self, points: Tensor
    ) -> Tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
        """Downsample point cloud and compute FPFH features.

        Args:
            points: Tensor [N,3] float32.

        Returns:
            Tuple of (downsampled Open3D point cloud, FPFH feature).
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
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
        """Run Open3D RANSAC-based registration."""
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
        """Estimate rigid transform that maps query into the database frame.

        Args:
            query_pc: Tensor [N,3] float32 for the query cloud.
            db_pc: Tensor [M,3] float32 for the database cloud.

        Returns:
            np.ndarray: 4×4 transformation matrix (float64) `T_db<-q` such that
            `x_db = T_db<-q * x_q`.

        Notes:
            To obtain the world pose of the query from a known `T_w<-db`, use
            `T_w<-q = T_w<-db * T_db<-q`.
        """
        source_down, source_fpfh = self._preprocess_point_cloud(query_pc)
        target_down, target_fpfh = self._preprocess_point_cloud(db_pc)
        result = self._execute_global_registration(source_down, target_down, source_fpfh, target_fpfh)
        return result.transformation
