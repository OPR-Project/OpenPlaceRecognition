"""Pointcloud registration pipeline."""
from os import PathLike
from time import time
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
        num_points_downsample: Optional[int] = None,
    ) -> None:
        """Pointcloud registration pipeline.

        Args:
            model (nn.Module): Model.
            model_weights_path (Union[str, PathLike], optional): Path to the model weights.
                If None, the weights are not loaded. Defaults to None.
            device (Union[str, int, torch.device]): Device to use. Defaults to "cuda".
            voxel_downsample_size (Optional[float]): Voxel downsample size. Defaults to 0.3.
            num_points_downsample (int, optional): Number number of points to keep. If num_points is bigger
                than the number of points in the pointcloud, the points are sampled with replacement.
                Defaults to None, which keeps all points.
        """
        self.device = parse_device(device)
        self.model = init_model(model, model_weights_path, self.device)
        self.voxel_downsample_size = voxel_downsample_size
        self.num_points_downsample = num_points_downsample
        self.stats_history = {"inference_time": [], "downsample_time": [], "total_time": []}

    def _downsample_pointcloud(self, pc: Tensor) -> Tensor:
        """Downsample the pointcloud.

        Args:
            pc (Tensor): Pointcloud. Coordinates array of shape (N, 3).

        Returns:
            Tensor: Downsampled pointcloud. Coordinates array of shape (M, 3), where M <= N.
        """
        pc_o3d = o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(pc))
        pcd = o3d.t.geometry.PointCloud(pc_o3d)
        pcd = pcd.voxel_down_sample(self.voxel_downsample_size)
        pc = torch.utils.dlpack.from_dlpack(pcd.point.positions.to_dlpack())
        if self.num_points_downsample:
            N = pc.shape[0]
            if N >= self.num_points_downsample:
                sample_idx = torch.randperm(N)[: self.num_points_downsample]
            else:
                sample_idx = torch.cat(
                    (torch.arange(N), torch.randint(0, N, (self.num_points_downsample - N,))), dim=0
                )
            pc = pc[sample_idx]
        return pc

    def infer(
        self, query_pc: Tensor, db_pc: Tensor | None = None, db_pc_feats: dict[str, Tensor] | None = None
    ) -> np.ndarray:
        """Infer the transformation between the query and the database pointclouds.

        Args:
            query_pc (Tensor): Query pointcloud. Coordinates array of shape (N, 3).
            db_pc (Tensor, optional): Database pointcloud. Coordinates array of shape (M, 3).
                If None, `db_pc_feats` must be provided. Defaults to None.
            db_pc_feats (dict[str, Tensor], optional): Database pointcloud features.
                If None, `db_pc` must be provided. Defaults to None.

        Returns:
            np.ndarray: Transformation matrix.

        Raises:
            ValueError: If both `db_pc` and `db_pc_feats` are provided or if none of them are provided.
        """
        if db_pc is None and db_pc_feats is None:
            raise ValueError("Either `db_pc` or `db_pc_feats` must be provided.")
        if db_pc is not None and db_pc_feats is not None:
            raise ValueError("Only one of `db_pc` or `db_pc_feats` must be provided.")

        start_time = time()

        query_pc = query_pc.to(self.device)
        query_pc = self._downsample_pointcloud(query_pc)

        if db_pc is not None:
            db_pc = db_pc.to(self.device)
            db_pc = self._downsample_pointcloud(db_pc)
        else:
            db_pc_feats = {k: v.to(self.device) for k, v in db_pc_feats.items()}

        self.stats_history["downsample_time"].append(time() - start_time)

        start_time = time()
        with torch.no_grad():
            if db_pc is not None:
                transform = self.model(query_pc=query_pc, db_pc=db_pc)["estimated_transform"]
            else:
                transform = self.model(query_pc=query_pc, db_pc_feats=db_pc_feats)["estimated_transform"]
        self.stats_history["inference_time"].append(time() - start_time)
        self.stats_history["total_time"].append(
            self.stats_history["downsample_time"][-1] + self.stats_history["inference_time"][-1]
        )
        return transform.cpu().numpy()


class SequencePointcloudRegistrationPipeline(PointcloudRegistrationPipeline):
    """Pointcloud registration pipeline that supports sequences."""

    def __init__(
        self,
        model: nn.Module,
        model_weights_path: Optional[Union[str, PathLike]] = None,
        device: Union[str, int, torch.device] = "cuda",
        voxel_downsample_size: Optional[float] = 0.3,
        num_points_downsample: Optional[int] = None,
    ) -> None:
        """Pointcloud registration pipeline that supports sequences.

        Args:
            model (nn.Module): Model.
            model_weights_path (Union[str, PathLike], optional): Path to the model weights.
                If None, the weights are not loaded. Defaults to None.
            device (Union[str, int, torch.device]): Device to use. Defaults to "cuda".
            voxel_downsample_size (Optional[float]): Voxel downsample size. Defaults to 0.3.
            num_points_downsample (int, optional): Number number of points to keep. If num_points is bigger
                than the number of points in the pointcloud, the points are sampled with replacement.
                Defaults to None, which keeps all points.
        """
        super().__init__(model, model_weights_path, device, voxel_downsample_size, num_points_downsample)
        self.ransac_pipeline = RansacGlobalRegistrationPipeline(
            voxel_downsample_size=0.5  # handcrafted optimal value for fast inference
        )

    def _transform_points(self, points: Tensor, transform: Tensor) -> Tensor:
        points_hom = torch.cat((points, torch.ones((points.shape[0], 1), device=points.device)), dim=1)
        # print(type(points_hom), type(transform))
        # print(points_hom.dtype, transform.dtype)
        points_transformed_hom = points_hom @ transform
        points_transformed = points_transformed_hom[:, :3] / points_transformed_hom[:, 3].unsqueeze(-1)
        return points_transformed

    def infer(
        self,
        query_pc_list: list[Tensor],
        db_pc: Tensor | None = None,
        db_pc_feats: dict[str, Tensor] | None = None,
    ) -> np.ndarray:
        """Infer the transformation between the query and the database pointclouds.

        Args:
            query_pc_list (list[Tensor]): List of query pointclouds. Each pointcloud
                is a coordinates array of shape (N, 3).
            db_pc (Tensor, optional): Database pointcloud. Coordinates array of shape (M, 3).
                If None, `db_pc_feats` must be provided. Defaults to None.
            db_pc_feats (dict[str, Tensor], optional): Database pointcloud features.
                If None, `db_pc` must be provided. Defaults to None.

        Returns:
            np.ndarray: Transformation matrix.
        """
        if len(query_pc_list) > 1:
            accumulated_query_pc = query_pc_list[-1]
            for pc in query_pc_list[-2::-1]:
                transform = torch.tensor(
                    self.ransac_pipeline.infer(accumulated_query_pc, pc), dtype=torch.float32
                )
                accumulated_query_pc = torch.cat(
                    [accumulated_query_pc, self._transform_points(pc, transform)], dim=0
                )
        else:
            accumulated_query_pc = query_pc_list[0]
        return super().infer(query_pc=accumulated_query_pc, db_pc=db_pc, db_pc_feats=db_pc_feats)


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
