"""Pointcloud registration pipeline."""
from os import PathLike
from typing import Optional, Union

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
            pc (Tensor): Pointcloud.

        Returns:
            Tensor: Downsampled pointcloud.
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc.cpu().numpy())
        pcd = pcd.voxel_down_sample(self.voxel_downsample_size)
        pc = torch.from_numpy(np.array(pcd.points).astype(np.float32)).float()
        return pc

    def infer(self, query_pc: Tensor, db_pc: Tensor) -> np.ndarray:
        """Infer the transformation between the query and the database pointclouds.

        Args:
            query_pc (Tensor): Query pointcloud.
            db_pc (Tensor): Database pointcloud.

        Returns:
            np.ndarray: Transformation matrix.
        """
        query_pc = self._downsample_pointcloud(query_pc)
        db_pc = self._downsample_pointcloud(db_pc)
        with torch.no_grad():
            transform = self.model(query_pc, db_pc)["estimated_transform"]
        return transform.cpu().numpy()
