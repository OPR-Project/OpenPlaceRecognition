"""Projection of pointcloud to camera image plane."""

from typing import Optional, Tuple, Union

import numpy as np
import quaternion
from loguru import logger
from omegaconf import OmegaConf
from quaternion import as_rotation_matrix


class Projector:
    """Class for projecting pointcloud to camera image plane."""

    def __init__(self, cam_cfg: OmegaConf, lidar_cfg: OmegaConf) -> None:
        """Initialize projector.

        Args:
            cam_cfg (OmegaConf): camera configuration
            lidar_cfg (OmegaConf): lidar configuration
        """
        self.proj_matrix = np.array(cam_cfg.left.rect.P)
        self.cam_res = cam_cfg.left.resolution
        self.lidar2cam_T = self._get_lidar2cam_matrix(cam_cfg, lidar_cfg)

    def _get_lidar2cam_matrix(self, cam_cfg: OmegaConf, lidar_cfg: OmegaConf) -> np.ndarray:
        """Get lidar2cam matrix.

        Args:
            cam_cfg (OmegaConf): camera configuration
            lidar_cfg (OmegaConf): lidar configuration

        Returns:
            np.ndarray: lidar2cam matrix
        """
        baselink2cam_q = np.quaternion(*cam_cfg.left.baselink2cam.q)  # w, x, y, z
        baselink2cam_t = np.asarray(cam_cfg.left.baselink2cam.t)
        baselink2cam_T = self.build_matrix(*baselink2cam_t, baselink2cam_q)  # left_cam

        baselink2lidar_q = np.quaternion(*lidar_cfg.baselink2lidar.q)  # w, x, y, z
        baselink2lidar_t = np.asarray(lidar_cfg.baselink2lidar.t)
        baselink2lidar_T = self.build_matrix(*baselink2lidar_t, baselink2lidar_q)

        # lidar2baselink_T = np.linalg.inv(baselink2lidar_T)
        cam2baselink_T = np.linalg.inv(baselink2cam_T)
        lidar2cam_T = cam2baselink_T @ baselink2lidar_T
        return lidar2cam_T

    def __call__(
        self,
        points: np.ndarray,
        return_mask: Optional[bool] = True,
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Project pointcloud to camera image plane.

        Args:
            points (np.ndarray): pointcloud to project
            return_mask (bool, optional): whether to return mask. Defaults to True.

        Returns:
            Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
                if return_mask: (uv, depths, in_image)
                else: (uv, depths)
        """
        return self.project_scan_to_camera(points, return_mask)

    def project_scan_to_camera(
        self,
        points: np.ndarray,
        return_mask: Optional[bool] = True,
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Project pointcloud to camera image plane.

        Args:
            points (np.ndarray): pointcloud to project
            return_mask (bool, optional): whether to return mask. Defaults to True.

        Returns:
            Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
                if return_mask: (uv, depths, in_image)
                else: (uv, depths)

        Raises:
            ValueError: if wrong shape of points array

        """
        if points.shape[0] != 3:
            # logger.debug(f"Transposing pointcloud {points.shape} -> {points.T.shape}")
            points = points.T

        points = np.vstack((points, np.ones((1, points.shape[1]))))

        points = self.lidar2cam_T @ points

        if points.shape[0] == 3:
            points = np.vstack((points, np.ones((1, points.shape[1]))))

        if len(points.shape) != 2 or points.shape[0] != 4:
            raise ValueError(
                f"Wrong shape of points array: {points.shape}; expected: (4, n), where n - number of points."
            )
        in_image = points[2, :] > 0
        depths = points[2, :]  # colors

        uvw = np.dot(self.proj_matrix, points)
        uv = uvw[:2, :]
        w = uvw[2, :]
        uv[0, :] /= w
        uv[1, :] /= w
        in_image = (
            (uv[0, :] >= 0)
            * (uv[0, :] < self.cam_res[0])
            * (uv[1, :] >= 0)
            * (uv[1, :] < self.cam_res[1])
            * in_image
        )
        if return_mask:
            return uv[:, in_image].astype(int), depths, in_image
        return uv[:, in_image].astype(int), depths

    def build_matrix(self, x: float, y: float, z: float, q: float) -> np.ndarray:
        """Build rotation matrix from quaternion.

        Args:
            x (float): x coordinate
            y (float): y coordinate
            z (float): z coordinate
            q (float): quaternion

        Returns:
            np.ndarray: rotation matrix
        """
        M = np.zeros((4, 4))
        if type(q) is not quaternion.quaternion:
            q = np.quaternion(*q)  # type: ignore
        M[:3, :3] = as_rotation_matrix(q)
        M[:, 3] = x, y, z, 1
        return M
