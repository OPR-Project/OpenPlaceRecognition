"""Projection of pointcloud to camera image plane."""

from typing import Optional, Tuple, Union

import numpy as np
import quaternion
from loguru import logger
from omegaconf import OmegaConf
from quaternion import as_rotation_matrix


class NCLTProjector:

    def __init__(self, front=True) -> None:
        self.front = front
        self.x_body_lb3 = [0.035, 0.002, -1.23, -179.93, -0.23, 0.50]
        self.x_lb3_c5 = np.array([0.041862, -0.001905, -0.000212, 160.868615, 89.914152, 160.619894])
        self.x_lb3_c2 = np.array([-0.032674, 0.025928, 0.000176, 160.101024, 89.836345, -56.101163])
        self.K_cam5 = np.array([[399.433184, 0.0, 826.361952],
                                [0.0, 399.433184, 621.668624],
                                [0.0, 0.0, 1.0]])
        self.K_cam2 = np.array([[408.385824, 0.0, 793.959536],
                                [0.0, 408.385824, 623.058320],
                                [0.0, 0.0, 1.0]])

    def project_vel_to_cam(self, hits, K, x_lb3_c):
        hits = np.hstack((hits, np.ones((hits.shape[0], 1)))).T
        T_lb3_c = self.ssc_to_homo(x_lb3_c)
        T_body_lb3 = self.ssc_to_homo(self.x_body_lb3)
        T_lb3_body = np.linalg.inv(T_body_lb3)
        T_c_lb3 = np.linalg.inv(T_lb3_c)
        T_c_body = np.matmul(T_c_lb3, T_lb3_body)
        hits_c = np.matmul(T_c_body, hits)
        hits_im = np.matmul(K, hits_c[0:3, :])
        hits_im[0, :] = hits_im[0, :] / hits_im[2, :]
        hits_im[1, :] = hits_im[1, :] / hits_im[2, :]
        hits_im[:2, :] = self.adjust_points(hits_im[:2, :])
        hits_im = hits_im.astype(int)
        in_image = (hits_im[2, :] > 0) * (hits_im[0, :] >= 0) * (hits_im[0, :] < 320) * (hits_im[1, :] >= 0) * (hits_im[1, :] < 256)
        hits_im[1, :] = 255 - hits_im[1, :]
        return hits_im[:2, in_image], hits_im[2, in_image], in_image

    def __call__(
        self,
        points: np.ndarray,
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
        if self.front:
            return self.project_vel_to_cam(points, self.K_cam5, self.x_lb3_c5)
        else:
            return self.project_vel_to_cam(points, self.K_cam2, self.x_lb3_c2)


    def ssc_to_homo(self, ssc):
        sr = np.sin(np.pi/180.0 * ssc[3])
        cr = np.cos(np.pi/180.0 * ssc[3])
        sp = np.sin(np.pi/180.0 * ssc[4])
        cp = np.cos(np.pi/180.0 * ssc[4])
        sh = np.sin(np.pi/180.0 * ssc[5])
        ch = np.cos(np.pi/180.0 * ssc[5])
        H = np.zeros((4, 4))
        H[0, 0] = ch*cp
        H[0, 1] = -sh*cr + ch*sp*sr
        H[0, 2] = sh*sr + ch*sp*cr
        H[1, 0] = sh*cp
        H[1, 1] = ch*cr + sh*sp*sr
        H[1, 2] = -ch*sr + sh*sp*cr
        H[2, 0] = -sp
        H[2, 1] = cp*sr
        H[2, 2] = cp*cr
        H[0, 3] = ssc[0]
        H[1, 3] = ssc[1]
        H[2, 3] = ssc[2]
        H[3, 3] = 1
        return H

    def adjust_points(self, projected_points, center_crop_size=(960,768), resize_size=(320,256), original_size=(1616,1232)):
        """Adjust 3D LiDAR points projected onto the image plane to correspond to the new image after center cropping and resizing.

        Parameters:
        - projected_points: np.ndarray of shape (N, 2) representing the 2D points on the original image.
        - center_crop_size: Tuple (W_c, H_c) representing the width and height of the crop.
        - resize_size: Tuple (W_r, H_r) representing the width and height of the resized image.
        - original_size: Tuple (W, H) representing the width and height of the original image.

        Returns:
        - adjusted_points: np.ndarray of shape (N, 2) representing the adjusted 2D points on the resized image.
        """
        W, H = original_size
        W_c, H_c = center_crop_size
        W_r, H_r = resize_size
        # Calculate the crop offsets
        x_crop = (W - W_c) / 2
        y_crop = (H - H_c) / 2
        # Calculate the scale factors
        s_x = W_r / W_c
        s_y = H_r / H_c
        # Shift the coordinates for the center crop
        shifted_points = projected_points.T - np.array([x_crop, y_crop])
        # Scale the shifted coordinates for resizing
        adjusted_points = shifted_points * np.array([s_x, s_y])
        return adjusted_points.T



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
