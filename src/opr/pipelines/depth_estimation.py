import numpy as np
import torch
import torch.nn as nn
from os import PathLike
from argparse import Namespace
from opr.utils import init_model, parse_device
from typing import Dict, Optional, Union
from torchvision.transforms import Resize
from skimage.transform import resize
from sklearn.linear_model import LinearRegression
import torch_tensorrt
import time

class DepthEstimationPipeline:
    def __init__(self, 
                 model: nn.Module,
                 model_type: str = 'AdelaiDepth',
                 align_type: str = 'average',
                 mode: str = 'indoor',
                 model_weights_path: Optional[Union[str, PathLike]] = None,
                 device: Union[str, int, torch.device] = "cuda"):
        """
        This class provides a pipeline for monocular depth estimation with use of sparse lidar point cloud data.

        Args:
            model (nn.Module): PyTorch model for depth estimation.
            model_type (str): Type of the neural depth reconstruction model, either 'AdelaiDepth' or 'DepthAnything'.
            align_type (str): Predicted depth and point cloud alignment type, either 'average' or 'regression'.
            mode (str): Mode of operation, either 'indoor' or 'outdoor'.
            model_weights_path (Optional[Union[str, PathLike]]): Path to the model weights (optional).
            device (Union[str, int, torch.device]): Device to run the model on (defaults to "cuda").
        """

        self.device = parse_device(device)
        self.model = init_model(model, model_weights_path, self.device)
        self.model_type = model_type
        assert self.model_type in ['AdelaiDepth', 'DepthAnything']
        self.align_type = align_type
        assert align_type in ['average', 'regression']
        self.mode = mode
        assert self.mode in ['indoor', 'outdoor']
        self.model.eval()
        self.forward_type = 'fp32'

    def set_camera_matrix(self, camera_matrix: Dict[str, float]):
        """Set the camera intrinsic matrix for calculations."""
        self.camera_matrix = Namespace(**camera_matrix)

    def set_lidar_to_camera_transform(self, transform):
        """Set lidar to camera transform"""
        self.lidar_to_camera_transform = transform
    
    def get_depth_with_lidar(self, image: np.ndarray, point_cloud: np.ndarray) -> np.ndarray:
        """Obtain depth estimation from the provided image and point cloud data.
        Args:
            image: np.ndarray - monocular image
            point_cloud: np.ndarray - sparse lidar point cloud

        Returns:
            depth: np.ndarray - reconstructed depth map with the same height and width as the input image
            zs: np.ndarray - z values of the lidar point cloud projected on the iamge
            errors: np.ndarray - absolute errors of depth reconstruction for the points of the projected lidar point cloud
            rel_errors: np.ndarray - relative errors of depth reconstruction for the points of the projected lidar point cloud
        """
        raw_img_h, raw_img_w = image.shape[0], image.shape[1]

        if self.model_type == 'DepthAnything':
            image = resize(image, (240, 320))
            image = (image * 255).astype(np.uint8)
        else:
            image = resize(image, (480, 640))
        start_time = time.time()
        if self.model_type == 'AdelaiDepth':
            image_tensor = torch.Tensor(np.transpose(image, [2, 0, 1])[np.newaxis, ...]).to(self.device)
            predicted_depth = self.model.inference(image_tensor).cpu().numpy()[0, 0]
        else:
            image_tensor, (h, w) = self.model.image2tensor(image)
            predicted_depth = self.model.infer_image(image)
        predicted_depth = resize(predicted_depth, (raw_img_h, raw_img_w))
        end_time = time.time()
        pcd_extended = np.concatenate((point_cloud, np.ones((point_cloud.shape[0], 1))), axis=1)
        pcd_transformed = pcd_extended @ np.linalg.inv(self.lidar_to_camera_transform).T
        pcd_transformed = pcd_transformed[:, :3] / pcd_transformed[:, 3:]
        pcd_forward_segment = pcd_transformed[pcd_transformed[:, 2] > 0]
        if self.mode == 'indoor':
            pcd_forward_segment = pcd_forward_segment[(pcd_forward_segment[:, 2] < 15) * \
                                (pcd_forward_segment[:, 0] > -15) * (pcd_forward_segment[:, 0] < 15) * \
                                (pcd_forward_segment[:, 1] > -5) * (pcd_forward_segment[:, 1] < 5)]
        else:
            pcd_forward_segment = pcd_forward_segment[(pcd_forward_segment[:, 2] < 50) * \
                                (pcd_forward_segment[:, 0] > -50) * (pcd_forward_segment[:, 0] < 50) * \
                                (pcd_forward_segment[:, 1] > -10) * (pcd_forward_segment[:, 1] < 10)]
        pcd_in_fov = pcd_forward_segment[np.abs(pcd_forward_segment[:, 0] / pcd_forward_segment[:, 2]) < self.camera_matrix.cx / self.camera_matrix.f]
        pcd_in_fov = pcd_in_fov[np.abs(pcd_in_fov[:, 1] / pcd_in_fov[:, 2]) < self.camera_matrix.cy / self.camera_matrix.f]
        pcd_in_fov_numpy = pcd_in_fov
        zs = []
        preds = []
        cnt = 0
        mask = np.zeros_like(predicted_depth)
        for x, y, z in pcd_in_fov_numpy:
            i = int(self.camera_matrix.cy + y / z * self.camera_matrix.f)
            j = int(self.camera_matrix.cx + x / z * self.camera_matrix.f)
            if self.mode == 'indoor':
                if i < raw_img_h / 3 or i > raw_img_h * 2 / 3:
                    continue
                if i < 0 or i >= raw_img_h or j < 0 or j >= raw_img_w:
                    continue
            else:
                if y > 0:
                    continue
            mask[i, j] = predicted_depth[i, j]
            zs.append(z)
            preds.append(predicted_depth[i, j])
            cnt += 1
        zs = np.array(zs)
        preds = np.array(preds)
        print('cnt:', cnt)
        if self.align_type == 'average':
            depth = predicted_depth * np.mean(zs / preds)
        else:
            lin = LinearRegression()
            lin.fit(X=preds[:, np.newaxis], y=zs)
            depth = predicted_depth * lin.coef_[0] + lin.intercept_
        errors = []
        rel_errors = []
        zs = []

        for x, y, z in pcd_in_fov_numpy:
            i = int(self.camera_matrix.cy + y / z * self.camera_matrix.f)
            j = int(self.camera_matrix.cx + x / z * self.camera_matrix.f)
            if i < 0 or i >= raw_img_h or j < 0 or j >= raw_img_w:
                continue
            if y > 0:
                continue
            zs.append(z)
            errors.append(np.abs(depth[i, j] - z))
            rel_errors.append(np.abs(depth[i, j] - z) / z)
            cnt += 1
        errors = np.array(errors)
        rel_errors = np.array(rel_errors)
        rmse = np.sqrt(np.mean(errors ** 2))
        rel = np.mean(rel_errors)
        return depth, zs, errors, rel_errors
