import numpy as np
import torch
import torch.nn as nn
from os import PathLike
from argparse import Namespace
from opr.utils import init_model, parse_device
from typing import Dict, Optional, Union
from torchvision.transforms import Resize
from skimage.transform import resize

class DepthEsitmation:

    def __init__(self, 
                 camera_matrix: Dict[str, float],
                 lidar_to_camera_transform: np.ndarray,
                 model: nn.Module,
                 model_weights_path: Optional[Union[str, PathLike]] = None,
                 device: Union[str, int, torch.device] = "cuda"):
        self.device = parse_device(device)
        self.model = init_model(model, model_weights_path, self.device)
        self.model.eval()
        self.camera_matrix = Namespace(**camera_matrix)
        self.lidar_to_camera_transform = lidar_to_camera_transform#torch.Tensor(lidar_to_camera_transform).to(self.device)
    
    def get_depth_with_lidar(self, image: np.ndarray, point_cloud: np.ndarray) -> np.ndarray:
        raw_img_h, raw_img_w = image.shape[0], image.shape[1]
        image = resize(image, (640, 1120))
        image_tensor = torch.Tensor(np.transpose(image, [2, 0, 1])[np.newaxis, ...]).to(self.device)
        #point_cloud = point_cloud.to(self.device)
        predicted_depth = self.model.inference(image_tensor).cpu().numpy()[0, 0]
        predicted_depth = resize(predicted_depth, (raw_img_h, raw_img_w))
        #pcd_extended = torch.cat((point_cloud, torch.ones(point_cloud.shape[0], 1).to(self.device)), dim=1)
        pcd_extended = np.concatenate((point_cloud, np.ones((point_cloud.shape[0], 1))), axis=1)
        print(pcd_extended.shape)
        pcd_transformed = pcd_extended @ self.lidar_to_camera_transform#torch.matmul(pcd_extended, self.lidar_to_camera_transform)
        pcd_transformed = pcd_transformed[:, :3] / pcd_transformed[:, 3:]
        pcd_forward_segment = pcd_transformed[pcd_transformed[:, 2] > 0]
        pcd_in_fov = pcd_forward_segment[np.abs(pcd_forward_segment[:, 0] / pcd_forward_segment[:, 2]) < self.camera_matrix.cx / self.camera_matrix.f]
        pcd_in_fov = pcd_in_fov[np.abs(pcd_in_fov[:, 1] / pcd_in_fov[:, 2]) < self.camera_matrix.cy / self.camera_matrix.f]
        pcd_in_fov_numpy = pcd_in_fov#.cpu().numpy()
        scale_coefs = []
        for x, y, z in pcd_in_fov_numpy:
            i = int(self.camera_matrix.cy + y / z * self.camera_matrix.f)
            j = int(self.camera_matrix.cx + x / z * self.camera_matrix.f)
            if i < raw_img_h / 3 or i > raw_img_h * 2 / 3:
                continue
            if i < 0 or i >= raw_img_h or j < 0 or j >= raw_img_w:
                continue
            #print('i, j:', i, j)
            #print('x y z:', x, y, z, 'depth:', predicted_depth[i, j])
            scale_coefs.append(z / predicted_depth[i, j])
        print(np.mean(scale_coefs), np.min(scale_coefs), np.max(scale_coefs))
        return predicted_depth * np.mean(scale_coefs)