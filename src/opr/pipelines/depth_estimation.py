import numpy as np
import torch
import torch.nn as nn
from os import PathLike
from argparse import Namespace
from opr.utils import init_model, parse_device

class DepthEsitmation:
    def __init__(self, 
                 camera_matrix: Dict[str, float],
                 lidar_to_camera_transform: np.ndarray,
                 model: nn.Module,
                 model_weights_path: Optional[Union[str, PathLike]] = None,
                 device: Union[str, int, torch.device] = "cuda"):
        self.device = parse_device(device)
        self.model = init_model(model, model_weights_path, self.device)
        self.camera_matrix = Namespace(**camera_matrix)
        self.lidar_to_camera_transform = torch.Tensor(lidar_to_camera_transform).to(self.device)
    

    def get_depth_with_lidar(image: torch.Tensor, point_cloud: torch.Tensor) -> np.ndarray:
        predicted_depth = self.model(image).cpu().numpy()
        pcd_extended = torch.cat((point_cloud, torh.ones(point_cloud.shape[0], 1)), dim=1)
        pcd_transformed = torch.matmul(pcd_extended, self.lidar_to_camera_transform)
        pcd_forward_segment = pcd_transformed[pcd_transformed[:, 2] > 0]
        pcd_in_fov = pcd_forward_segment[torch.abs(pcd_forward_segment[:, 0] / pcd_forward_segment[:, 2]) < self.camera_matrix.cx / self.camera_matrix.f]
        pcd_in_fov = pcd_in_fov[torch.abs(pcd_in_fov[:, 1] / pcd_in_fov[:, 2]) < self.camera_matrix.cy / self.camera_matrix.f]
        pcd_in_fov_numpy = pcd_in_fov.cpu().numpy()
        scale_coefs = []
        for x, y, z in pcd_in_fov_numpy:
            i = int(self.camera_matrix.cx + x / z * f)
            j = int(self.camera_matrix.cy + y / z * f)
            scale_coefs.append(z / predicted_depth[i, j])
        return predicted_depth * np.mean(scale_coefs)