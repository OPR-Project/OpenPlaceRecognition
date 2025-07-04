"""Basic Place Recognition pipelines."""

from os import PathLike
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from opr.optional_deps import lazy
from opr.utils import init_model, parse_device

try:
    import faiss
except ImportError as import_error:
    raise ImportError(
        "The 'faiss' package is not installed. Please install it manually. "
        "Details: https://github.com/facebookresearch/faiss",
    ) from import_error

# Lazy import for MinkowskiEngine
ME = lazy("MinkowskiEngine", feature="sparse convolutions")


class PlaceRecognitionPipeline:
    """Basic Place Recognition pipeline."""

    def __init__(
        self,
        database_dir: Union[str, PathLike],
        model: nn.Module,
        model_weights_path: Optional[Union[str, PathLike]] = None,
        device: Union[str, int, torch.device] = "cpu",
        pointcloud_quantization_size: float = 0.5,
    ) -> None:
        """Basic Place Recognition pipeline.

        Args:
            database_dir (Union[str, PathLike]): Path to the database directory. The directory must contain
                "track.csv" and "index.faiss" files.
            model (nn.Module): Model. The forward method must take a dictionary and return a dictionary
                in the predefined format. See the "infer" method for details.
            model_weights_path (Union[str, PathLike], optional): Path to the model weights.
                If None, the weights are not loaded. Defaults to None.
            device (Union[str, int, torch.device]): Device to use. Defaults to "cpu".
            pointcloud_quantization_size (float): Pointcloud quantization size. Defaults to 0.5.
        """
        self.device = parse_device(device)
        self.model = init_model(model, model_weights_path, self.device)
        self.database_dir = Path(database_dir)
        self._init_database(self.database_dir)
        self._pointcloud_quantization_size = pointcloud_quantization_size

    def _init_database(self, database_dir: Union[str, PathLike]) -> None:
        """Initialize database."""
        self.database_df = pd.read_csv(Path(database_dir) / "track.csv", index_col=0)
        database_index_filepath = Path(database_dir) / "index.faiss"
        if not database_index_filepath.exists():
            raise FileNotFoundError(f"Database index not found: {database_index_filepath}. Create it first.")
        self.database_index = faiss.read_index(str(database_index_filepath))
        if self.device.type == "cuda":
            res = faiss.StandardGpuResources()
            idx = self.device.index if self.device.index is not None else 0
            self.database_index = faiss.index_cpu_to_gpu(res, idx, self.database_index)

    def _preprocess_input(self, input_data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Preprocess input data."""
        out_dict: Dict[str, Tensor] = {}
        for key in input_data:
            if key.startswith("image_"):
                out_dict[f"images_{key[6:]}"] = input_data[key].unsqueeze(0).to(self.device)
            elif key.startswith("mask_"):
                out_dict[f"masks_{key[5:]}"] = input_data[key].unsqueeze(0).to(self.device)
            elif key == "pointcloud_lidar_coords":
                quantized_coords, quantized_feats = ME.utils.sparse_quantize(
                    coordinates=input_data["pointcloud_lidar_coords"],
                    features=input_data["pointcloud_lidar_feats"],
                    quantization_size=self._pointcloud_quantization_size,
                )
                out_dict["pointclouds_lidar_coords"] = ME.utils.batched_coordinates([quantized_coords]).to(
                    self.device
                )
                out_dict["pointclouds_lidar_feats"] = quantized_feats.to(self.device)
            elif key == "soc":
                out_dict["soc"] = input_data[key].unsqueeze(0).to(self.device)
        return out_dict

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

                "idx" for predicted index in the database,

                "pose" for predicted pose in the format [tx, ty, tz, qx, qy, qz, qw],

                "descriptor" for predicted descriptor.
        """
        input_data = self._preprocess_input(input_data)
        output = {}
        with torch.no_grad():
            descriptor = self.model(input_data)["final_descriptor"].cpu().numpy().reshape(1, -1)
        _, pred_i = self.database_index.search(descriptor, 1)
        pred_i = pred_i[0][0]
        pred_pose = self.database_df.iloc[pred_i][['northing', 'easting', 'down']].to_numpy(
            dtype=float
        )
        output["idx"] = pred_i
        output["pose"] = pred_pose
        output["descriptor"] = descriptor[0]
        return output
