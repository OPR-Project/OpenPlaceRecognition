"""Point cloud preprocessing to MinkowskiEngine-friendly format."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


class PointCloudMinkPreprocessor:
    """Convert raw point clouds to MinkowskiEngine batched coords and features.

    This preprocessor accepts numpy arrays or torch tensors of shape [N,3] or
    [N,4] where the last column is treated as intensity. It quantizes the cloud
    with a specified voxel size using MinkowskiEngine's `sparse_quantize` and
    returns a dict ready for PlaceRecognition models.
    """

    def __init__(self, quantization_size: float = 0.5, use_intensity: bool = False) -> None:
        """Initialize the point cloud preprocessor.

        Args:
            quantization_size (float): Voxel size for MinkowskiEngine sparse quantization.
            use_intensity (bool): Whether to use intensity channel when available.
        """
        self.quantization_size = quantization_size
        self.use_intensity = use_intensity

    def __call__(self, pc: np.ndarray | torch.Tensor) -> dict[str, Tensor]:
        """Preprocess a single point cloud.

        Args:
            pc: Point cloud array or tensor with shape [N,3] or [N,4].

        Returns:
            dict[str, Tensor]: keys `pointclouds_lidar_coords`, `pointclouds_lidar_feats`.

        Raises:
            ValueError: If the input shape is invalid.
            ImportError: If MinkowskiEngine is not installed.
        """
        try:
            import MinkowskiEngine as ME
        except Exception as exc:  # pragma: no cover - env dependent
            raise ImportError("MinkowskiEngine is required for point cloud preprocessing") from exc

        if isinstance(pc, np.ndarray):
            pts = torch.from_numpy(pc.astype(np.float32, copy=False))
        elif isinstance(pc, torch.Tensor):
            pts = pc.to(dtype=torch.float32)
        else:
            raise ValueError("pc must be a numpy.ndarray or torch.Tensor")

        if pts.ndim != 2 or pts.shape[1] not in (3, 4):
            raise ValueError("pc must have shape [N,3] or [N,4]")

        coords = pts[:, :3]
        if self.use_intensity and pts.shape[1] == 4:
            feats = pts[:, 3:4]
        else:
            feats = torch.ones((pts.shape[0], 1), dtype=torch.float32, device=pts.device)

        q_coords, q_feats = ME.utils.sparse_quantize(
            coordinates=coords, features=feats, quantization_size=self.quantization_size
        )
        batch_coords = ME.utils.batched_coordinates([q_coords])
        return {
            "pointclouds_lidar_coords": batch_coords,
            "pointclouds_lidar_feats": q_feats,
        }
