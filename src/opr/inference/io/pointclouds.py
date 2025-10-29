"""Point cloud loading utilities for localization/registration.

This module provides a minimal `PointCloudStore` that resolves relative paths
from `meta.parquet` against a root directory and loads `.pcd` (via Open3D)
or `.bin` (via NumPy) into `torch.Tensor` for downstream pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class PointCloudStore:
    """Simple storage/loader for database point clouds.

    Attributes:
        root_dir: Base directory used to resolve relative `pointcloud_path` values.
        default_num_properties: Expected number of properties per point in `.bin` files;
            typically 3 (x,y,z) or 4 (x,y,z,intensity). The loader will slice to first 3 dims.
    """

    root_dir: Path
    default_num_properties: int = 3

    def resolve(self, rel_path: str) -> Path:
        """Resolve a relative path under `root_dir`.

        Args:
            rel_path: Relative file path (e.g., "scans/000227.bin").

        Returns:
            Path: Absolute path to the file.
        """
        return (self.root_dir / rel_path).resolve()

    def load(self, rel_path_or_nan: object, num_properties: int | None = None) -> torch.Tensor:
        """Load a point cloud from a relative path or return an empty tensor if missing.

        Args:
            rel_path_or_nan: Relative path string or NaN/None.
            num_properties: Number of properties per point for `.bin` files. If None,
                uses `default_num_properties`.

        Returns:
            torch.Tensor: Tensor of shape [N, 3] (float32). Empty tensor if path is missing.

        Raises:
            FileNotFoundError: If the resolved path does not exist.
            ValueError: If file extension is unsupported.
            ImportError: If `.pcd` loading is requested but `open3d` is not installed.
        """
        if rel_path_or_nan is None or (isinstance(rel_path_or_nan, float) and np.isnan(rel_path_or_nan)):
            return torch.empty((0, 3), dtype=torch.float32)

        if not isinstance(rel_path_or_nan, str):
            raise ValueError("pointcloud_path must be a relative string or NaN/None")

        path = self.resolve(rel_path_or_nan)
        if not path.exists():
            raise FileNotFoundError(f"Point cloud file not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".pcd":
            try:
                import open3d as o3d
            except Exception as exc:
                raise ImportError("open3d is required to load .pcd files") from exc
            pcd = o3d.io.read_point_cloud(str(path))
            pts = np.asarray(pcd.points, dtype=np.float32)
            return torch.from_numpy(pts)
        elif suffix == ".bin":
            props = int(self.default_num_properties if num_properties is None else num_properties)
            pts = np.fromfile(str(path), dtype=np.float32).reshape(-1, props)[:, :3]
            pts = pts.astype(np.float32, copy=False)
            return torch.from_numpy(pts)
        else:
            raise ValueError(f"Unsupported point cloud extension: {suffix}")
