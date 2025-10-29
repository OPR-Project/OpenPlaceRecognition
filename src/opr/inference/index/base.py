"""Abstract interfaces for retrieval index backends.

Use `FaissFlatIndex` for the current FAISS Flat (L2/IP) implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np  # type: ignore


class IndexMetric(str, Enum):
    """Distance/Similarity metric used by the index."""

    L2 = "l2"
    IP = "ip"  # inner product


@dataclass(frozen=True)
class IndexSchema:
    """Schema information loaded from schema.json."""

    version: str
    dim: int
    metric: IndexMetric
    created_at: str
    opr_version: str | None
    descriptors_sha256: str | None


class Index(ABC):
    """Abstract base for retrieval index backends."""

    @classmethod
    @abstractmethod
    def load(cls, directory: str | Path) -> "Index":
        """Load index from a directory containing descriptors, meta and schema."""

    @abstractmethod
    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Search top-k.

        Args:
            queries: float32 array of shape [Q, D].
            k: number of nearest neighbors to return.

        Returns:
            indices: int64 array [Q, k] of internal row positions (0..N-1).
            distances: float32 array [Q, k] of raw backend distances.
        """

    @abstractmethod
    def size(self) -> int:
        """Number of database items (N)."""

    @abstractmethod
    def dim(self) -> int:
        """Descriptor dimensionality (D)."""

    @abstractmethod
    def metric(self) -> IndexMetric:
        """Metric used (L2 or IP)."""

    @abstractmethod
    def get_meta(self, row_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fetch aligned metadata for given row positions.

        Args:
            row_positions: Array of internal row indices (0..N-1) to fetch metadata for.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple of `(db_idx, db_pose, db_pointcloud_path)` where:
                - `db_idx` is int64 array of shape [M] with dataset item ids.
                - `db_pose` is float32 array of shape [M, 7] with poses
                  in order `tx, ty, tz, qx, qy, qz, qw`.
                - `db_pointcloud_path` is object array of shape [M] with each element being
                  a relative path string like "scans/000227.pcd" or "scans/000227.bin",
                  or `numpy.nan` when not available.
        """
