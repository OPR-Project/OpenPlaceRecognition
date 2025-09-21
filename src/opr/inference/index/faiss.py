"""FAISS Flat index implementation.

Builds `IndexFlatL2` or `IndexFlatIP` at load time from descriptors.npy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover - surfaced clearly at import time
    raise ImportError("FAISS is required for FaissFlatIndex. Please install faiss-cpu or faiss-gpu.") from e

from .base import Index, IndexMetric, IndexSchema
from .io import (
    load_descriptors,
    load_meta,
    load_schema,
    validate_consistency,
    validate_files,
)


class FaissFlatIndex(Index):
    """FAISS Flat backend using L2 or IP metric.

    Builds an in-memory `faiss.IndexFlatL2` or `faiss.IndexFlatIP` at load time
    from `descriptors.npy` and exposes a minimal search API returning raw
    distances and row positions.
    """

    def __init__(
        self,
        descriptors: np.ndarray,
        db_idx: np.ndarray,
        db_pose: np.ndarray,
        schema: IndexSchema,
    ) -> None:
        """Initialize the index instance.

        Args:
            descriptors: Float32 array [N, D] of database descriptors.
            db_idx: Int64 array [N] of dataset item ids aligned with descriptors.
            db_pose: Float32 array [N, 7] of poses aligned with descriptors.
            schema: Parsed `IndexSchema` with dim/metric.
        """
        self._schema = schema
        self._descriptors = np.ascontiguousarray(descriptors.astype(np.float32, copy=False))
        self._db_idx = db_idx.astype(np.int64, copy=False)
        self._db_pose = db_pose.astype(np.float32, copy=False)

        d = self._descriptors.shape[1]
        if self._schema.metric == IndexMetric.IP:
            self._index = faiss.IndexFlatIP(d)
        else:
            self._index = faiss.IndexFlatL2(d)
        self._index.add(self._descriptors)

    @classmethod
    def load(cls, directory: str | Path) -> "FaissFlatIndex":
        """Load index from a directory with descriptors/meta/schema.

        Args:
            directory: Path containing `descriptors.npy`, `meta.parquet`, `schema.json`.

        Returns:
            FaissFlatIndex: Loaded index ready for search.
        """
        desc_path, meta_path, schema_path = validate_files(directory)
        descriptors = load_descriptors(desc_path)
        db_idx, db_pose, _ = load_meta(meta_path)
        schema = load_schema(schema_path)
        validate_consistency(descriptors, db_idx, schema, Path(desc_path))
        return cls(descriptors=descriptors, db_idx=db_idx, db_pose=db_pose, schema=schema)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Search top-k nearest neighbors.

        Args:
            queries: Float32 array [Q, D] of query descriptors.
            k: Number of neighbors to return.

        Returns:
            tuple[np.ndarray, np.ndarray]: `(indices, distances)` where `indices`
            is int64 array [Q, k] of row positions and `distances` is float32
            array [Q, k] of raw FAISS distances.
        """
        q = np.ascontiguousarray(queries.astype(np.float32, copy=False))
        distances, inds = self._index.search(q, k)
        return inds.astype(np.int64, copy=False), distances.astype(np.float32, copy=False)

    def size(self) -> int:
        """Return number of database items (N)."""
        return self._descriptors.shape[0]

    def dim(self) -> int:
        """Return descriptor dimensionality (D)."""
        return self._descriptors.shape[1]

    def metric(self) -> IndexMetric:
        """Return metric used by the index (L2 or IP)."""
        return self._schema.metric

    def get_meta(self, row_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map row positions to dataset indices and poses.

        Args:
            row_positions: Int64 array [M] of internal row positions (0..N-1).

        Returns:
            tuple[np.ndarray, np.ndarray]: `(db_idx, db_pose)` aligned with input rows.
        """
        rows = row_positions.astype(np.int64, copy=False)
        return self._db_idx[rows], self._db_pose[rows]
