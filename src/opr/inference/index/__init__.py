"""Index module for runtime retrieval.

This package provides a minimal, high-performance FAISS Flat index built at load
time from `descriptors.npy`, alongside utilities to read associated metadata.

Public APIs:
- `Index` (abstract base) and `IndexMetric`
- `FaissFlatIndex` (L2/IP)

On-disk layout (required):
- `descriptors.npy` (float32 [N,D]), `meta.parquet` (columns: idx:int, pose:[7]),
  `schema.json` (versioned)
"""

from .base import Index, IndexMetric
from .faiss import FaissFlatIndex

__all__ = [
    "Index",
    "IndexMetric",
    "FaissFlatIndex",
]
