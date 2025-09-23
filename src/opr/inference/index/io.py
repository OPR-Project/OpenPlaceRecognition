"""IO helpers for descriptors, metadata and schema files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from .base import IndexMetric, IndexSchema


def _sha256_file(path: Path, chunk_size: int = 2**20) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha.update(chunk)
    return sha.hexdigest()


def load_descriptors(path: str | Path, mmap: bool = True) -> np.ndarray:
    """Load descriptors.npy as float32 [N, D].

    Args:
        path: Path to `descriptors.npy`.
        mmap: If True, memory-map the array in read-only mode.

    Returns:
        np.ndarray: Float32 array of shape [N, D].

    Raises:
        ValueError: If the loaded array is not 2D.
    """
    arr = np.load(str(path), mmap_mode="r" if mmap else None)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    if arr.ndim != 2:
        raise ValueError(f"descriptors at {path} must be 2D; got shape {arr.shape}")
    return arr


def load_meta(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Load meta.parquet and return (db_idx [N], db_pose [N,7], pointcloud_path [N], full_df).

    Args:
        path: Path to `meta.parquet`.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]: `(db_idx, db_pose, db_pointcloud_path, df)`.

    Raises:
        ValueError: If required columns are missing or pose length is not 7.
    """
    df = pd.read_parquet(path)
    if "idx" not in df.columns:
        raise ValueError("meta.parquet must contain 'idx' column")
    if "pose" not in df.columns:
        raise ValueError("meta.parquet must contain 'pose' column (length 7)")
    poses = df["pose"].to_numpy()
    # Ensure each pose element is length-7
    pose_arr = np.stack([np.asarray(p, dtype=np.float32) for p in poses], axis=0)
    if pose_arr.shape[1] != 7:
        raise ValueError(f"pose must be length 7; got shape {pose_arr.shape}")
    idx_arr = df["idx"].to_numpy(dtype=np.int64)

    # Optional pointcloud_path column: relative path to pointcloud file (pcd/bin) or NaN
    if "pointcloud_path" in df.columns:
        paths_series = df["pointcloud_path"]

        # Normalize: allow NaN/None; otherwise expect str ending with .pcd or .bin
        def _validate_path(val: object) -> object:
            if pd.isna(val):
                return np.nan
            if isinstance(val, str):
                lower = val.lower()
                if (lower.endswith(".pcd") or lower.endswith(".bin")) and not Path(val).is_absolute():
                    return val
            raise ValueError(
                "meta.parquet 'pointcloud_path' must be a relative str ending with .pcd or .bin, or NaN"
            )

        db_pc_path = np.array([_validate_path(v) for v in paths_series.to_list()], dtype=object)
    else:
        # If missing, fill with NaN object array for compatibility
        db_pc_path = np.array([np.nan] * idx_arr.shape[0], dtype=object)

    return idx_arr, pose_arr, db_pc_path, df


def load_schema(path: str | Path) -> IndexSchema:
    """Load schema.json into IndexSchema.

    Args:
        path: Path to `schema.json`.

    Returns:
        IndexSchema: Parsed schema information.
    """
    with Path(path).open("r") as f:
        data = json.load(f)
    metric_raw = data.get("metric")
    metric = IndexMetric(metric_raw) if metric_raw is not None else IndexMetric.L2
    return IndexSchema(
        version=str(data.get("version", "1")),
        dim=int(data["dim"]),
        metric=metric,
        created_at=str(data.get("created_at", "")),
        opr_version=str(data.get("opr_version", "")) if data.get("opr_version") is not None else None,
        descriptors_sha256=(
            str(data.get("descriptors_sha256", "")) if data.get("descriptors_sha256") is not None else None
        ),
    )


def validate_files(base_dir: str | Path) -> tuple[Path, Path, Path]:
    """Validate required files exist and return their paths.

    Args:
        base_dir: Directory containing index files.

    Returns:
        tuple[Path, Path, Path]: Paths to `(descriptors.npy, meta.parquet, schema.json)`.

    Raises:
        FileNotFoundError: If any of the required files is missing.
    """
    base = Path(base_dir)
    desc = base / "descriptors.npy"
    meta = base / "meta.parquet"
    schema = base / "schema.json"
    for p in (desc, meta, schema):
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")
    return desc, meta, schema


def validate_consistency(
    descriptors: np.ndarray, db_idx: np.ndarray, schema: IndexSchema, desc_path: Path
) -> None:
    """Validate shapes and optional hash consistency between files.

    Args:
        descriptors: Float32 [N, D] descriptors array.
        db_idx: Int64 [N] dataset ids from meta.
        schema: Parsed schema.
        desc_path: Path to descriptors file for hash calculation when provided.

    Raises:
        ValueError: If shapes/lengths mismatch or descriptors hash does not match schema.
    """
    if descriptors.shape[0] != db_idx.shape[0]:
        raise ValueError(
            f"Row count mismatch: descriptors N={descriptors.shape[0]} vs meta N={db_idx.shape[0]}"
        )
    if descriptors.shape[1] != schema.dim:
        raise ValueError(f"Dim mismatch: descriptors D={descriptors.shape[1]} vs schema dim={schema.dim}")
    # Optional hash validation
    if schema.descriptors_sha256:
        file_hash = _sha256_file(desc_path)
        if file_hash != schema.descriptors_sha256:
            raise ValueError("descriptors hash mismatch: schema.descriptors_sha256 does not match file")
