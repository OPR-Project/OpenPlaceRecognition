"""Localization pipeline built on top of Place Recognition and Registration.

This pipeline:
- runs top-k Place Recognition to retrieve database candidates
- loads each candidate's point cloud by `pointcloud_path` from the index metadata
- runs registration to estimate the rigid transform
- composes an absolute 7-DoF pose estimate for each candidate and selects the best

Notes:
- `registration_confidence` is a placeholder (1.0) for now. If your registration
  method exposes a meaningful score, adapt the pipeline to return it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch import Tensor

from opr.inference.index import Index
from opr.inference.io import PointCloudStore
from opr.inference.pipelines.place_recognition import (
    PlaceRecognitionPipeline,
    PlaceRecognitionResult,
)


def _pose7_to_matrix(pose7: np.ndarray) -> np.ndarray:
    """Convert pose [tx,ty,tz,qx,qy,qz,qw] to 4x4 matrix."""
    t = pose7[:3]
    q = pose7[3:]
    R = Rotation.from_quat(q).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _matrix_to_pose7(T: np.ndarray) -> np.ndarray:
    """Convert 4x4 matrix to pose [tx,ty,tz,qx,qy,qz,qw]."""
    Rm = T[:3, :3]
    t = T[:3, 3]
    q = Rotation.from_matrix(Rm).as_quat()
    return np.concatenate([t, q]).astype(np.float64, copy=False)


@dataclass
class LocalizedCandidate:
    """Per-candidate localization result."""

    idx: int
    pr_distance: float
    db_pose: np.ndarray  # [7]
    db_pointcloud_path: str | None
    estimated_pose: np.ndarray  # [7]
    registration_confidence: float


@dataclass
class LocalizationResult:
    """Full localization result."""

    version: str
    candidates: list[LocalizedCandidate]
    chosen_idx: int


class LocalizationPipeline:
    """Top-k localization using PR candidates and point cloud registration.

    Args:
        index: Loaded index providing metadata (`db_idx`, `db_pose`, `db_pointcloud_path`).
        place_recognition: Place Recognition pipeline.
        registration: Registration pipeline exposing `infer(query_pc, db_pc) -> np.ndarray(4,4)`.
        index_root: Root directory to resolve relative database pointcloud paths.
        require_db_pointcloud: If True, raise when a candidate lacks a pointcloud path
            or file; otherwise skip such candidates.
    """

    def __init__(
        self,
        index: Index,
        place_recognition: PlaceRecognitionPipeline,
        registration: object,
        index_root: str | Path,
        require_db_pointcloud: bool = False,
    ) -> None:
        """Initialize the localization pipeline.

        Args:
            index (Index): Retrieval index that provides metadata mapping
                `(db_idx, db_pose, db_pointcloud_path)` for row positions.
            place_recognition (PlaceRecognitionPipeline): Pipeline to obtain top-k
                candidates and raw distances for a query.
            registration (object): Registration component exposing
                `infer(query_pc: Tensor, db_pc: Tensor) -> np.ndarray (4x4)`.
            index_root (str | Path): Root directory for resolving relative
                database point cloud paths from the index metadata.
            require_db_pointcloud (bool): If True, missing/invalid DB point cloud
                paths cause a FileNotFoundError; if False, such candidates are
                skipped. Defaults to False.
        """
        self.index = index
        self.pr = place_recognition
        self.reg = registration
        self.store = PointCloudStore(root_dir=Path(index_root))
        self.require_db_pointcloud = require_db_pointcloud

    @torch.inference_mode()
    def infer(
        self,
        pr_input: dict[str, Tensor],
        query_pc: Tensor,
        k: int = 5,
    ) -> LocalizationResult:
        """Run localization for a single query.

        Args:
            pr_input: Dict of tensors for the PR model input.
            query_pc: Raw query point cloud tensor [N,3] for registration.
            k: Number of PR candidates to evaluate.

        Returns:
            LocalizationResult: Per-candidate estimated poses and the chosen match.

        Raises:
            FileNotFoundError: When `require_db_pointcloud=True` and a candidate
                has a missing/invalid `pointcloud_path` or the file is empty.
            RuntimeError: When no valid candidates remain after loading DB point clouds.
        """
        # Run PR
        pr_res: PlaceRecognitionResult = self.pr.infer(pr_input, k=k)
        inds = pr_res.indices
        dists = pr_res.distances
        db_idx, db_pose, db_pc_path = self.index.get_meta(inds)

        candidates: list[LocalizedCandidate] = []

        for i in range(inds.shape[0]):
            # Load DB PC if available
            rel_path_obj = db_pc_path[i]
            rel_path: str | None
            if isinstance(rel_path_obj, str):
                rel_path = rel_path_obj
            else:
                rel_path = None

            if rel_path is None:
                if self.require_db_pointcloud:
                    raise FileNotFoundError("Database pointcloud path is missing for a candidate")
                # Skip candidate
                continue

            db_pc = self.store.load(rel_path)
            if db_pc.numel() == 0:
                if self.require_db_pointcloud:
                    raise FileNotFoundError("Database pointcloud file is missing or empty")
                continue

            # Run registration
            T_q_to_db = self.reg.infer(query_pc=query_pc, db_pc=db_pc)
            # Compose absolute pose: db_pose ∘ inv(T_q_to_db)
            T_db = _pose7_to_matrix(db_pose[i])
            T_est = T_db @ np.linalg.inv(T_q_to_db)
            est_pose = _matrix_to_pose7(T_est)

            candidate = LocalizedCandidate(
                idx=int(db_idx[i]),
                pr_distance=float(dists[i]),
                db_pose=db_pose[i],
                db_pointcloud_path=rel_path,
                estimated_pose=est_pose,
                registration_confidence=1.0,
            )
            candidates.append(candidate)

        if not candidates:
            raise RuntimeError("No valid candidates after loading database point clouds")

        # Choose best by registration_confidence (all equal -> fallback to smallest PR distance)
        best = min(candidates, key=lambda c: (-c.registration_confidence, c.pr_distance))
        return LocalizationResult(version="1", candidates=candidates, chosen_idx=best.idx)
