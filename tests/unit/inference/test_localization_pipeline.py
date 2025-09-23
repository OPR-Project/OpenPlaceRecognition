"""Unit tests for LocalizationPipeline."""

from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.mark.unit
def test_localization_success_identity_registration(tmp_path: Path) -> None:
    """Two valid candidates: identity registration should yield db_pose as estimate."""
    from opr.inference.pipelines.localization import LocalizationPipeline
    from opr.inference.pipelines.place_recognition import PlaceRecognitionResult

    # Create tiny DB point clouds as .bin (x,y,z triples)
    pc_a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    pc_b = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    path_a = tmp_path / "scans" / "000100.bin"
    path_b = tmp_path / "scans" / "000101.bin"
    path_a.parent.mkdir(parents=True, exist_ok=True)
    pc_a.tofile(path_a)
    pc_b.tofile(path_b)

    # Stub index returning metadata for rows [0,1]
    db_idx = np.array([100, 101], dtype=np.int64)
    db_pose = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    db_pc_path = np.array(["scans/000100.bin", "scans/000101.bin"], dtype=object)

    class FakeIndex:
        def get_meta(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            return db_idx[rows], db_pose[rows], db_pc_path[rows]

    # Stub PR returning two candidates
    class FakePR:
        def infer(self, _input: dict, k: int) -> PlaceRecognitionResult:  # pylint: disable=unused-argument
            return PlaceRecognitionResult(
                descriptor=np.zeros(8, dtype=np.float32),
                indices=np.array([0, 1], dtype=np.int64),
                distances=np.array([0.5, 1.0], dtype=np.float32),
                db_idx=db_idx,
                db_pose=db_pose,
            )

    # Stub registration returning identity transform
    class FakeReg:
        def infer(self, query_pc: torch.Tensor, db_pc: torch.Tensor) -> np.ndarray:  # noqa: D401
            return np.eye(4, dtype=np.float64)

    pipeline = LocalizationPipeline(
        index=FakeIndex(),
        place_recognition=FakePR(),
        registration=FakeReg(),
        index_root=tmp_path,
        require_db_pointcloud=False,
    )

    pr_input: dict = {}
    query_pc = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float32)

    res = pipeline.infer(pr_input, query_pc, k=2)

    assert len(res.candidates) == 2
    # With identity registration, estimated pose must equal db_pose
    for cand, pose in zip(res.candidates, db_pose):
        np.testing.assert_allclose(cand.estimated_pose, pose, atol=1e-6)
    # chosen_idx should be 100 (smaller PR distance tie-breaker)
    assert res.chosen_idx == 100


@pytest.mark.unit
def test_missing_pointcloud_skipped_by_default(tmp_path: Path) -> None:
    """Candidate without pointcloud is skipped when require flag is False."""
    from opr.inference.pipelines.localization import LocalizationPipeline
    from opr.inference.pipelines.place_recognition import PlaceRecognitionResult

    # One valid, one missing
    pc = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    path_ok = tmp_path / "db" / "000200.bin"
    path_ok.parent.mkdir(parents=True, exist_ok=True)
    pc.tofile(path_ok)

    db_idx = np.array([200, 201], dtype=np.int64)
    db_pose = np.array(
        [
            [0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    db_pc_path = np.array(["db/000200.bin", np.nan], dtype=object)

    class FakeIndex:
        def get_meta(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            return db_idx[rows], db_pose[rows], db_pc_path[rows]

    class FakePR:
        def infer(self, _input: dict, k: int) -> PlaceRecognitionResult:  # pylint: disable=unused-argument
            return PlaceRecognitionResult(
                descriptor=np.zeros(8, dtype=np.float32),
                indices=np.array([0, 1], dtype=np.int64),
                distances=np.array([0.1, 0.2], dtype=np.float32),
                db_idx=db_idx,
                db_pose=db_pose,
            )

    class FakeReg:
        def infer(self, query_pc: torch.Tensor, db_pc: torch.Tensor) -> np.ndarray:
            return np.eye(4, dtype=np.float64)

    pipeline = LocalizationPipeline(
        index=FakeIndex(),
        place_recognition=FakePR(),
        registration=FakeReg(),
        index_root=tmp_path,
        require_db_pointcloud=False,
    )
    res = pipeline.infer({}, torch.zeros((1, 3), dtype=torch.float32), k=2)

    assert len(res.candidates) == 1
    assert res.candidates[0].idx == 200


@pytest.mark.unit
def test_missing_pointcloud_raises_when_required(tmp_path: Path) -> None:
    """Missing pointcloud raises FileNotFoundError when required."""
    from opr.inference.pipelines.localization import LocalizationPipeline
    from opr.inference.pipelines.place_recognition import PlaceRecognitionResult

    db_idx = np.array([300], dtype=np.int64)
    db_pose = np.array([[0, 0, 0, 0, 0, 0, 1]], dtype=np.float64)
    db_pc_path = np.array([np.nan], dtype=object)

    class FakeIndex:
        def get_meta(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            return db_idx[rows], db_pose[rows], db_pc_path[rows]

    class FakePR:
        def infer(self, _input: dict, k: int) -> PlaceRecognitionResult:  # pylint: disable=unused-argument
            return PlaceRecognitionResult(
                descriptor=np.zeros(8, dtype=np.float32),
                indices=np.array([0], dtype=np.int64),
                distances=np.array([0.1], dtype=np.float32),
                db_idx=db_idx,
                db_pose=db_pose,
            )

    class FakeReg:
        def infer(self, query_pc: torch.Tensor, db_pc: torch.Tensor) -> np.ndarray:
            return np.eye(4, dtype=np.float64)

    pipeline = LocalizationPipeline(
        index=FakeIndex(),
        place_recognition=FakePR(),
        registration=FakeReg(),
        index_root=tmp_path,
        require_db_pointcloud=True,
    )

    with pytest.raises(FileNotFoundError):
        pipeline.infer({}, torch.zeros((1, 3), dtype=torch.float32), k=1)
