"""Unit tests for RansacPointCloudRegistrationPipeline."""

import numpy as np
import pytest
import torch

o3d = pytest.importorskip("open3d")


@pytest.mark.unit
def test_infer_returns_result_with_4x4_transform() -> None:
    """Pipeline should return a RegistrationResult with a 4x4 transform."""
    from opr.inference.pipelines import RansacPointCloudRegistrationPipeline

    rng = np.random.default_rng(0)
    pts = rng.normal(size=(200, 3)).astype(np.float32)

    query_pc = torch.from_numpy(pts.copy())
    db_pc = torch.from_numpy(pts.copy())

    pipe = RansacPointCloudRegistrationPipeline(voxel_downsample_size=0.2)
    res = pipe.infer(query_pc=query_pc, db_pc=db_pc)

    T = res.transformation
    assert T.shape == (4, 4)
    # last row should be [0,0,0,1] approximately
    np.testing.assert_allclose(T[3], np.array([0, 0, 0, 1], dtype=T.dtype), atol=1e-6)


@pytest.mark.unit
def test_identity_pointclouds_produce_near_identity_transform() -> None:
    """Registering identical clouds should produce a transform near identity."""
    from opr.inference.pipelines import RansacPointCloudRegistrationPipeline

    rng = np.random.default_rng(1)
    # a structured cloud: points on a cube surface for more stable features
    base = []
    for x in (-1.0, 0.0, 1.0):
        for y in (-1.0, 0.0, 1.0):
            for z in (-1.0, 0.0, 1.0):
                base.append([x, y, z])
    base = np.array(base, dtype=np.float32)
    # add a few random points
    rnd = rng.uniform(-1.0, 1.0, size=(200, 3)).astype(np.float32)
    pts = np.concatenate([base, rnd], axis=0)

    query_pc = torch.from_numpy(pts)
    db_pc = torch.from_numpy(pts.copy())

    pipe = RansacPointCloudRegistrationPipeline(voxel_downsample_size=0.2)
    res = pipe.infer(query_pc=query_pc, db_pc=db_pc)
    T = res.transformation

    identity = np.eye(4, dtype=T.dtype)
    # Allow small numerical tolerance
    assert np.linalg.norm(T - identity) < 0.2


@pytest.mark.unit
def test_infer_returns_registration_result_with_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that the pipeline returns a RegistrationResult with metrics."""
    from opr.inference.pipelines.registration import (
        RansacPointCloudRegistrationPipeline,
        RegistrationResult,
    )

    pipeline = RansacPointCloudRegistrationPipeline(voxel_downsample_size=0.5)

    def fake_preprocess(self, points: torch.Tensor) -> tuple[object, object]:
        return object(), object()

    class FakeResult:
        def __init__(self) -> None:
            self.transformation = np.eye(4, dtype=np.float64)
            self.fitness = 0.5
            self.inlier_rmse = 0.1
            self.correspondence_set = list(range(42))

    def fake_execute(self, s: object, t: object, sf: object, tf: object) -> FakeResult:
        return FakeResult()

    monkeypatch.setattr(RansacPointCloudRegistrationPipeline, "_preprocess_point_cloud", fake_preprocess)
    monkeypatch.setattr(RansacPointCloudRegistrationPipeline, "_execute_global_registration", fake_execute)

    q = torch.zeros((10, 3), dtype=torch.float32)
    d = torch.zeros((10, 3), dtype=torch.float32)
    res = pipeline.infer(query_pc=q, db_pc=d)

    assert isinstance(res, RegistrationResult)
    assert res.transformation.shape == (4, 4)
    assert res.transformation.dtype == np.float64
    assert np.allclose(res.transformation, np.eye(4))
    assert res.success is True
    assert res.fitness == 0.5
    assert res.inlier_rmse == 0.1
    assert res.num_inliers == 42


@pytest.mark.unit
def test_infer_handles_missing_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that the pipeline returns a RegistrationResult with missing metrics."""
    from opr.inference.pipelines.registration import (
        RansacPointCloudRegistrationPipeline,
        RegistrationResult,
    )

    pipeline = RansacPointCloudRegistrationPipeline(voxel_downsample_size=0.5)

    def fake_preprocess(self, points: torch.Tensor) -> tuple[object, object]:
        return object(), object()

    class FakeResult:
        def __init__(self) -> None:
            self.transformation = np.eye(4, dtype=np.float64)

    def fake_execute(self, s: object, t: object, sf: object, tf: object) -> FakeResult:
        return FakeResult()

    monkeypatch.setattr(RansacPointCloudRegistrationPipeline, "_preprocess_point_cloud", fake_preprocess)
    monkeypatch.setattr(RansacPointCloudRegistrationPipeline, "_execute_global_registration", fake_execute)

    q = torch.zeros((10, 3), dtype=torch.float32)
    d = torch.zeros((10, 3), dtype=torch.float32)
    res = pipeline.infer(query_pc=q, db_pc=d)

    assert isinstance(res, RegistrationResult)
    assert res.success is False
    assert res.fitness is None
    assert res.inlier_rmse is None
    assert res.num_inliers is None
