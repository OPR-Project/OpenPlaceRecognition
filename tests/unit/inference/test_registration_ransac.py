"""Unit tests for RansacPointCloudRegistrationPipeline."""

import numpy as np
import pytest
import torch

o3d = pytest.importorskip("open3d")


@pytest.mark.unit
def test_infer_returns_4x4_matrix() -> None:
    """Pipeline should return a 4x4 transform with a valid homogeneous row."""
    from opr.inference.pipelines import RansacPointCloudRegistrationPipeline

    rng = np.random.default_rng(0)
    pts = rng.normal(size=(200, 3)).astype(np.float32)

    query_pc = torch.from_numpy(pts.copy())
    db_pc = torch.from_numpy(pts.copy())

    pipe = RansacPointCloudRegistrationPipeline(voxel_downsample_size=0.2)
    T = pipe.infer(query_pc=query_pc, db_pc=db_pc)

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
    T = pipe.infer(query_pc=query_pc, db_pc=db_pc)

    identity = np.eye(4, dtype=T.dtype)
    # Allow small numerical tolerance
    assert np.linalg.norm(T - identity) < 0.2
