"""Unit tests for PointCloudMinkPreprocessor."""

import numpy as np
import pytest
import torch

pytest.importorskip("MinkowskiEngine")


@pytest.mark.unit
def test_numpy_xyz_produces_coords_and_one_features() -> None:
    """[N,3] numpy input should yield ME coords and ones features of shape [N,1]."""
    from opr.inference.preprocessing import PointCloudMinkPreprocessor

    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    pre = PointCloudMinkPreprocessor(quantization_size=1e-6, use_intensity=False)
    out = pre(pts)

    assert set(out.keys()) == {"pointclouds_lidar_coords", "pointclouds_lidar_feats"}
    coords = out["pointclouds_lidar_coords"]
    feats = out["pointclouds_lidar_feats"]
    assert isinstance(coords, torch.Tensor) and isinstance(feats, torch.Tensor)
    # With tiny voxel size, no merges expected
    assert feats.shape == (pts.shape[0], 1)
    np.testing.assert_allclose(feats.cpu().numpy(), np.ones((pts.shape[0], 1), dtype=np.float32))
    # ME batched coords have 4 columns (x,y,z,b)
    assert coords.shape[1] == 4


@pytest.mark.unit
def test_torch_xyzi_uses_intensity_when_enabled() -> None:
    """[N,4] torch input with intensity should propagate intensity to features when enabled."""
    from opr.inference.preprocessing import PointCloudMinkPreprocessor

    pts = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.5],
            [0.1, 0.0, 0.0, 1.5],
            [0.0, 0.1, 0.0, -2.0],
            [0.0, 0.0, 0.1, 3.0],
        ],
        dtype=torch.float32,
    )
    pre = PointCloudMinkPreprocessor(quantization_size=1e-6, use_intensity=True)
    out = pre(pts)

    feats = out["pointclouds_lidar_feats"]
    # With tiny voxel size, ordering preserved and no merges expected
    np.testing.assert_allclose(feats.cpu().numpy().ravel(), pts[:, 3].cpu().numpy())


@pytest.mark.unit
def test_invalid_shape_raises() -> None:
    """Invalid input shapes should raise ValueError."""
    from opr.inference.preprocessing import PointCloudMinkPreprocessor

    pre = PointCloudMinkPreprocessor()
    with pytest.raises(ValueError):
        pre(np.zeros((5, 2), dtype=np.float32))
