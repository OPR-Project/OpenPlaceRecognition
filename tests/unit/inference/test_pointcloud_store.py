"""Tests for PointCloudStore loading behavior."""

from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.mark.unit
def test_load_nan_returns_empty(tmp_path: Path) -> None:
    """When path is NaN/None, return an empty [0,3] tensor."""
    from opr.inference.io import PointCloudStore

    store = PointCloudStore(root_dir=tmp_path)
    pts = store.load(np.nan)
    assert isinstance(pts, torch.Tensor)
    assert pts.dtype == torch.float32
    assert pts.shape == (0, 3)


@pytest.mark.unit
def test_load_bin_with_3_properties(tmp_path: Path) -> None:
    """Load .bin with 3 props (x,y,z) and preserve values."""
    from opr.inference.io import PointCloudStore

    rel = Path("scans/000001.bin")
    abs_path = (tmp_path / rel).resolve()
    abs_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    data.tofile(abs_path)

    store = PointCloudStore(root_dir=tmp_path, default_num_properties=3)
    pts = store.load(str(rel))

    assert pts.shape == (2, 3)
    np.testing.assert_allclose(pts.numpy(), data)


@pytest.mark.unit
def test_load_bin_with_4_properties_truncates_to_xyz(tmp_path: Path) -> None:
    """Load .bin with 4 props (x,y,z,intensity) and slice to first 3 dims."""
    from opr.inference.io import PointCloudStore

    rel = Path("scans/000002.bin")
    abs_path = (tmp_path / rel).resolve()
    abs_path.parent.mkdir(parents=True, exist_ok=True)

    data4 = np.array([[1, 2, 3, 9], [7, 8, 9, 0]], dtype=np.float32)
    data4.tofile(abs_path)

    store = PointCloudStore(root_dir=tmp_path, default_num_properties=4)
    pts = store.load(str(rel))

    assert pts.shape == (2, 3)
    np.testing.assert_allclose(pts.numpy(), data4[:, :3])


@pytest.mark.unit
def test_missing_file_raises(tmp_path: Path) -> None:
    """Nonexistent file should raise FileNotFoundError."""
    from opr.inference.io import PointCloudStore

    store = PointCloudStore(root_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        store.load("scans/missing.bin")


@pytest.mark.unit
def test_unsupported_extension_raises(tmp_path: Path) -> None:
    """Unsupported file extension should raise ValueError."""
    from opr.inference.io import PointCloudStore

    rel = Path("scans/000003.xyz")
    abs_path = (tmp_path / rel).resolve()
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_bytes(b"xyz")

    store = PointCloudStore(root_dir=tmp_path)
    with pytest.raises(ValueError):
        store.load(str(rel))


@pytest.mark.unit
def test_load_pcd_if_open3d_available(tmp_path: Path) -> None:
    """If open3d is available, .pcd loads and returns Nx3 tensor."""
    o3d = pytest.importorskip("open3d")
    from opr.inference.io import PointCloudStore

    rel = Path("scans/000004.pcd")
    abs_path = (tmp_path / rel).resolve()
    abs_path.parent.mkdir(parents=True, exist_ok=True)

    pcd = o3d.geometry.PointCloud()
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.io.write_point_cloud(str(abs_path), pcd)

    store = PointCloudStore(root_dir=tmp_path)
    loaded = store.load(str(rel))
    assert loaded.shape == (2, 3)
    np.testing.assert_allclose(loaded.numpy(), pts)
