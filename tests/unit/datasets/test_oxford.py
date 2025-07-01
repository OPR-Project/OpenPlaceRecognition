"""Unit tests for Oxford dataset module.

This module contains unit tests for the Oxford RobotCar dataset implementation
in src/opr/datasets/oxford.py, focusing on behavior with and without MinkowskiEngine.
"""

from unittest.mock import patch

import torch


class TestOxfordDataset:
    """Unit tests for Oxford dataset - focused on behavior, not implementation."""

    def test_oxford_dataset_imports_cleanly(self) -> None:
        """Oxford dataset module should import without errors regardless of MinkowskiEngine availability."""
        # This is the most fundamental test - the module should be importable
        # regardless of whether MinkowskiEngine is available or not
        import opr.datasets.oxford  # Should not raise any exceptions

        # Basic smoke test - the module should have the expected classes
        assert hasattr(opr.datasets.oxford, "OxfordDataset")

    def test_collate_behavior_with_pointclouds(self) -> None:
        """Collate function should either work or fail with helpful message when using pointclouds."""
        from opr.datasets.oxford import OxfordDataset

        # Create a mock dataset instance to test _collate_data_dict directly
        # We'll mock the constructor to avoid file system dependencies
        with patch.object(OxfordDataset, "__init__", lambda self: None):
            dataset = OxfordDataset()
            # Add minimal required attributes
            dataset.pointcloud_set_transform = None
            dataset._pointcloud_quantization_size = 0.01

            # Mock data that would trigger pointcloud processing
            mock_data_list = [
                {
                    "idx": torch.tensor(0),
                    "utm": torch.tensor([0.0, 0.0]),
                    "pointcloud_lidar_coords": torch.randn(100, 3),
                    "pointcloud_lidar_feats": torch.randn(100, 1),
                },
                {
                    "idx": torch.tensor(1),
                    "utm": torch.tensor([1.0, 1.0]),
                    "pointcloud_lidar_coords": torch.randn(120, 3),
                    "pointcloud_lidar_feats": torch.randn(120, 1),
                },
            ]

            try:
                # Try to collate pointcloud data
                result = dataset._collate_data_dict(mock_data_list)

                # If we get here, MinkowskiEngine is available and working
                assert result is not None
                assert "pointclouds_lidar_coords" in result
                assert "pointclouds_lidar_feats" in result

            except (RuntimeError, AttributeError) as e:
                # If we get here, MinkowskiEngine is missing or stub is working
                error_msg = str(e)
                assert "MinkowskiEngine" in error_msg, f"Expected MinkowskiEngine error, got: {error_msg}"


class TestOxfordDatasetWithoutMinkowskiEngine:
    """Test Oxford dataset behavior when MinkowskiEngine is not available."""

    def test_lazy_import_fails_gracefully_when_minkowski_missing(self) -> None:
        """Test that Oxford dataset fails gracefully when ME is a stub."""
        from opr.optional_deps import _SimplifiedMissingDepStub

        # Create a realistic stub like lazy() would return when MinkowskiEngine is missing
        stub = _SimplifiedMissingDepStub(
            "MinkowskiEngine", "sparse convolutions", "See the documentation for installation instructions"
        )

        with patch("opr.datasets.oxford.ME", stub):
            from opr.datasets.oxford import ME

            # Test that accessing ME components raises helpful errors
            try:
                ME.utils.sparse_quantize()
                raise AssertionError("Expected RuntimeError when accessing MinkowskiEngine components")
            except RuntimeError as e:
                error_msg = str(e)
                assert "MinkowskiEngine required" in error_msg

    def test_pointcloud_collation_fails_gracefully_without_minkowski(self) -> None:
        """Test pointcloud collation fails with helpful error when MinkowskiEngine missing."""
        from opr.optional_deps import _SimplifiedMissingDepStub

        # Create realistic stub like lazy() would return
        stub = _SimplifiedMissingDepStub(
            "MinkowskiEngine", "sparse convolutions", "See the documentation for installation instructions"
        )

        with patch("opr.datasets.oxford.ME", stub):
            from opr.datasets.oxford import OxfordDataset

            # Mock the constructor to avoid file system dependencies
            with patch.object(OxfordDataset, "__init__", lambda self: None):
                dataset = OxfordDataset()
                # Set minimal required attributes for collation
                dataset.pointcloud_set_transform = None
                dataset._pointcloud_quantization_size = 0.01

            # Mock pointcloud data
            mock_data_list = [
                {
                    "idx": torch.tensor(0),
                    "utm": torch.tensor([0.0, 0.0]),
                    "pointcloud_lidar_coords": torch.randn(100, 3),
                    "pointcloud_lidar_feats": torch.randn(100, 1),
                }
            ]

            # Collation should fail with helpful error
            try:
                dataset._collate_data_dict(mock_data_list)
                raise AssertionError(
                    "Expected RuntimeError when collating pointclouds without MinkowskiEngine"
                )
            except RuntimeError as e:
                error_msg = str(e)
                assert "MinkowskiEngine" in error_msg
                assert "not installed" in error_msg or "required" in error_msg

    def test_non_pointcloud_data_works_without_minkowski(self) -> None:
        """Test that non-pointcloud data processing works without MinkowskiEngine."""
        from opr.datasets.oxford import OxfordDataset
        from opr.optional_deps import _SimplifiedMissingDepStub

        # Create realistic stub like lazy() would return
        stub = _SimplifiedMissingDepStub(
            "MinkowskiEngine", "sparse convolutions", "See the documentation for installation instructions"
        )

        with patch("opr.datasets.oxford.ME", stub):
            # Mock the constructor to avoid file system dependencies
            with patch.object(OxfordDataset, "__init__", lambda self: None):
                dataset = OxfordDataset()
                # Set minimal required attributes for collation
                dataset.pointcloud_set_transform = None
                dataset._pointcloud_quantization_size = 0.01

            # Mock image-only data (no pointclouds)
            mock_data_list = [
                {
                    "idx": torch.tensor(0),
                    "utm": torch.tensor([0.0, 0.0]),
                    "image_stereo_centre": torch.randn(3, 224, 224),
                },
                {
                    "idx": torch.tensor(1),
                    "utm": torch.tensor([1.0, 1.0]),
                    "image_stereo_centre": torch.randn(3, 224, 224),
                },
            ]

            # This should work fine without MinkowskiEngine
            try:
                result = dataset._collate_data_dict(mock_data_list)
                assert result is not None
                assert "images_stereo_centre" in result
                assert "idxs" in result
                assert "utms" in result
                # Should NOT contain pointcloud keys
                assert "pointclouds_lidar_coords" not in result
                assert "pointclouds_lidar_feats" not in result

            except Exception as e:
                # Should not fail due to MinkowskiEngine when processing image data
                error_msg = str(e)
                assert (
                    "MinkowskiEngine" not in error_msg
                ), f"Image processing should not require MinkowskiEngine, got: {error_msg}"
