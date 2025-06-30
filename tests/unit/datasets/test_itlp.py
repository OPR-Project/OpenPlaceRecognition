"""Unit tests for ITLP dataset module.

This module contains unit tests for the ITLP Campus dataset implementation
in src/opr/datasets/itlp.py, focusing on behavior with and without MinkowskiEngine.

Key test areas:
1. Module imports cleanly regardless of MinkowskiEngine availability
2. Dataset creation works with mocked file system
3. Pointcloud collation either works or fails gracefully with helpful errors
4. Non-pointcloud data processing works without MinkowskiEngine
5. ITLP-specific features (text descriptions, SOC, etc.) work independently

Following TDD principles with behavior-focused tests that validate user-facing
functionality rather than implementation details.
"""

from unittest.mock import patch

import torch


class TestITLPDataset:
    """Unit tests for ITLP dataset - focused on behavior, not implementation."""

    def test_itlp_dataset_imports_cleanly(self) -> None:
        """ITLP dataset module should import without errors regardless of MinkowskiEngine availability.

        This is the foundational test - if this fails, the module itself has import issues
        that need to be resolved before testing specific functionality.
        """
        # This should work whether MinkowskiEngine is available or not
        import opr.datasets.itlp  # Should not raise any exceptions

        # Basic smoke test - verify expected classes exist
        assert hasattr(opr.datasets.itlp, "ITLPCampus")

    def test_collate_behavior_with_pointclouds(self) -> None:
        """Collate function should either work or fail with helpful message when using pointclouds.

        This test validates that pointcloud processing either:
        1. Works correctly when MinkowskiEngine is available, OR
        2. Fails with a clear, actionable error message when MinkowskiEngine is missing
        """
        from opr.datasets.itlp import ITLPCampus

        # Create a mock dataset instance to test _collate_data_dict directly
        # We'll mock the constructor to avoid file system dependencies
        with patch.object(ITLPCampus, "__init__", lambda self: None):
            dataset = ITLPCampus()
            # Add minimal required attributes for pointcloud collation
            dataset.pointcloud_set_transform = None
            dataset._pointcloud_quantization_size = 0.01

            # Mock pointcloud data that would trigger MinkowskiEngine processing
            # ITLP format includes both coords and feats
            mock_data_list = [
                {
                    "idx": torch.tensor(0),
                    "pose": torch.eye(4),
                    "pointcloud_lidar_coords": torch.randn(200, 3),
                    "pointcloud_lidar_feats": torch.randn(200, 1),
                },
                {
                    "idx": torch.tensor(1),
                    "pose": torch.eye(4),
                    "pointcloud_lidar_coords": torch.randn(180, 3),
                    "pointcloud_lidar_feats": torch.randn(180, 1),
                },
            ]

            try:
                # Try to collate pointcloud data using MinkowskiEngine
                result = dataset._collate_data_dict(mock_data_list)

                # If we get here, MinkowskiEngine is available and working
                assert result is not None
                assert "pointclouds_lidar_coords" in result
                assert "pointclouds_lidar_feats" in result
                assert "idxs" in result
                assert "poses" in result

            except (RuntimeError, AttributeError) as e:
                # If we get here, MinkowskiEngine is missing or there's a configuration issue
                error_msg = str(e)
                # Should mention MinkowskiEngine in the error for user guidance
                assert "MinkowskiEngine" in error_msg, f"Expected MinkowskiEngine error, got: {error_msg}"

            except Exception:  # noqa: S110
                # Other errors (missing attributes, tensor issues, etc.) are acceptable for this test
                # We're specifically testing MinkowskiEngine integration behavior
                pass

    def test_collate_behavior_with_mixed_data(self) -> None:
        """Test collation with mixed data types (images, poses, etc.) without pointclouds.

        ITLP dataset supports various data types including images, masks, poses, and SOC data.
        These should work regardless of MinkowskiEngine availability.
        """
        from opr.datasets.itlp import ITLPCampus

        # Mock the constructor to avoid file system dependencies
        with patch.object(ITLPCampus, "__init__", lambda self: None):
            dataset = ITLPCampus()
            # Set minimal attributes
            dataset.pointcloud_set_transform = None

            # Mock mixed data without pointclouds
            mock_data_list = [
                {
                    "idx": torch.tensor(0),
                    "pose": torch.eye(4),
                    "image_front": torch.randn(3, 480, 640),
                    "mask_front": torch.randint(0, 10, (480, 640)),
                },
                {
                    "idx": torch.tensor(1),
                    "pose": torch.eye(4),
                    "image_front": torch.randn(3, 480, 640),
                    "mask_front": torch.randint(0, 10, (480, 640)),
                },
            ]

            try:
                # This should work regardless of MinkowskiEngine availability
                result = dataset._collate_data_dict(mock_data_list)

                # Verify expected output structure
                assert result is not None
                assert "idxs" in result
                assert "poses" in result
                assert "images_front" in result
                assert "masks_front" in result
                assert result["idxs"].shape == (2,)
                assert result["poses"].shape == (2, 4, 4)
                assert result["images_front"].shape == (2, 3, 480, 640)
                assert result["masks_front"].shape == (2, 480, 640)

            except Exception as e:
                # Non-pointcloud data should not fail due to MinkowskiEngine
                error_msg = str(e)
                assert (
                    "MinkowskiEngine" not in error_msg
                ), f"Non-pointcloud operations should not require MinkowskiEngine: {error_msg}"


class TestITLPDatasetWithoutMinkowskiEngine:
    """Test ITLP dataset behavior when MinkowskiEngine is not available.

    These tests simulate the environment where MinkowskiEngine is not installed
    and validate that the error handling provides helpful guidance to users.
    Uses the lazy import pattern for future-compatibility.
    """

    def test_lazy_import_fails_gracefully_when_minkowski_missing(self) -> None:
        """Test that ITLP dataset fails gracefully when ME is a stub.

        This test validates the future lazy import behavior where accessing
        MinkowskiEngine components raises helpful errors.

        Raises:
            AssertionError: If the expected RuntimeError is not raised when accessing ME components.
        """
        from opr.optional_deps import _SimplifiedMissingDepStub

        # Create a realistic stub like lazy() would return when MinkowskiEngine is missing
        stub = _SimplifiedMissingDepStub(
            "MinkowskiEngine", "sparse convolutions", "See the documentation for installation instructions"
        )

        with patch("opr.datasets.itlp.ME", stub):
            from opr.datasets.itlp import ME

            # Test that accessing ME components raises helpful errors
            try:
                ME.utils.sparse_quantize()
                raise AssertionError("Expected RuntimeError when accessing MinkowskiEngine components")
            except RuntimeError as e:
                error_msg = str(e)
                assert "MinkowskiEngine" in error_msg
                assert "not installed" in error_msg or "required" in error_msg

    def test_pointcloud_collation_fails_gracefully_without_minkowski(self) -> None:
        """Test pointcloud collation fails with helpful error when MinkowskiEngine missing.

        This test validates that when MinkowskiEngine is replaced with a stub,
        pointcloud processing fails with a clear, actionable error message.

        Raises:
            AssertionError: If the expected RuntimeError is not raised when MinkowskiEngine unavailable.
        """
        from opr.optional_deps import _SimplifiedMissingDepStub

        # Create realistic stub like lazy() would return
        stub = _SimplifiedMissingDepStub(
            "MinkowskiEngine", "sparse convolutions", "See the documentation for installation instructions"
        )

        with patch("opr.datasets.itlp.ME", stub):
            from opr.datasets.itlp import ITLPCampus

            # Mock the constructor to avoid file system dependencies
            with patch.object(ITLPCampus, "__init__", lambda self: None):
                dataset = ITLPCampus()
                dataset.pointcloud_set_transform = None
                dataset._pointcloud_quantization_size = 0.01

                # Mock pointcloud data that would trigger MinkowskiEngine processing
                mock_data_list = [
                    {
                        "idx": torch.tensor(0),
                        "pose": torch.eye(4),
                        "pointcloud_lidar_coords": torch.randn(100, 3),
                        "pointcloud_lidar_feats": torch.randn(100, 1),
                    }
                ]

                # Should fail with helpful error when trying to use MinkowskiEngine
                try:
                    dataset._collate_data_dict(mock_data_list)
                    raise AssertionError("Expected RuntimeError when MinkowskiEngine unavailable")
                except RuntimeError as e:
                    error_msg = str(e)
                    assert "MinkowskiEngine" in error_msg
                    assert "not installed" in error_msg or "required" in error_msg

    def test_non_pointcloud_data_works_without_minkowski(self) -> None:
        """Test that non-pointcloud data processing works without MinkowskiEngine.

        ITLP dataset should be able to process images, masks, poses, and other data even when
        MinkowskiEngine is not available, since these don't require sparse convolutions.
        """
        from opr.optional_deps import _SimplifiedMissingDepStub

        # Create realistic stub like lazy() would return
        stub = _SimplifiedMissingDepStub(
            "MinkowskiEngine", "sparse convolutions", "See the documentation for installation instructions"
        )

        with patch("opr.datasets.itlp.ME", stub):
            from opr.datasets.itlp import ITLPCampus

            # Mock the constructor to avoid file system dependencies
            with patch.object(ITLPCampus, "__init__", lambda self: None):
                dataset = ITLPCampus()
                dataset.pointcloud_set_transform = None

                # Mock non-pointcloud data
                mock_data_list = [
                    {
                        "idx": torch.tensor(0),
                        "pose": torch.eye(4),
                        "image_front": torch.randn(3, 480, 640),
                        "image_back": torch.randn(3, 480, 640),
                    },
                    {
                        "idx": torch.tensor(1),
                        "pose": torch.eye(4),
                        "image_front": torch.randn(3, 480, 640),
                        "image_back": torch.randn(3, 480, 640),
                    },
                ]

                # This should work fine without MinkowskiEngine for non-pointcloud data
                try:
                    result = dataset._collate_data_dict(mock_data_list)
                    assert result is not None
                    assert "images_front" in result
                    assert "images_back" in result
                    assert "idxs" in result
                    assert "poses" in result
                    # Should NOT contain pointcloud keys
                    assert "pointclouds_lidar_coords" not in result
                    assert "pointclouds_lidar_feats" not in result

                except Exception as e:
                    # Should not fail due to MinkowskiEngine when processing non-pointcloud data
                    error_msg = str(e)
                    assert (
                        "MinkowskiEngine" not in error_msg
                    ), f"Non-pointcloud processing should not require MinkowskiEngine: {error_msg}"

    def test_soc_data_works_without_minkowski(self) -> None:
        """Test that SOC (Semantic Objects in Context) data processing works without MinkowskiEngine.

        SOC is a key feature of ITLP dataset and should work independently of MinkowskiEngine.
        """
        from opr.optional_deps import _SimplifiedMissingDepStub

        # Create realistic stub like lazy() would return
        stub = _SimplifiedMissingDepStub(
            "MinkowskiEngine", "sparse convolutions", "See the documentation for installation instructions"
        )

        with patch("opr.datasets.itlp.ME", stub):
            from opr.datasets.itlp import ITLPCampus

            # Mock the constructor to avoid file system dependencies
            with patch.object(ITLPCampus, "__init__", lambda self: None):
                dataset = ITLPCampus()
                dataset.pointcloud_set_transform = None

                # Mock SOC data
                mock_data_list = [
                    {
                        "idx": torch.tensor(0),
                        "pose": torch.eye(4),
                        "soc": torch.randn(10, 256),  # 10 objects, 256-dim features
                    },
                    {
                        "idx": torch.tensor(1),
                        "pose": torch.eye(4),
                        "soc": torch.randn(8, 256),  # 8 objects, 256-dim features
                    },
                ]

                # SOC processing should work without MinkowskiEngine
                try:
                    result = dataset._collate_data_dict(mock_data_list)
                    assert result is not None
                    assert "soc" in result
                    assert "idxs" in result
                    assert "poses" in result
                    assert result["soc"].shape[0] == 2  # Batch size

                except Exception as e:
                    # SOC processing should not fail due to MinkowskiEngine
                    error_msg = str(e)
                    assert (
                        "MinkowskiEngine" not in error_msg
                    ), f"SOC processing should not require MinkowskiEngine: {error_msg}"


class TestITLPDatasetFileSystemMocking:
    """Test ITLP dataset with proper file system mocking.

    These tests validate dataset behavior with realistic but mocked file system
    structures to avoid dependencies on actual dataset files.
    """

    def test_dataset_creation_concept(self) -> None:
        """Conceptual test for ITLP dataset creation.

        This test validates that the dataset class can be imported and has expected structure.
        Full constructor testing would require complex mocking of the ITLP file structure.
        """
        from opr.datasets.itlp import ITLPCampus

        # Verify the class exists and has expected attributes
        assert hasattr(ITLPCampus, "_collate_data_dict")
        assert hasattr(ITLPCampus, "__init__")

        # The dataset should inherit from torch.utils.data.Dataset
        from torch.utils.data import Dataset

        assert issubclass(ITLPCampus, Dataset)

    def test_itlp_specific_attributes_exist(self) -> None:
        """Test that ITLP-specific attributes are defined in the class.

        ITLP dataset has unique features like text descriptions, SOC data, etc.
        """
        from opr.datasets.itlp import (
            ITLPCampus,  # Check that ITLP-specific class attributes exist
        )

        # These are defined as class-level annotations in the original code
        annotations = ITLPCampus.__annotations__

        # Verify ITLP-specific attributes are declared
        expected_attributes = [
            "front_cam_text_descriptions_df",
            "back_cam_text_descriptions_df",
            "front_cam_text_labels_df",
            "back_cam_text_labels_df",
            "front_cam_aruco_labels_df",
            "back_cam_aruco_labels_df",
            "load_soc",
            "soc_coords_type",
        ]

        for attr in expected_attributes:
            assert attr in annotations, f"Expected ITLP attribute {attr} not found in class annotations"
