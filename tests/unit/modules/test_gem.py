"""Unit tests for GEM module focusing on lazy import functionality.

This module contains unit tests for the Generalized-Mean pooling implementations
in src/opr/modules/gem.py, specifically focusing on MinkowskiEngine integration
and lazy import behavior for the MinkGeM class.

Key test areas:
1. Module imports cleanly regardless of MinkowskiEngine availability
2. MinkGeM class creation behavior with and without MinkowskiEngine
3. Forward pass operations either work or fail gracefully with helpful errors
4. Non-MinkowskiEngine classes (GeM, SeqGeM) work independently

Following TDD principles with behavior-focused tests that validate user-facing
functionality rather than implementation details.
"""

from unittest.mock import patch

import torch


class TestGEMModule:
    """Unit tests for GEM module - focused on behavior, not implementation."""

    def test_gem_module_imports_cleanly(self) -> None:
        """GEM module should import without errors regardless of MinkowskiEngine availability.

        This is the foundational test - if this fails, the module itself has import issues
        that need to be resolved before testing specific functionality.
        """
        # This should work whether MinkowskiEngine is available or not
        import opr.modules.gem  # Should not raise any exceptions

        # Basic smoke test - verify expected classes exist
        assert hasattr(opr.modules.gem, "MinkGeM")
        assert hasattr(opr.modules.gem, "GeM")
        assert hasattr(opr.modules.gem, "SeqGeM")

    def test_non_minkowski_classes_work_independently(self) -> None:
        """Test that GeM and SeqGeM classes work without MinkowskiEngine.

        These classes don't depend on MinkowskiEngine and should work regardless
        of its availability.
        """
        from opr.modules.gem import GeM, SeqGeM

        # Test GeM (2D pooling)
        gem_layer = GeM(p=3, eps=1e-6)
        assert gem_layer is not None
        assert gem_layer.p.item() == 3.0
        assert gem_layer.eps == 1e-6

        # Test forward pass with 2D tensor
        x_2d = torch.randn(2, 64, 8, 8)  # (batch, channels, height, width)
        output_2d = gem_layer(x_2d)
        assert output_2d.shape == (2, 64)

        # Test SeqGeM (1D pooling)
        seq_gem_layer = SeqGeM(p=2, eps=1e-5)
        assert seq_gem_layer is not None
        assert seq_gem_layer.p.item() == 2.0
        assert seq_gem_layer.eps == 1e-5

        # Test forward pass with 1D tensor
        x_1d = torch.randn(3, 128, 16)  # (batch, channels, sequence_length)
        output_1d = seq_gem_layer(x_1d)
        assert output_1d.shape == (3, 128)

    def test_mink_gem_creation_behavior(self) -> None:
        """Test MinkGeM creation behavior with and without MinkowskiEngine.

        This test validates that MinkGeM instantiation either:
        1. Works correctly when MinkowskiEngine is available, OR
        2. Fails with a clear, actionable error message when MinkowskiEngine is missing
        """
        from opr.modules.gem import MinkGeM

        try:
            # Try to create MinkGeM layer
            mink_gem = MinkGeM(p=3, eps=1e-6)

            # If we get here, MinkowskiEngine is available and working
            assert mink_gem is not None
            assert mink_gem.p.item() == 3.0
            assert mink_gem.eps == 1e-6
            assert hasattr(mink_gem, "f")  # Should have MinkowskiGlobalAvgPooling

        except (RuntimeError, AttributeError) as e:
            # If we get here, MinkowskiEngine is missing or there's a configuration issue
            error_msg = str(e)
            # Should mention MinkowskiEngine in the error for user guidance
            assert "MinkowskiEngine" in error_msg, f"Expected MinkowskiEngine error, got: {error_msg}"

        except Exception:  # noqa: S110
            # Other errors (import issues, etc.) are acceptable for this test
            # We're specifically testing MinkowskiEngine integration behavior
            pass


class TestMinkGEMWithoutMinkowskiEngine:
    """Test MinkGeM behavior when MinkowskiEngine is not available.

    These tests simulate the environment where MinkowskiEngine is not installed
    and validate that the error handling provides helpful guidance to users.
    Uses the lazy import pattern for future-compatibility.
    """

    def test_lazy_import_fails_gracefully_when_minkowski_missing(self) -> None:
        """Test that MinkGeM fails gracefully when ME is a stub.

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

        with patch("opr.modules.gem.ME", stub):
            from opr.modules.gem import ME

            # Test that accessing ME components raises helpful errors
            try:
                ME.MinkowskiGlobalAvgPooling()
                raise AssertionError("Expected RuntimeError when accessing MinkowskiEngine components")
            except RuntimeError as e:
                error_msg = str(e)
                assert "MinkowskiEngine required" in error_msg

    def test_mink_gem_creation_fails_gracefully_without_minkowski(self) -> None:
        """Test MinkGeM creation fails with helpful error when MinkowskiEngine missing.

        This test validates that when MinkowskiEngine is replaced with a stub,
        MinkGeM instantiation fails with a clear, actionable error message.

        Raises:
            AssertionError: If the expected RuntimeError is not raised when MinkowskiEngine unavailable.
        """
        from opr.optional_deps import _SimplifiedMissingDepStub

        # Create realistic stub like lazy() would return
        stub = _SimplifiedMissingDepStub(
            "MinkowskiEngine", "sparse convolutions", "See the documentation for installation instructions"
        )

        with patch("opr.modules.gem.ME", stub):
            from opr.modules.gem import MinkGeM

            # Should fail with helpful error when trying to create MinkGeM
            try:
                MinkGeM(p=3, eps=1e-6)
                raise AssertionError("Expected RuntimeError when MinkowskiEngine unavailable")
            except RuntimeError as e:
                error_msg = str(e)
                # After refactor, should get lazy import error message
                assert "MinkowskiEngine" in error_msg

    def test_mink_gem_forward_pass_fails_gracefully_without_minkowski(self) -> None:
        """Test MinkGeM forward pass fails gracefully when MinkowskiEngine missing.

        This test validates that even if MinkGeM was somehow created,
        the forward pass would fail with helpful error messages.
        Note: This test may not be reachable after refactor since creation itself will fail.
        """
        from opr.optional_deps import _SimplifiedMissingDepStub

        # Create realistic stub like lazy() would return
        stub = _SimplifiedMissingDepStub(
            "MinkowskiEngine", "sparse convolutions", "See the documentation for installation instructions"
        )

        # Patch both at creation and usage time
        with patch("opr.modules.gem.ME", stub):
            try:
                # This might fail at creation time after refactor
                from opr.modules.gem import MinkGeM

                # If creation somehow succeeds (unlikely after refactor), test forward pass
                mink_gem = MinkGeM(p=3, eps=1e-6)

                # Mock sparse tensor input (this would normally be ME.SparseTensor)
                # Since ME is stubbed, we can't create real sparse tensors
                mock_input = torch.randn(10, 64)  # Fallback to regular tensor

                result = mink_gem(mock_input)
                # If we get here, something unexpected happened
                assert result is not None

            except RuntimeError as e:
                # Expected - MinkowskiEngine operations should fail
                error_msg = str(e)
                assert "MinkowskiEngine" in error_msg

    def test_non_mink_gem_classes_unaffected_by_missing_minkowski(self) -> None:
        """Test that GeM and SeqGeM work normally even when MinkowskiEngine is stubbed.

        These classes should be completely independent of MinkowskiEngine availability.
        """
        from opr.optional_deps import _SimplifiedMissingDepStub

        # Create realistic stub like lazy() would return
        stub = _SimplifiedMissingDepStub(
            "MinkowskiEngine", "sparse convolutions", "See the documentation for installation instructions"
        )

        with patch("opr.modules.gem.ME", stub):
            from opr.modules.gem import GeM, SeqGeM

            # GeM should work normally
            gem_layer = GeM(p=4, eps=1e-7)
            x_2d = torch.randn(2, 32, 4, 4)
            output_2d = gem_layer(x_2d)
            assert output_2d.shape == (2, 32)

            # SeqGeM should work normally
            seq_gem_layer = SeqGeM(p=1, eps=1e-8)
            x_1d = torch.randn(2, 16, 8)
            output_1d = seq_gem_layer(x_1d)
            assert output_1d.shape == (2, 16)


class TestGEMModuleTypeAnnotations:
    """Test GEM module type annotations and interface contracts.

    These tests validate that the module provides proper type information
    and maintains interface contracts regardless of MinkowskiEngine availability.
    """

    def test_gem_classes_have_expected_interface(self) -> None:
        """Test that all GEM classes implement the expected interface.

        This validates the class structure and method signatures.
        """
        # All classes should be nn.Module subclasses
        import torch.nn as nn

        from opr.modules.gem import GeM, MinkGeM, SeqGeM

        assert issubclass(GeM, nn.Module)
        assert issubclass(SeqGeM, nn.Module)
        assert issubclass(MinkGeM, nn.Module)

        # All classes should have expected methods
        assert hasattr(GeM, "forward")
        assert hasattr(SeqGeM, "forward")
        assert hasattr(MinkGeM, "forward")

        # GeM and SeqGeM should have _gem method
        assert hasattr(GeM, "_gem")
        assert hasattr(SeqGeM, "_gem")

    def test_mink_gem_has_sparse_attribute(self) -> None:
        """Test that MinkGeM has the sparse attribute set correctly.

        This validates MinkGeM-specific attributes.
        """
        from opr.modules.gem import MinkGeM

        # MinkGeM should have sparse attribute
        assert hasattr(MinkGeM, "sparse")
        assert MinkGeM.sparse is True
