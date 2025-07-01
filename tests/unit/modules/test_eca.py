"""Unit tests for ECA module.

This module contains unit tests for the Efficient Channel Attention (ECA)
implementation in src/opr/modules/eca.py.
"""

from unittest.mock import patch


class TestECAModule:
    """Unit tests for ECA module - focused on behavior, not implementation."""

    def test_eca_module_imports_cleanly(self) -> None:
        """ECA module should import without errors regardless of MinkowskiEngine availability."""
        # This is the most fundamental test - the module should be importable
        # regardless of whether MinkowskiEngine is available or not
        import opr.modules.eca  # Should not raise any exceptions

        # Basic smoke test - the module should have the expected classes
        assert hasattr(opr.modules.eca, "MinkECALayer")
        assert hasattr(opr.modules.eca, "MinkECABasicBlock")

    def test_eca_layer_behavior(self) -> None:
        """ECA layer should either work or fail with helpful message."""
        from opr.modules.eca import MinkECALayer

        try:
            # Attempt to create an ECA layer
            layer = MinkECALayer(channels=64)
            # If we get here, MinkowskiEngine is available and working
            assert layer is not None
            assert hasattr(layer, "avg_pool")
            assert hasattr(layer, "broadcast_mul")

        except (RuntimeError, AttributeError) as e:
            # If we get here, MinkowskiEngine is missing or stub is working
            error_msg = str(e)
            assert "MinkowskiEngine" in error_msg, f"Expected MinkowskiEngine error, got: {error_msg}"

    def test_basic_block_behavior(self) -> None:
        """ECA BasicBlock should either work or fail with helpful message."""
        from opr.modules.eca import MinkECABasicBlock

        try:
            # Attempt to create an ECA basic block
            block = MinkECABasicBlock(inplanes=32, planes=64)
            # If we get here, MinkowskiEngine is available and working
            assert block is not None
            assert hasattr(block, "eca")

        except (RuntimeError, AttributeError) as e:
            # If we get here, MinkowskiEngine is missing or dependencies failed
            error_msg = str(e)
            assert "MinkowskiEngine" in error_msg, f"Expected MinkowskiEngine error, got: {error_msg}"


class TestECAModuleWithoutMinkowskiEngine:
    """Test ECA module behavior when MinkowskiEngine is not available."""

    def test_lazy_import_fails_gracefully_when_minkowski_missing(self) -> None:
        """Test that ECA components fail gracefully when ME is a stub."""
        from opr.optional_deps import _SimplifiedMissingDepStub

        # Create a realistic stub like lazy() would return when MinkowskiEngine is missing
        stub = _SimplifiedMissingDepStub(
            "MinkowskiEngine", "sparse convolutions", "See the documentation for installation instructions"
        )

        with patch("opr.modules.eca.ME", stub):
            from opr.modules.eca import ME

            # Test that accessing ME components raises helpful errors
            try:
                ME.MinkowskiGlobalPooling()
                raise AssertionError("Expected RuntimeError when accessing MinkowskiEngine components")
            except RuntimeError as e:
                error_msg = str(e)
                assert "MinkowskiEngine required" in error_msg
                assert "sparse convolutions" in error_msg

    def test_eca_layer_creation_fails_gracefully_without_minkowski(self) -> None:
        """Test MinkECALayer creation fails with helpful error when MinkowskiEngine missing."""
        from opr.optional_deps import _SimplifiedMissingDepStub

        # Create a realistic stub like lazy() would return
        stub = _SimplifiedMissingDepStub(
            "MinkowskiEngine", "sparse convolutions", "See the documentation for installation instructions"
        )

        with patch("opr.modules.eca.ME", stub):
            from opr.modules.eca import MinkECALayer

            # Creating ECA layer should fail with helpful error when trying to access ME components
            try:
                MinkECALayer(channels=64)
                raise AssertionError(
                    "Expected RuntimeError when creating MinkECALayer without MinkowskiEngine"
                )
            except RuntimeError as e:
                error_msg = str(e)
                assert "MinkowskiEngine required" in error_msg

    def test_basic_block_creation_fails_gracefully_without_minkowski(self) -> None:
        """Test MinkECABasicBlock creation fails with helpful error when MinkowskiEngine missing."""
        from opr.optional_deps import _SimplifiedMissingDepStub

        # Create realistic stubs for both ME and BasicBlock
        me_stub = _SimplifiedMissingDepStub(
            "MinkowskiEngine", "sparse convolutions", "See the documentation for installation instructions"
        )

        # Mock the BasicBlock to simulate the fallback BasicBlock that raises on __init__
        class MockBasicBlock:
            def __init__(self, *args: object, **kwargs: object) -> None:
                raise RuntimeError(
                    "MinkowskiEngine required for sparse convolutions.\n"
                    "See the documentation for installation instructions"
                )

        with patch("opr.modules.eca.ME", me_stub):
            with patch("opr.modules.eca.BasicBlock", MockBasicBlock):
                from opr.modules.eca import MinkECABasicBlock

                # Creating BasicBlock should fail with helpful error
                try:
                    MinkECABasicBlock(inplanes=32, planes=64)
                    raise AssertionError(
                        "Expected RuntimeError when creating MinkECABasicBlock without MinkowskiEngine"
                    )
                except RuntimeError as e:
                    error_msg = str(e)
                    assert "MinkowskiEngine required" in error_msg
