"""MinkowskiEngine-specific test fixtures.

This module provides fixtures for testing code that uses MinkowskiEngine,
including both mock and real MinkowskiEngine objects when available.
"""

from typing import Any

import pytest

from opr.optional_deps import has_minkowski


@pytest.fixture
def skip_if_no_minkowski() -> None:
    """Skip test if MinkowskiEngine is not available.

    Use this fixture for tests that require actual MinkowskiEngine functionality.
    """
    if not has_minkowski():
        pytest.skip("MinkowskiEngine not available")


@pytest.fixture
def minkowski_available() -> bool:
    """Check if MinkowskiEngine is available.

    Returns:
        True if MinkowskiEngine is available and importable.
    """
    return has_minkowski()


@pytest.fixture
def sample_sparse_tensor() -> Any:
    """Create a sample sparse tensor for testing.

    Returns:
        MinkowskiEngine SparseTensor if available, otherwise skips test.
    """
    if not has_minkowski():
        pytest.skip("MinkowskiEngine not available")

    import MinkowskiEngine as ME
    import torch

    # Create simple sparse tensor
    coords = torch.IntTensor([[0, 0, 0], [0, 1, 1], [1, 0, 1]])
    feats = torch.FloatTensor([[1], [2], [3]])

    return ME.SparseTensor(feats, coords)


@pytest.fixture
def mock_sparse_tensor_class() -> type:
    """Mock SparseTensor class for testing without MinkowskiEngine.

    Returns:
        Mock class that behaves like ME.SparseTensor.
    """
    from unittest.mock import Mock

    mock_class = Mock()
    mock_class.return_value = Mock()
    mock_class.return_value.F = Mock()  # Features
    mock_class.return_value.C = Mock()  # Coordinates

    return mock_class


__all__ = [
    "skip_if_no_minkowski",
    "minkowski_available",
    "sample_sparse_tensor",
    "mock_sparse_tensor_class",
]
