"""Test fixtures for OpenPlaceRecognition.

This package contains shared fixtures and utilities for testing.
Import fixtures from specific modules as needed.
"""

from .minkowski_fixtures import (
    minkowski_available,
    mock_sparse_tensor_class,
    sample_sparse_tensor,
    skip_if_no_minkowski,
)

# Make common fixtures easily accessible
from .mock_fixtures import (
    mock_import_error,
    mock_minkowski_engine,
    mock_paddle_ocr,
    mock_successful_import,
)

__all__ = [
    # Mock fixtures
    "mock_minkowski_engine",
    "mock_paddle_ocr",
    "mock_import_error",
    "mock_successful_import",
    # MinkowskiEngine fixtures
    "skip_if_no_minkowski",
    "minkowski_available",
    "sample_sparse_tensor",
    "mock_sparse_tensor_class",
]
