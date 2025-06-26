"""Mock fixtures for testing optional dependencies.

This module provides mock objects and stubs for testing code that depends
on optional packages like MinkowskiEngine, PaddleOCR, etc.
"""

from typing import Any, Callable
from unittest.mock import MagicMock, Mock

import pytest


@pytest.fixture
def mock_minkowski_engine() -> Mock:
    """Mock MinkowskiEngine module for testing.

    Returns:
        Mock object that behaves like MinkowskiEngine.
    """
    mock_me = Mock()
    mock_me.SparseTensor = Mock()
    mock_me.MinkowskiGlobalPooling = Mock()
    mock_me.MinkowskiConvolution = Mock()
    return mock_me


@pytest.fixture
def mock_paddle_ocr() -> Mock:
    """Mock PaddleOCR for testing.

    Returns:
        Mock object that behaves like PaddleOCR.
    """
    mock_ocr = Mock()
    mock_ocr.ocr = Mock(return_value=[])
    return mock_ocr


@pytest.fixture
def mock_import_error() -> Callable[..., None]:
    """Fixture that raises ImportError when called.

    Useful for testing fallback behavior when optional dependencies are missing.

    Returns:
        Callable that raises ImportError.
    """

    def raise_import_error(*args: Any, **kwargs: Any) -> None:
        raise ImportError("Mocked import failure")

    return raise_import_error


@pytest.fixture
def mock_successful_import() -> Callable[..., MagicMock]:
    """Fixture that returns a mock module when called.

    Useful for testing successful import scenarios.

    Returns:
        Callable that returns a MagicMock object.
    """

    def return_mock_module(*args: Any, **kwargs: Any) -> MagicMock:
        return MagicMock()

    return return_mock_module


__all__ = [
    "mock_minkowski_engine",
    "mock_paddle_ocr",
    "mock_import_error",
    "mock_successful_import",
]
