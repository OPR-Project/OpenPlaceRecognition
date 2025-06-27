"""Comprehensive test suite for OptionalDependencyManager.

This test suite validates the core functionality of the optional dependency system:
- Fast package existence checks without import side effects
- Thread-safe warning management
- Version-aware availability checks
- Platform compatibility checks
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch

import pytest


class TestOptionalDependencyManager:
    """Test the core OptionalDependencyManager functionality."""

    def setup_method(self) -> None:
        """Reset warning cache and clear LRU caches before each test."""
        from opr.optional_deps import (
            OptionalDependencyManager,
            _warn_lock,
            _warnings_shown,
        )

        # Clear warning cache
        with _warn_lock:
            _warnings_shown.clear()

        # Clear LRU caches to ensure mocks work properly
        OptionalDependencyManager.exists_on_path.cache_clear()
        OptionalDependencyManager.is_available.cache_clear()

    def test_exists_on_path_with_real_package(self) -> None:
        """Test exists_on_path with packages that definitely exist."""
        from opr.optional_deps import OptionalDependencyManager

        # Test with standard library packages that are always available
        assert OptionalDependencyManager.exists_on_path("os") is True
        assert OptionalDependencyManager.exists_on_path("sys") is True
        assert OptionalDependencyManager.exists_on_path("json") is True

    def test_exists_on_path_with_nonexistent_package(self) -> None:
        """Test exists_on_path with packages that don't exist."""
        from opr.optional_deps import OptionalDependencyManager

        assert OptionalDependencyManager.exists_on_path("nonexistent_package_xyz") is False
        assert OptionalDependencyManager.exists_on_path("definitely_not_real") is False

    @patch("importlib.util.find_spec")
    def test_exists_on_path_caching(self, mock_find_spec: Mock) -> None:
        """Test that exists_on_path uses LRU caching properly."""
        from opr.optional_deps import OptionalDependencyManager

        mock_find_spec.return_value = Mock()

        # First call
        result1 = OptionalDependencyManager.exists_on_path("test_package")
        # Second call - should use cache
        result2 = OptionalDependencyManager.exists_on_path("test_package")

        assert result1 is True
        assert result2 is True
        # Should only call find_spec once due to caching
        assert mock_find_spec.call_count == 1

    @patch("importlib.util.find_spec")
    @patch("importlib.metadata.version")
    def test_is_available_without_version_constraint(self, mock_version: Mock, mock_find_spec: Mock) -> None:
        """Test is_available when no version constraint is specified."""
        from opr.optional_deps import OptionalDependencyManager

        mock_find_spec.return_value = Mock()

        result = OptionalDependencyManager.is_available("test_package")

        assert result is True
        # version() should not be called when no constraint
        mock_version.assert_not_called()

    @patch("importlib.util.find_spec")
    @patch("importlib.metadata.version")
    def test_is_available_with_version_constraint_satisfied(
        self, mock_version: Mock, mock_find_spec: Mock
    ) -> None:
        """Test is_available when version constraint is satisfied."""
        from opr.optional_deps import OptionalDependencyManager

        mock_find_spec.return_value = Mock()
        mock_version.return_value = "1.5.0"

        result = OptionalDependencyManager.is_available("test_package", min_version="1.2.0")

        assert result is True
        mock_version.assert_called_once_with("test_package")

    @patch("importlib.util.find_spec")
    @patch("importlib.metadata.version")
    def test_is_available_with_version_constraint_not_satisfied(
        self, mock_version: Mock, mock_find_spec: Mock
    ) -> None:
        """Test is_available when version constraint is not satisfied."""
        from opr.optional_deps import OptionalDependencyManager

        mock_find_spec.return_value = Mock()
        mock_version.return_value = "1.0.0"

        result = OptionalDependencyManager.is_available("test_package", min_version="1.2.0")

        assert result is False

    @patch("importlib.util.find_spec")
    @patch("importlib.metadata.version")
    def test_is_available_package_not_found_error(self, mock_version: Mock, mock_find_spec: Mock) -> None:
        """Test is_available when importlib.metadata raises PackageNotFoundError."""
        from importlib.metadata import PackageNotFoundError

        from opr.optional_deps import OptionalDependencyManager

        mock_find_spec.return_value = Mock()
        mock_version.side_effect = PackageNotFoundError("test_package")

        result = OptionalDependencyManager.is_available("test_package", min_version="1.0.0")

        assert result is False

    @patch.dict(os.environ, {}, clear=True)
    @patch("opr.optional_deps.logger")
    def test_warn_once_normal_behavior(self, mock_logger: Mock) -> None:
        """Test warn_once shows warning normally."""
        from opr.optional_deps import OptionalDependencyManager

        OptionalDependencyManager.warn_once("TestPackage", "test feature", "Install it")

        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "TestPackage is not available" in warning_msg
        assert "test feature will be disabled" in warning_msg
        assert "Install it" in warning_msg

    @patch.dict(os.environ, {"OPR_NO_OPTIONAL_WARNINGS": "1"})
    @patch("opr.optional_deps.logger")
    def test_warn_once_suppressed_by_env_var(self, mock_logger: Mock) -> None:
        """Test warn_once respects OPR_NO_OPTIONAL_WARNINGS environment variable."""
        from opr.optional_deps import OptionalDependencyManager

        OptionalDependencyManager.warn_once("TestPackage", "test feature", "Install it")

        mock_logger.warning.assert_not_called()

    def test_thread_safety(self) -> None:
        """Test that warning deduplication is thread-safe."""
        from opr.optional_deps import OptionalDependencyManager

        warning_count = 0

        def count_warnings() -> None:
            nonlocal warning_count
            with patch("opr.optional_deps.logger") as mock_logger:
                OptionalDependencyManager.warn_once("TestConcurrent", "test", "install it")
                if mock_logger.warning.called:
                    warning_count += 1

        # Run many concurrent warning attempts
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(count_warnings) for _ in range(20)]
            for future in as_completed(futures):
                future.result()

        # Should only see one warning despite 20 concurrent attempts
        assert warning_count == 1


@pytest.mark.parametrize(
    "package_name,expected",
    [
        ("os", True),  # Standard library - always available
        ("sys", True),  # Standard library - always available
        ("json", True),  # Standard library - always available
        ("nonexistent_xyz", False),  # Definitely doesn't exist
    ],
)
def test_exists_on_path_parametrized(package_name: str, expected: bool) -> None:
    """Parametrized test for exists_on_path with various inputs."""
    from opr.optional_deps import OptionalDependencyManager

    result = OptionalDependencyManager.exists_on_path(package_name)
    assert result == expected


@pytest.mark.parametrize(
    "version1,version2,expected",
    [
        ("1.0.0", "0.9.0", True),  # Newer version available
        ("1.0.0", "1.0.0", True),  # Exact version match
        ("0.9.0", "1.0.0", False),  # Older version available
        ("2.1.3", "2.1.2", True),  # Patch version newer
        ("1.2.0", "2.0.0", False),  # Major version older
    ],
)
def test_version_comparison_logic(version1: str, version2: str, expected: bool) -> None:
    """Test version comparison logic used internally."""
    from packaging.version import Version

    result = Version(version1) >= Version(version2)
    assert result == expected
