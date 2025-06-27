"""Optional dependency management for OpenPlaceRecognition.

This module provides centralized, thread-safe handling of optional dependencies
with fast existence checks and version constraints.

Public API:
    - has_minkowski() -> bool
    - require_minkowski(feature: str) -> bool
    - has_paddleocr() -> bool
    - require_paddleocr(feature: str) -> bool

Environment Variables:
    - OPR_NO_OPTIONAL_WARNINGS=1: Suppress all optional dependency warnings
    - OPR_OPTIONAL_WARNINGS_ONCE=1: Show warnings only once (default)
"""

import importlib
import importlib.metadata as im
import importlib.util
import os
import threading
from functools import lru_cache
from typing import Set

from loguru import logger

# Thread-safe warning cache
_warn_lock = threading.Lock()
_warnings_shown: Set[str] = set()

# Standard message for missing optional dependencies
_INSTALLATION_DOCS_MESSAGE = "See the documentation for installation instructions"


class _MissingDepStub:
    """Stub class that raises informative errors when accessed."""

    def __init__(self, package: str, feature: str, install_cmd: str) -> None:
        """Initialize the missing dependency stub.

        Args:
            package: Name of the missing package.
            feature: Description of the feature that requires this package.
            install_cmd: Installation command suggestion.
        """
        self._package = package
        self._feature = feature
        self._install_cmd = install_cmd

    def __getattr__(self, name: str) -> None:
        """Raise error when accessing any attribute."""
        raise RuntimeError(f"{self._package} is required for {self._feature}. {self._install_cmd}")

    def __call__(self, *args: object, **kwargs: object) -> None:
        """Raise error when calling the stub."""
        raise RuntimeError(f"{self._package} is required for {self._feature}. {self._install_cmd}")


class OptionalDependencyManager:
    """Centralized manager for optional dependencies with fast checks and version constraints."""

    @classmethod
    @lru_cache(maxsize=None)
    def exists_on_path(cls, package: str) -> bool:
        """Fast check if package exists on sys.path without importing it.

        This avoids expensive side effects like CUDA initialization that
        occur during actual import of compiled libraries.

        Args:
            package: Name of the package to check.

        Returns:
            True if package exists on path, False otherwise.
        """
        return importlib.util.find_spec(package) is not None

    @classmethod
    @lru_cache(maxsize=None)
    def is_available(cls, package: str, min_version: str | None = None) -> bool:
        """Check if package is available and meets version requirements.

        Args:
            package: Name of the package to check.
            min_version: Minimum required version string, if any.

        Returns:
            True if package is available and meets version requirements.
        """
        if not cls.exists_on_path(package):
            return False

        if min_version is None:
            return True

        try:
            from packaging.version import Version

            installed_version = im.version(package)
            return Version(installed_version) >= Version(min_version)
        except im.PackageNotFoundError:
            return False
        except Exception:
            # Handle malformed version strings gracefully
            logger.debug(f"Could not parse version for {package}")
            return True

    @classmethod
    def is_importable(cls, package: str) -> bool:
        """Check if package can actually be imported (handles platform issues).

        Args:
            package: Name of the package to check.

        Returns:
            True if package can be imported successfully.
        """
        if not cls.exists_on_path(package):
            return False

        try:
            importlib.import_module(package)
            return True
        except ImportError as e:
            logger.debug(f"Package {package} found but not importable: {e}")
            return False

    @classmethod
    def warn_once(
        cls,
        package: str,
        feature: str,
        suggestion: str = "",
        platform_issue: bool = False,
    ) -> None:
        """Show a warning only once, with thread safety.

        Args:
            package: Name of the missing package.
            feature: Description of the feature that needs this package.
            suggestion: Installation suggestion text.
            platform_issue: Whether this is a platform compatibility issue.
        """
        # Check environment variables for warning suppression
        if os.getenv("OPR_NO_OPTIONAL_WARNINGS") == "1":
            return

        key = f"{package}:{feature}"
        with _warn_lock:
            if key in _warnings_shown:
                return
            _warnings_shown.add(key)

        if platform_issue:
            msg = f"{package} found but is not importable on this platform. {feature} will be disabled."
        else:
            msg = f"{package} is not available. {feature} will be disabled."

        if suggestion:
            msg += f" {suggestion}"

        logger.warning(msg)

    @classmethod
    def require_or_warn(
        cls,
        package: str,
        feature: str,
        suggestion: str = "",
        min_version: str | None = None,
    ) -> bool:
        """Check package availability and warn if unavailable or wrong version.

        Args:
            package: Name of the required package.
            feature: Description of the feature that needs this package.
            suggestion: Installation suggestion text.
            min_version: Minimum required version string, if any.

        Returns:
            True if package is available and meets requirements.
        """
        # Fast path: check if package exists on path
        if not cls.exists_on_path(package):
            cls.warn_once(package, feature, suggestion)
            return False

        # Check version requirements
        if not cls.is_available(package, min_version):
            version_msg = f" (requires >= {min_version})" if min_version else ""
            cls.warn_once(package, feature, f"{suggestion}{version_msg}")
            return False

        # Check if actually importable (platform compatibility)
        if not cls.is_importable(package):
            cls.warn_once(package, feature, suggestion, platform_issue=True)
            return False

        return True


# Simplified lazy import functionality based on expert review


@lru_cache(maxsize=128)
def lazy(package: str, *, feature: str, min_version: str | None = None) -> object:
    """Lazy import with existence check, version validation, and fallback stub.

    This function implements the expert-recommended pattern: a single entry point
    for optional dependency handling that performs existence check → version check
    → import → fallback stub in one streamlined flow.

    Args:
        package: Package name (e.g., "MinkowskiEngine")
        feature: Feature description for error messages (keyword-only for clarity)
        min_version: Minimum required version (optional)

    Returns:
        Real module if available and importable, or stub that raises helpful errors

    Example:
        >>> ME = lazy("MinkowskiEngine", feature="sparse convs", min_version="0.7.0")
        >>> tensor = ME.SparseTensor(features, coords)  # Works or fails helpfully

    Educational Note:
        This function demonstrates several important patterns:
        - @lru_cache for performance (avoids repeated checks)
        - Progressive validation (existence → version → import)
        - Graceful degradation with helpful error messages
        - Integration with existing warning system
    """
    # Fast existence check (no side effects like CUDA initialization)
    if not OptionalDependencyManager.exists_on_path(package):
        OptionalDependencyManager.warn_once(package, feature, _INSTALLATION_DOCS_MESSAGE)
        return _SimplifiedMissingDepStub(package, feature, _INSTALLATION_DOCS_MESSAGE)

    # Version check if specified (only after confirming package exists)
    if min_version and not OptionalDependencyManager.is_available(package, min_version):
        msg = f"{_INSTALLATION_DOCS_MESSAGE} (requires >= {min_version})"
        OptionalDependencyManager.warn_once(package, feature, msg)
        return _SimplifiedMissingDepStub(package, feature, msg)

    # Actually import the module (this is where CUDA initialization etc. happens)
    try:
        return importlib.import_module(package)
    except ImportError as e:
        # Platform-specific import failure (package exists but can't be loaded)
        msg = f"{_INSTALLATION_DOCS_MESSAGE} (import failed: {e})"
        OptionalDependencyManager.warn_once(package, feature, msg, platform_issue=True)
        return _SimplifiedMissingDepStub(package, feature, msg)


class _SimplifiedMissingDepStub:
    """Minimal stub that raises helpful errors - expert's simplified design.

    This simplified version only implements __getattr__ (the most common case)
    and stores a single error message for simplicity and performance.

    Educational Note:
        This demonstrates the "fail fast with helpful messages" pattern.
        When optional dependencies are missing, we want clear, actionable errors
        rather than confusing attribute errors.
    """

    def __init__(self, package: str, feature: str, install_cmd: str) -> None:
        """Initialize stub with helpful error message.

        Args:
            package: Name of the missing package
            feature: Description of the feature requiring this package
            install_cmd: Installation instructions
        """
        self._msg = f"{package} required for {feature}.\n{install_cmd}"

    def __getattr__(self, _: str) -> None:
        """Raise helpful error for any attribute access.

        Args:
            _: Attribute name (ignored - all attributes raise same error)

        Raises:
            RuntimeError: With helpful installation message

        Educational Note:
            Using __getattr__ means this only triggers for missing attributes,
            allowing built-in attributes like __class__ to work normally.
        """
        raise RuntimeError(self._msg)


# Convenience functions for key packages


def has_minkowski(min_version: str | None = None) -> bool:
    """Check if MinkowskiEngine is available.

    Args:
        min_version: Minimum required version string, if any.

    Returns:
        True if MinkowskiEngine is available and meets version requirements.
    """
    return OptionalDependencyManager.is_available("MinkowskiEngine", min_version)


def require_minkowski(feature: str, min_version: str | None = None) -> bool:
    """Require MinkowskiEngine with informative error messages.

    Args:
        feature: Description of the feature that needs MinkowskiEngine.
        min_version: Minimum required version string, if any.

    Returns:
        True if MinkowskiEngine is available and meets requirements.
    """
    suggestion = _INSTALLATION_DOCS_MESSAGE
    return OptionalDependencyManager.require_or_warn("MinkowskiEngine", feature, suggestion, min_version)


def get_minkowski_stub(feature: str) -> _MissingDepStub:
    """Get a stub that raises informative errors when MinkowskiEngine is missing.

    Args:
        feature: Description of the feature that needs MinkowskiEngine.

    Returns:
        A stub object that raises helpful errors when accessed.
    """
    return _MissingDepStub(
        "MinkowskiEngine",
        feature,
        _INSTALLATION_DOCS_MESSAGE,
    )


def has_paddleocr(min_version: str | None = None) -> bool:
    """Check if PaddleOCR and PaddlePaddle are available.

    Args:
        min_version: Minimum required version string for PaddleOCR, if any.

    Returns:
        True if both PaddleOCR and PaddlePaddle are available.
    """
    return OptionalDependencyManager.is_available(
        "paddleocr", min_version
    ) and OptionalDependencyManager.is_available("paddlepaddle")


def require_paddleocr(feature: str, min_version: str | None = None) -> bool:
    """Require PaddleOCR with informative error messages.

    Args:
        feature: Description of the feature that needs PaddleOCR.
        min_version: Minimum required version string for PaddleOCR, if any.

    Returns:
        True if PaddleOCR is available and meets requirements.
    """
    if has_paddleocr(min_version):
        return True

    suggestion = _INSTALLATION_DOCS_MESSAGE
    OptionalDependencyManager.warn_once("PaddleOCR", feature, suggestion)
    return False


def get_paddleocr_stub(feature: str) -> _MissingDepStub:
    """Get a stub that raises informative errors when PaddleOCR is missing.

    Args:
        feature: Description of the feature that needs PaddleOCR.

    Returns:
        A stub object that raises helpful errors when accessed.
    """
    return _MissingDepStub("PaddleOCR", feature, _INSTALLATION_DOCS_MESSAGE)


# Public API for external use
__all__ = [
    "OptionalDependencyManager",
    "has_minkowski",
    "require_minkowski",
    "get_minkowski_stub",
    "has_paddleocr",
    "require_paddleocr",
    "get_paddleocr_stub",
    "lazy",
]
