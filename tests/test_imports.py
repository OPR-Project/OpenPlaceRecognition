"""Test that all library modules can be imported without errors."""

import importlib
import os
import pkgutil
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any

import pytest

# Default timeout for module imports (seconds)
DEFAULT_TIMEOUT = 10


def find_modules(package_name: str) -> list[str]:
    """Find all modules within a package."""
    modules = []
    to_process = deque([package_name])

    while to_process:
        current = to_process.popleft()
        try:
            pkg = importlib.import_module(current)
            pkg_path = os.path.dirname(pkg.__file__)

            for _, name, is_pkg in pkgutil.iter_modules([pkg_path]):
                full_name = f"{current}.{name}"
                modules.append(full_name)
                if is_pkg:
                    to_process.append(full_name)
        except (ImportError, AttributeError):
            continue

    return modules


def try_import(module_name: str, timeout: int = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """Try to import a module with timeout protection."""
    result = {"success": False, "error": None}

    def _import() -> None:
        try:
            importlib.import_module(module_name)
            result["success"] = True
        except Exception as e:
            result["error"] = str(e)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_import)
        try:
            future.result(timeout=timeout)
        except TimeoutError:
            result["error"] = f"Import timed out after {timeout} seconds"

    return result


@pytest.mark.unit
def test_imports() -> None:
    """Test that all library modules can be imported without errors."""
    package_name = "opr"

    # Ensure main package can be imported
    try:
        importlib.import_module(package_name)
    except ImportError as e:
        pytest.fail(f"Failed to import main package {package_name!r}: {e}")

    # Find and test all modules
    modules = find_modules(package_name)
    assert modules, f"No modules found in package {package_name!r}"

    failures = [(name, result["error"]) for name in modules if not (result := try_import(name))["success"]]

    if failures:
        error_msg = "\n".join(f"- {mod}: {err}" for mod, err in failures)
        pytest.fail(f"Failed imports:\n{error_msg}")
