"""Definition of package-level constants."""
import os
import sys
import tempfile
from pathlib import Path

from . import __name__


def get_cache_folder() -> Path:
    """Returns the path to the cache folder.

    Raises:
        ValueError: If the system's platform is unexpected.

    Returns:
        Path: The path to the cache folder.
    """
    if sys.platform == "linux" or sys.platform == "linux2":
        return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / PACKAGE_NAME

    elif sys.platform == "darwin":  # mac os
        return Path.home() / "Library" / "Caches" / PACKAGE_NAME

    elif sys.platform.startswith("win"):
        return Path.home() / ".cache" / PACKAGE_NAME

    else:
        raise ValueError(f"Unexpected platform '{sys.platform}'.")


PACKAGE_NAME = __name__

PROJECT_ROOT = Path(__file__).parent.parent
CACHE_PATH = get_cache_folder()
TMP_PATH = Path(tempfile.gettempdir())
