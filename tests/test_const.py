import os
import sys
from pathlib import Path

import pytest

from opr.const import PACKAGE_NAME, get_cache_folder


def test_get_cache_folder(monkeypatch):
    # Test for Linux
    monkeypatch.setattr(sys, "platform", "linux")
    os.environ["XDG_CACHE_HOME"] = "/tmp/my_cache"
    assert get_cache_folder() == Path("/tmp/my_cache") / PACKAGE_NAME

    # Test for macOS
    monkeypatch.setattr(sys, "platform", "darwin")
    assert get_cache_folder() == Path.home() / "Library" / "Caches" / PACKAGE_NAME

    # Test for Windows
    monkeypatch.setattr(sys, "platform", "win32")
    assert get_cache_folder() == Path.home() / ".cache" / PACKAGE_NAME

    # Test for unexpected platform
    monkeypatch.setattr(sys, "platform", "unknown")
    with pytest.raises(ValueError) as e:
        get_cache_folder()
    assert str(e.value) == "Unexpected platform 'unknown'."
