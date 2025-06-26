"""Package-wide test fixtures."""

from _pytest.config import Config


def pytest_configure(config: Config) -> None:
    """Pytest configuration hook."""
    config.addinivalue_line("markers", "e2e: mark as end-to-end test.")
    config.addinivalue_line("markers", "unit: mark as unit test.")
    config.addinivalue_line("markers", "integration: mark as integration test.")
    config.addinivalue_line("markers", "model: tests for model implementations.")
    config.addinivalue_line("markers", "dataset: tests for dataset loading and preprocessing.")
    config.addinivalue_line("markers", "metrics: tests for evaluation metrics.")
    config.addinivalue_line("markers", "gpu: tests requiring GPU resources.")
    config.addinivalue_line("markers", "slow: mark tests that take a long time to run.")
