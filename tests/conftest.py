"""Package-wide test fixtures."""

from typing import Any, Callable, Iterator

import pandas as pd
import pytest
import torch
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


@pytest.fixture
def mock_model() -> torch.nn.Module:
    """Simple mock model returning zero embeddings of fixed dimension."""

    class SimpleMockModel(torch.nn.Module):
        """Mock PyTorch model that returns zero tensors."""

        def __init__(self, embedding_dim: int = 64) -> None:
            """Initialize with embedding dimension.

            Args:
                embedding_dim: Dimension of output embeddings
            """
            super().__init__()
            self.embedding_dim = embedding_dim

        def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            """Forward pass returning zero embeddings.

            Args:
                inputs: Dictionary of input tensors

            Returns:
                Dictionary with final_descriptor tensor
            """
            batch_size = next(iter(inputs.values())).size(0)
            return {"final_descriptor": torch.zeros(batch_size, self.embedding_dim)}

    return SimpleMockModel()


@pytest.fixture
def mock_dataloader() -> Callable[[pd.DataFrame, int, tuple[int, ...]], Any]:
    """Factory: exposes dataset_df and yields batches with key "images"."""

    def _factory(
        dataset_df: pd.DataFrame, batch_size: int = 2, image_shape: tuple[int, ...] = (1, 2, 2)
    ) -> Any:
        """Create a mock dataloader with specified parameters.

        Args:
            dataset_df: DataFrame with dataset metadata
            batch_size: Number of samples per batch
            image_shape: Shape of mock images (C, H, W)

        Returns:
            Mock dataloader instance
        """

        class DummyDataset:
            """Simple dataset wrapper around a DataFrame."""

            def __init__(self, df: pd.DataFrame) -> None:
                """Initialize with a DataFrame.

                Args:
                    df: DataFrame with dataset metadata
                """
                self.dataset_df = df

        class DummyDataLoader:
            """Mock dataloader that yields batches of zero tensors."""

            def __init__(self, df: pd.DataFrame, bs: int, img_shape: tuple[int, ...]) -> None:
                """Initialize dataloader with dataset and batch parameters.

                Args:
                    df: DataFrame with dataset metadata
                    bs: Batch size
                    img_shape: Shape of mock images
                """
                self.dataset = DummyDataset(df)
                self._n = len(df)
                self.batch_size = bs
                self.image_shape = img_shape

            def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
                """Iterate through batches of zero tensors.

                Yields:
                    Dictionary with 'images' key containing zero tensors
                """
                for start in range(0, self._n, self.batch_size):
                    bs = min(self.batch_size, self._n - start)
                    # yield a zero‐tensor batch of shape (batch_size, C, H, W)
                    yield {"images": torch.zeros(bs, *self.image_shape)}

        return DummyDataLoader(dataset_df, batch_size, image_shape)

    return _factory
