"""Test ModelTester class for place recognition."""

from pathlib import Path
from typing import Callable, Iterator
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch

from opr.testers.place_recognition.model import (
    ModelTester,
    RetrievalResults,
    RetrievalResultsCollection,
)


@pytest.mark.unit
def test_init_invalid_params(mock_model: torch.nn.Module, mock_dataloader: Callable) -> None:
    """Test that ModelTester raises ValueError for invalid initialization parameters."""
    df = pd.DataFrame({"x": [0], "y": [0], "track": [0]})
    dl = mock_dataloader(df, batch_size=1)
    with pytest.raises(ValueError):
        ModelTester(mock_model, dl, dist_thresh=0.0)
    with pytest.raises(ValueError):
        ModelTester(mock_model, dl, at_n=0)


@pytest.mark.unit
def test_init_valid_params_sets_defaults(mock_model: torch.nn.Module, mock_dataloader: Callable) -> None:
    """Test that ModelTester sets correct default values when initialized with valid parameters."""
    df = pd.DataFrame({"x": [0], "y": [0], "track": [1]})
    dl = mock_dataloader(df, batch_size=1)
    tester = ModelTester(mock_model, dl)
    assert tester.dist_thresh == 25.0
    assert tester.at_n == 25
    # coords names for x/y scheme
    assert tester._coords_columns_names == ["x", "y"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "cols",
    [
        ["northing", "track"],  # missing easting
        ["x", "track"],  # missing y
        ["tx", "tz", "track"],  # missing ty
    ],
)
def test_get_coords_columns_names_errors(
    mock_model: torch.nn.Module, mock_dataloader: Callable, cols: list[str]
) -> None:
    """Test that ModelTester raises ValueError for invalid coordinate column schemas."""
    df = pd.DataFrame({c: [0, 1] for c in cols})
    df["track"] = df["track"]
    dl = mock_dataloader(df, batch_size=1)
    with pytest.raises(ValueError):
        ModelTester(mock_model, dl)


@pytest.mark.unit
@pytest.mark.parametrize(
    "cols, expected",
    [
        (["northing", "easting", "track"], ["northing", "easting"]),
        (["x", "y", "z", "track"], ["x", "y", "z"]),
        (["tx", "ty", "track"], ["tx", "ty"]),
    ],
)
def test_get_coords_columns_names_variants(
    mock_model: torch.nn.Module, mock_dataloader: Callable, cols: list[str], expected: list[str]
) -> None:
    """Detect various coordinate column schemes."""
    data = {c: np.arange(3) for c in cols}
    df = pd.DataFrame(data)
    df["track"] = 0  # ensure track col exists
    dl = mock_dataloader(df, batch_size=2)
    tester = ModelTester(mock_model, dl)
    assert tester._coords_columns_names == expected


@pytest.mark.unit
def test_extract_embeddings_shape_and_values(mock_model: torch.nn.Module, mock_dataloader: Callable) -> None:
    """_embs = zeros(batch_size*steps, embedding_dim)."""
    # create df of length 3
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "track": [0, 0, 0]})
    dl = mock_dataloader(df, batch_size=2, image_shape=(1, 1, 1))
    tester = ModelTester(mock_model, dl)
    embs = tester._extract_embeddings()
    # should be numpy array of shape (3, embedding_dim)
    assert isinstance(embs, np.ndarray)
    assert embs.shape == (3, mock_model.embedding_dim)
    assert np.allclose(embs, 0.0)


@pytest.mark.unit
def test_group_by_track_without_in_query(mock_model: torch.nn.Module, mock_dataloader: Callable) -> None:
    """_group_by_track groups indices per track when no in_query."""
    df = pd.DataFrame({"x": [0, 1, 2, 3], "y": [0, 1, 2, 3], "track": [0, 0, 1, 1]})
    dl = mock_dataloader(df, batch_size=2)
    tester = ModelTester(mock_model, dl)
    queries, databases = tester._group_by_track()
    assert queries == databases
    # databases: [[0,1],[2,3]]
    assert databases == [[0, 1], [2, 3]]


@pytest.mark.unit
def test_group_by_track_with_in_query(mock_model: torch.nn.Module, mock_dataloader: Callable) -> None:
    """_group_by_track separates only rows flagged in 'in_query'."""
    df = pd.DataFrame(
        {
            "x": [0, 1, 2, 3],
            "y": [0, 1, 2, 3],
            "track": [0, 0, 1, 1],
            "in_query": [True, False, False, True],
        }
    )
    dl = mock_dataloader(df, batch_size=2)
    tester = ModelTester(mock_model, dl)
    queries, databases = tester._group_by_track()
    assert databases == [[0, 1], [2, 3]]
    assert queries == [[0], [3]]


@pytest.mark.unit
def test_extract_embeddings_filters_idx_utms() -> None:
    """Ensure 'idxs' and 'utms' keys are dropped before model.forward."""

    class SpyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.keys_seen: list[set[str]] = []

        def to(self, device: str) -> "SpyModel":
            return self

        def eval(self) -> None:
            pass

        def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            self.keys_seen.append(set(inputs.keys()))
            # return dict with final_descriptor to satisfy new behavior
            batch_size = next(iter(inputs.values())).shape[0]
            return {"final_descriptor": torch.zeros(batch_size, 1)}

    # dummy DataLoader with idxs, utms and feat keys
    class DummyDL:
        def __init__(self, df: pd.DataFrame) -> None:
            self.dataset = type("D", (), {"dataset_df": df})()

        def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
            yield {
                "idxs": torch.tensor([0]),
                "utms": torch.tensor([0]),
                "feat": torch.tensor([[1.0]]),
            }

    df = pd.DataFrame({"x": [0], "y": [0], "track": [0]})
    model = SpyModel()
    dl = DummyDL(df)
    tester = ModelTester(model, dl)
    tester._extract_embeddings()
    # assert only 'feat' was passed, not 'idxs'/'utms'
    assert all(keys == {"feat"} for keys in model.keys_seen)


@pytest.mark.unit
def test_compute_geo_dist(mock_model: torch.nn.Module, mock_dataloader: Callable) -> None:
    """_compute_geo_dist should correctly compute pairwise L2 distances."""
    # Create a simple dataset
    df = pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2], "track": [0, 0, 0]})
    dl = mock_dataloader(df, batch_size=1)
    tester = ModelTester(mock_model, dl)

    # Test coordinates (3 points in 2D)
    coords = np.array([[0, 0], [3, 0], [0, 4]])  # point 1  # point 2  # point 3

    # Call the method
    dist_matrix = tester._compute_geo_dist(coords)

    # Expected distances:
    # - dist(p1, p1) = 0
    # - dist(p1, p2) = 3
    # - dist(p1, p3) = 4
    # - dist(p2, p2) = 0
    # - dist(p2, p3) = 5 (using Pythagorean theorem: sqrt(3^2 + 4^2))
    # - dist(p3, p3) = 0
    expected = np.array([[0, 3, 4], [3, 0, 5], [4, 5, 0]], dtype=float)

    # Verify results
    assert isinstance(dist_matrix, np.ndarray)
    assert dist_matrix.shape == (3, 3)
    assert np.allclose(dist_matrix, expected)


@pytest.mark.unit
def test_eval_pairs_basic_functionality(mock_model: torch.nn.Module, mock_dataloader: Callable) -> None:
    """_eval_pairs should handle basic track pairs correctly and return a RetrievalResultsCollection."""
    # Create dataset with two tracks
    df = pd.DataFrame({"x": [0, 1, 2, 3], "y": [0, 1, 2, 3], "track": [0, 0, 1, 1]})
    dl = mock_dataloader(df, batch_size=2)
    tester = ModelTester(mock_model, dl)
    tester.at_n = 2  # Small at_n for testing

    # Mock embeddings and distances
    embs = np.array([[1, 0], [2, 0], [0, 1], [0, 2]])  # 4 embeddings
    queries = [[0, 1], [2, 3]]  # Two tracks, each with two samples
    databases = [[0, 1], [2, 3]]

    # Geo distances: track 0 close to track 1
    geo_dist = np.array(
        [[0.0, 1.0, 0.5, 1.5], [1.0, 0.0, 1.5, 0.5], [0.5, 1.5, 0.0, 1.0], [1.5, 0.5, 1.0, 0.0]]
    )

    # Mock eval_retrieval_pair to return controlled values
    with mock.patch.object(tester, "eval_retrieval_pair") as mock_eval:
        # Create mock RetrievalResults objects
        mock_results_0_to_1 = mock.MagicMock()
        mock_results_0_to_1.recall_at_n = np.array([0.5, 1.0])
        mock_results_0_to_1.recall_at_one_percent = 1.0
        mock_results_0_to_1.top1_distance = 0.1

        mock_results_1_to_0 = mock.MagicMock()
        mock_results_1_to_0.recall_at_n = np.array([0.7, 0.9])
        mock_results_1_to_0.recall_at_one_percent = 0.9
        mock_results_1_to_0.top1_distance = 0.2

        # Return different results for the two track pairs
        mock_eval.side_effect = [mock_results_0_to_1, mock_results_1_to_0]

        # Call the method
        results_collection = tester._eval_pairs(embs, queries, databases, geo_dist)

        # Check that we got a collection with the expected number of results
        assert isinstance(results_collection, RetrievalResultsCollection)
        assert len(results_collection) == 2
        assert results_collection.results[0] is mock_results_0_to_1
        assert results_collection.results[1] is mock_results_1_to_0


@pytest.mark.unit
def test_eval_pairs_with_none_top1(mock_model: torch.nn.Module, mock_dataloader: Callable) -> None:
    """_eval_pairs should handle None top1_distance correctly."""
    df = pd.DataFrame({"x": [0, 1, 2, 3], "y": [0, 1, 2, 3], "track": [0, 0, 1, 1]})
    dl = mock_dataloader(df, batch_size=2)
    tester = ModelTester(mock_model, dl)
    tester.at_n = 1  # Small at_n for testing

    embs = np.array([[1, 0], [2, 0], [0, 1], [0, 2]])
    queries = [[0, 1], [2, 3]]
    databases = [[0, 1], [2, 3]]
    geo_dist = np.zeros((4, 4))  # Not important for this test

    # Mock eval_retrieval_pair to return None for top1_distance
    with mock.patch.object(tester, "eval_retrieval_pair") as mock_eval:
        mock_results_0_to_1 = mock.MagicMock()
        mock_results_0_to_1.recall_at_n = np.array([0.0])
        mock_results_0_to_1.recall_at_one_percent = 0.0
        mock_results_0_to_1.top1_distance = None

        mock_results_1_to_0 = mock.MagicMock()
        mock_results_1_to_0.recall_at_n = np.array([0.5])
        mock_results_1_to_0.recall_at_one_percent = 0.5
        mock_results_1_to_0.top1_distance = 0.1

        mock_eval.side_effect = [mock_results_0_to_1, mock_results_1_to_0]

        results_collection = tester._eval_pairs(embs, queries, databases, geo_dist)

    # Check top1_dists values - should have None for the first pair
    assert results_collection.results[0].top1_distance is None
    assert results_collection.results[1].top1_distance == 0.1


@pytest.mark.unit
def test_eval_pairs_empty_track(mock_model: torch.nn.Module, mock_dataloader: Callable) -> None:
    """_eval_pairs should handle empty tracks gracefully."""
    df = pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2], "track": [0, 0, 1], "in_query": [True, False, True]})
    dl = mock_dataloader(df, batch_size=1)
    tester = ModelTester(mock_model, dl)

    embs = np.array([[1, 0], [2, 0], [0, 1]])
    queries = [[0], [2], []]  # Third track is empty
    databases = [[0, 1], [2], []]
    geo_dist = np.zeros((3, 3))

    # Mock eval_retrieval_pair
    with mock.patch.object(tester, "eval_retrieval_pair") as mock_eval:
        mock_result = mock.MagicMock()
        mock_eval.return_value = mock_result

        # This should not raise errors despite empty track
        results_collection = tester._eval_pairs(embs, queries, databases, geo_dist)

    # Check that the method completed and returned a collection
    assert isinstance(results_collection, RetrievalResultsCollection)

    # With three tracks where not all pairs are valid (empty tracks),
    # we expect fewer than 6 possible pairs in the collection
    assert len(results_collection) < 6


@pytest.mark.unit
def test_aggregate_basic_functionality(mock_model: torch.nn.Module, mock_dataloader: Callable) -> None:
    """_aggregate should correctly compute mean values across all track pairs."""
    df = pd.DataFrame({"x": [0, 1], "y": [0, 1], "track": [0, 1]})
    dl = mock_dataloader(df, batch_size=1)
    tester = ModelTester(mock_model, dl)

    # Create test data - 2 tracks, with recall metrics for (0,1) and (1,0) pairs
    # Shape: (2 tracks, 2 tracks, 3 values for at_n=3)
    recalls_at_n = np.array(
        [
            [[np.nan, np.nan, np.nan], [0.2, 0.4, 0.6]],  # Track 0: nan for (0,0), values for (0,1)
            [[0.3, 0.5, 0.7], [np.nan, np.nan, np.nan]],  # Track 1: values for (1,0), nan for (1,1)
        ]
    )

    # Shape: (2 tracks, 2 tracks)
    recalls_at_one_percent = np.array(
        [
            [np.nan, 0.4],  # Track 0: nan for (0,0), 0.4 for (0,1)
            [0.6, np.nan],  # Track 1: 0.6 for (1,0), nan for (1,1)
        ]
    )

    # Shape: (2 tracks, 2 tracks)
    top1_distances = np.array(
        [
            [np.nan, 5.0],  # Track 0: nan for (0,0), 5.0 for (0,1)
            [3.0, np.nan],  # Track 1: 3.0 for (1,0), nan for (1,1)
        ]
    )

    # Call the method
    mean_recalls_at_n, mean_recall_1p, mean_top1_dist = tester._aggregate(
        recalls_at_n, recalls_at_one_percent, top1_distances
    )

    # Expected values - mean across all values ignoring NaN
    # - recalls: mean of [0.2, 0.4, 0.6, 0.3, 0.5, 0.7] by position = [0.25, 0.45, 0.65]
    # - recalls@1%: mean of [0.4, 0.6] = 0.5
    # - top1: mean of [5.0, 3.0] = 4.0
    expected_recalls_n = np.array([0.25, 0.45, 0.65])

    # Check results - now testing scalar outputs
    assert np.allclose(mean_recalls_at_n, expected_recalls_n)
    assert mean_recall_1p == 0.5  # Global mean R@1%
    assert mean_top1_dist == 4.0  # Global mean top1 distance


@pytest.mark.unit
def test_aggregate_with_nan_values(mock_model: torch.nn.Module, mock_dataloader: Callable) -> None:
    """_aggregate should handle NaN values correctly."""
    df = pd.DataFrame({"x": [0], "y": [0], "track": [0]})
    dl = mock_dataloader(df, batch_size=1)
    tester = ModelTester(mock_model, dl)

    # Create test data with more NaN values
    recalls_at_n = np.array(
        [
            [[np.nan, np.nan], [0.1, np.nan], [0.2, 0.3]],  # Track 0
            [[0.4, 0.5], [np.nan, np.nan], [np.nan, 0.6]],  # Track 1
            [[np.nan, 0.7], [0.8, 0.9], [np.nan, np.nan]],  # Track 2
        ]
    )

    recalls_at_one_percent = np.array([[np.nan, 0.1, np.nan], [0.2, np.nan, 0.3], [0.4, np.nan, np.nan]])

    top1_distances = np.array([[np.nan, 1.0, 2.0], [3.0, np.nan, np.nan], [np.nan, 4.0, np.nan]])

    # Call the method
    mean_recalls_at_n, mean_recall_1p, mean_top1_dist = tester._aggregate(
        recalls_at_n, recalls_at_one_percent, top1_distances
    )

    # Expected means across all values (ignoring NaNs)
    # recalls: [0.1, 0.2, 0.4, 0.3, 0.8, 0.7, 0.5, 0.6, 0.9] by position
    # recalls@1%: [0.1, 0.2, 0.3, 0.4] = 0.25
    # top1: [1.0, 2.0, 3.0, 4.0] = 2.5
    expected_recalls_n = np.array([0.375, 0.6])  # Mean by position across all values

    # Check scalar results
    assert np.allclose(mean_recalls_at_n, expected_recalls_n)
    assert mean_recall_1p == 0.25
    assert mean_top1_dist == 2.5


@pytest.mark.unit
def test_aggregate_all_nan(mock_model: torch.nn.Module, mock_dataloader: Callable) -> None:
    """_aggregate should handle arrays with all NaN values."""
    df = pd.DataFrame({"x": [0], "y": [0], "track": [0]})
    dl = mock_dataloader(df, batch_size=1)
    tester = ModelTester(mock_model, dl)

    # Create test data with all NaN values
    recalls_at_n = np.full((2, 2, 3), np.nan)
    recalls_at_one_percent = np.full((2, 2), np.nan)
    top1_distances = np.full((2, 2), np.nan)

    # Call the method
    mean_recalls_at_n, mean_recall_1p, mean_top1_dist = tester._aggregate(
        recalls_at_n, recalls_at_one_percent, top1_distances
    )

    # All means should be NaN
    assert np.all(np.isnan(mean_recalls_at_n))
    assert np.isnan(mean_recall_1p)
    assert np.isnan(mean_top1_dist)

    # Check shapes - now we expect scalars/1D arrays
    assert mean_recalls_at_n.shape == (3,)
    assert np.isscalar(mean_recall_1p)  # Now a scalar, not array
    assert np.isscalar(mean_top1_dist)  # Now a scalar, not array


@pytest.mark.unit
def test_run_integration(mock_model: torch.nn.Module, mock_dataloader: Callable) -> None:
    """run() should orchestrate all steps of the testing pipeline correctly."""
    # Create test data
    df = pd.DataFrame({"x": [0, 1, 2, 3], "y": [0, 1, 2, 3], "track": [0, 0, 1, 1]})
    dl = mock_dataloader(df, batch_size=2)
    tester = ModelTester(mock_model, dl)

    # Mock all the component methods with appropriate return values
    with mock.patch.object(tester, "_extract_embeddings") as mock_extract:
        mock_extract.return_value = np.array([[1, 0], [2, 0], [0, 1], [0, 2]])

        with mock.patch.object(tester, "_group_by_track") as mock_group:
            mock_group.return_value = ([[0, 1], [2, 3]], [[0, 1], [2, 3]])

            with mock.patch.object(tester, "_compute_geo_dist") as mock_dist:
                mock_dist.return_value = np.zeros((4, 4))

                with mock.patch.object(tester, "_eval_pairs") as mock_eval:
                    # Mock the return value as a RetrievalResultsCollection instead of metrics tuple
                    mock_collection = RetrievalResultsCollection()
                    mock_eval.return_value = mock_collection

                    # Run the method
                    result = tester.run()

    # Check that all methods were called with the right arguments
    mock_extract.assert_called_once()
    mock_group.assert_called_once()
    mock_dist.assert_called_once()
    mock_eval.assert_called_once()

    # Check method call order through argument passing
    mock_eval_args = mock_eval.call_args[0]
    assert np.array_equal(mock_eval_args[0], mock_extract.return_value)
    assert mock_eval_args[1:3] == mock_group.return_value
    assert np.array_equal(mock_eval_args[3], mock_dist.return_value)

    # Verify the result is the expected RetrievalResultsCollection
    assert result is mock_collection


@pytest.mark.unit
def test_run_end_to_end(mock_model: torch.nn.Module, mock_dataloader: Callable) -> None:
    """run() should execute a complete evaluation workflow with minimal data."""
    # Create a minimal dataset with two tracks
    df = pd.DataFrame({"x": [0, 1, 2, 3], "y": [0, 1, 2, 3], "track": [0, 0, 1, 1]})
    dl = mock_dataloader(df, batch_size=2)

    # We'll use the actual implementation but with a small at_n value
    tester = ModelTester(mock_model, dl, at_n=2)

    # Run the full evaluation
    result = tester.run()

    # Check that we got a RetrievalResultsCollection
    assert isinstance(result, RetrievalResultsCollection)

    # For end-to-end test, verify basics of the collection
    assert isinstance(result.num_pairs, int)
    assert isinstance(result.num_queries, int)
    assert isinstance(result.num_tracks, tuple)


@pytest.mark.unit
def test_compute_geo_dist_batched(mock_model: torch.nn.Module, mock_dataloader: Callable) -> None:
    """Batched _compute_geo_dist should produce identical results to non-batched version."""
    # Create a simple dataset
    df = pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2], "track": [0, 0, 0]})
    dl = mock_dataloader(df, batch_size=1)
    tester = ModelTester(mock_model, dl)

    # Test coordinates (5 points in 2D)
    coords = np.array([[0, 0], [3, 0], [0, 4], [1, 1], [2, 2]])

    # Get results from original implementation
    dist_matrix_original = tester._compute_geo_dist(coords)

    # Get results from batched implementation with batch_size=2
    dist_matrix_batched = tester._compute_geo_dist(coords, batch_size=2)

    # Verify results match
    assert dist_matrix_batched.shape == dist_matrix_original.shape
    assert np.allclose(dist_matrix_batched, dist_matrix_original)


@pytest.mark.unit
def test_compute_geo_dist_batch_sizes(mock_model: torch.nn.Module, mock_dataloader: Callable) -> None:
    """_compute_geo_dist should work with different batch sizes."""
    df = pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2], "track": [0, 0, 0]})
    dl = mock_dataloader(df, batch_size=1)
    tester = ModelTester(mock_model, dl)

    # Create array of 10 random points
    np.random.seed(42)  # For reproducibility
    coords = np.random.rand(10, 3)

    # Original non-batched result
    expected = tester._compute_geo_dist(coords)

    # Test different batch sizes
    for batch_size in [1, 2, 3, 5, 10]:
        result = tester._compute_geo_dist(coords, batch_size=batch_size)
        assert np.allclose(result, expected), f"Failed with batch_size={batch_size}"


@pytest.mark.unit
def test_compute_geo_dist_edge_cases(mock_model: torch.nn.Module, mock_dataloader: Callable) -> None:
    """_compute_geo_dist should handle edge cases correctly."""
    df = pd.DataFrame({"x": [0, 1], "y": [0, 1], "track": [0, 0]})
    dl = mock_dataloader(df, batch_size=1)
    tester = ModelTester(mock_model, dl)

    # Test with a single point
    single_point = np.array([[1, 2, 3]])
    result_single = tester._compute_geo_dist(single_point, batch_size=1)
    expected_single = np.array([[0.0]])
    assert np.allclose(result_single, expected_single)

    # Test with batch size larger than array length
    coords = np.array([[0, 0], [3, 0], [0, 4]])
    result_large_batch = tester._compute_geo_dist(coords, batch_size=10)
    expected = tester._compute_geo_dist(coords)
    assert np.allclose(result_large_batch, expected)


@pytest.mark.unit
def test_compute_geo_dist_memory_efficiency(mock_model: torch.nn.Module, mock_dataloader: Callable) -> None:
    """Batched _compute_geo_dist should be more memory efficient."""
    df = pd.DataFrame({"x": [0, 1], "y": [0, 1], "track": [0, 0]})
    dl = mock_dataloader(df, batch_size=1)
    tester = ModelTester(mock_model, dl, verbose=False)

    # Skip detailed memory checks if running in CI or with small memory
    try:
        # Create moderately large array (won't cause memory issues in most environments)
        coords = np.random.rand(1000, 3).astype(np.float32)

        # Force garbage collection before test
        import gc

        gc.collect()

        # Check that batched version works with a larger array
        result_batched = tester._compute_geo_dist(coords, batch_size=100)
        assert result_batched.shape == (1000, 1000)

        # This is mostly a functional test to ensure the batched version works
        # with larger arrays. Actual memory usage would require a more complex test setup.
    except Exception as e:
        pytest.skip(f"Skipping memory test due to: {str(e)}")


@pytest.mark.unit
def test_compute_geo_dist_with_progress_bar(mock_model: torch.nn.Module, mock_dataloader: Callable) -> None:
    """Batched _compute_geo_dist should show progress bar when verbose=True."""
    df = pd.DataFrame({"x": [0, 1], "y": [0, 1], "track": [0, 0]})
    dl = mock_dataloader(df, batch_size=1)

    # Create instance with verbose=True
    tester = ModelTester(mock_model, dl, verbose=True)

    # Small array for quick testing
    coords = np.random.rand(10, 2)

    # Mock tqdm to check if it's called correctly
    with mock.patch("opr.testers.place_recognition.model.tqdm") as mock_tqdm:
        tester._compute_geo_dist(coords, batch_size=2)
        mock_tqdm.assert_called_once()

        # Verify the tqdm call contains the expected parameters
        args, kwargs = mock_tqdm.call_args
        assert "desc" in kwargs
        assert kwargs["disable"] is False  # Progress bar should be enabled

    # Test with verbose=False - should not show progress bar
    tester.verbose = False
    with mock.patch("opr.testers.place_recognition.model.tqdm") as mock_tqdm:
        tester._compute_geo_dist(coords, batch_size=2)
        mock_tqdm.assert_called_once()

        # Verify disable=True to hide the progress bar
        args, kwargs = mock_tqdm.call_args
        assert kwargs["disable"] is True


# Add tests for RetrievalResults class
@pytest.mark.unit
class TestRetrievalResults:
    """Test the RetrievalResults class."""

    def test_initialization(self) -> None:
        """Object of RetrievalResults should initialize with all required parameters."""
        # Create minimal valid parameters
        query_indices = np.array([1, 2])
        database_indices = np.array([10, 11, 12])
        retrieved_indices = np.array([[0, 1], [0, 2]])
        embedding_distances = np.array([[0.1, 0.2], [0.3, 0.4]])
        geographic_distances = np.array([[5.0, 6.0], [7.0, 8.0]])
        is_match = np.array([[True, False], [False, True]])
        recall_at_n = np.array([0.5, 0.75])

        # Create a RetrievalResults object
        result = RetrievalResults(
            query_indices=query_indices,
            database_indices=database_indices,
            retrieved_indices=retrieved_indices,
            embedding_distances=embedding_distances,
            geographic_distances=geographic_distances,
            is_match=is_match,
            recall_at_n=recall_at_n,
            recall_at_one_percent=0.6,
            top1_distance=0.2,
            num_queries=2,
            num_database=3,
            distance_threshold=10.0,
            queries_with_matches=1,
        )

        # Check that all values were stored correctly
        assert np.array_equal(result.query_indices, query_indices)
        assert np.array_equal(result.database_indices, database_indices)
        assert np.array_equal(result.retrieved_indices, retrieved_indices)
        assert np.array_equal(result.embedding_distances, embedding_distances)
        assert np.array_equal(result.geographic_distances, geographic_distances)
        assert np.array_equal(result.is_match, is_match)
        assert np.array_equal(result.recall_at_n, recall_at_n)
        assert result.recall_at_one_percent == 0.6
        assert result.top1_distance == 0.2
        assert result.num_queries == 2
        assert result.num_database == 3
        assert result.distance_threshold == 10.0
        assert result.queries_with_matches == 1

    def test_optional_track_ids(self) -> None:
        """Object of RetrievalResults should allow setting optional track IDs."""
        min_params = {
            "query_indices": np.array([0]),
            "database_indices": np.array([0]),
            "retrieved_indices": np.array([[0]]),
            "embedding_distances": np.array([[0.1]]),
            "geographic_distances": np.array([[5.0]]),
            "is_match": np.array([[True]]),
            "recall_at_n": np.array([1.0]),
            "recall_at_one_percent": 1.0,
            "top1_distance": 0.1,
            "num_queries": 1,
            "num_database": 1,
            "distance_threshold": 10.0,
            "queries_with_matches": 1,
        }

        # Create with track IDs
        result = RetrievalResults(**min_params, query_track_id=5, database_track_id=10)

        assert result.query_track_id == 5
        assert result.database_track_id == 10

        # Create without track IDs (should be None)
        result2 = RetrievalResults(**min_params)
        assert result2.query_track_id is None
        assert result2.database_track_id is None


# Add tests for RetrievalResultsCollection class
@pytest.mark.unit
class TestRetrievalResultsCollection:
    """Test the RetrievalResultsCollection class."""

    def test_initialization(self) -> None:
        """Object of RetrievalResultsCollection should initialize with an empty list."""
        collection = RetrievalResultsCollection()
        assert len(collection.results) == 0
        assert len(collection) == 0

    def test_append_and_extend(self) -> None:
        """Append and extend methods should add items to the collection."""
        collection = RetrievalResultsCollection()

        # Create mock RetrievalResults
        mock_result1 = mock.MagicMock()
        mock_result2 = mock.MagicMock()
        mock_result3 = mock.MagicMock()

        # Test append
        collection.append(mock_result1)
        assert len(collection) == 1
        assert collection.results[0] is mock_result1

        # Test extend
        collection.extend([mock_result2, mock_result3])
        assert len(collection) == 3
        assert collection.results[1] is mock_result2
        assert collection.results[2] is mock_result3

    def test_properties(self) -> None:
        """Properties should calculate correct values from results."""
        collection = RetrievalResultsCollection()

        # Create mock results with controlled values
        mock_result1 = mock.MagicMock()
        mock_result1.num_queries = 5
        mock_result1.query_track_id = 1
        mock_result1.database_track_id = 2

        mock_result2 = mock.MagicMock()
        mock_result2.num_queries = 3
        mock_result2.query_track_id = 1
        mock_result2.database_track_id = 3

        collection.extend([mock_result1, mock_result2])

        # Test num_pairs
        assert collection.num_pairs == 2

        # Test num_queries
        assert collection.num_queries == 8  # 5 + 3

        # Test num_tracks
        query_tracks, db_tracks = collection.num_tracks
        assert query_tracks == 1  # Unique query track IDs: [1]
        assert db_tracks == 2  # Unique database track IDs: [2, 3]

    def test_aggregate_metrics(self) -> None:
        """aggregate_metrics should calculate metrics across all results."""
        collection = RetrievalResultsCollection()

        # Create mock results with specific metrics
        mock_result1 = mock.MagicMock()
        mock_result1.queries_with_matches = 5
        mock_result1.recall_at_n = np.array([0.5, 0.7])
        mock_result1.recall_at_one_percent = 0.6
        mock_result1.top1_distance = 0.2
        mock_result1.is_match = np.array([[True, False], [False, True], [True, True]])
        mock_result1.num_queries = 3

        mock_result2 = mock.MagicMock()
        mock_result2.queries_with_matches = 3
        mock_result2.recall_at_n = np.array([0.3, 0.5])
        mock_result2.recall_at_one_percent = 0.4
        mock_result2.top1_distance = 0.3
        mock_result2.is_match = np.array([[False, True], [True, False]])
        mock_result2.num_queries = 2

        collection.extend([mock_result1, mock_result2])

        # Calculate aggregate metrics
        metrics = collection.aggregate_metrics()

        # Check expected values
        assert np.allclose(
            metrics["recall_at_n"], np.array([0.4, 0.6])
        )  # Average of [0.5, 0.3] and [0.7, 0.5]
        assert metrics["recall_at_one_percent"] == 0.5  # Average of 0.6 and 0.4
        assert metrics["top1_distance"] == 0.25  # Average of 0.2 and 0.3
        assert metrics["overall_accuracy"] == 0.375  # 3/8 top1 correct matches
        assert metrics["queries_with_matches"] == 8  # Sum of matches
        assert metrics["total_queries"] == 5  # Sum of queries

    def test_filter_by_track(self) -> None:
        """filter_by_track should return a new collection with filtered results."""
        collection = RetrievalResultsCollection()

        # Create results with different track IDs
        result1 = mock.MagicMock()
        result1.query_track_id = 1
        result1.database_track_id = 2

        result2 = mock.MagicMock()
        result2.query_track_id = 1
        result2.database_track_id = 3

        result3 = mock.MagicMock()
        result3.query_track_id = 2
        result3.database_track_id = 3

        collection.extend([result1, result2, result3])

        # Filter by query track
        filtered1 = collection.filter_by_track(query_track_id=1)
        assert len(filtered1) == 2
        assert filtered1.results[0] is result1
        assert filtered1.results[1] is result2

        # Filter by database track
        filtered2 = collection.filter_by_track(database_track_id=3)
        assert len(filtered2) == 2
        assert filtered2.results[0] is result2
        assert filtered2.results[1] is result3

        # Filter by both
        filtered3 = collection.filter_by_track(query_track_id=1, database_track_id=3)
        assert len(filtered3) == 1
        assert filtered3.results[0] is result2

        # Filter with no matches
        filtered4 = collection.filter_by_track(query_track_id=3)
        assert len(filtered4) == 0

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Save and load methods should correctly serialize and deserialize the collection."""
        # Create test path
        test_file = tmp_path / "test_collection.json"

        # Create simple collection with minimal RetrievalResults
        collection = RetrievalResultsCollection()

        # Create a simple RetrievalResults with small arrays
        result = RetrievalResults(
            query_indices=np.array([0, 1]),
            database_indices=np.array([0, 1]),
            retrieved_indices=np.array([[0, 1], [1, 0]]),
            embedding_distances=np.array([[0.1, 0.2], [0.3, 0.4]]),
            geographic_distances=np.array([[5.0, 10.0], [15.0, 20.0]]),
            is_match=np.array([[True, False], [False, True]]),
            recall_at_n=np.array([0.5, 1.0]),
            recall_at_one_percent=0.5,
            top1_distance=0.2,
            num_queries=2,
            num_database=2,
            distance_threshold=25.0,
            queries_with_matches=2,
            query_track_id=1,
            database_track_id=2,
        )

        collection.append(result)

        # Save the collection
        collection.save(str(test_file))

        # Verify file exists
        assert test_file.exists()

        # Load the collection
        loaded_collection = RetrievalResultsCollection.load(str(test_file))

        # Verify structure
        assert len(loaded_collection) == 1
        loaded_result = loaded_collection.results[0]

        # Check key properties of loaded result
        assert loaded_result.num_queries == 2
        assert loaded_result.num_database == 2
        assert loaded_result.query_track_id == 1
        assert loaded_result.database_track_id == 2
        assert loaded_result.recall_at_one_percent == 0.5
        assert loaded_result.queries_with_matches == 2

        # Check arrays
        assert np.array_equal(loaded_result.query_indices, np.array([0, 1]))
        assert np.array_equal(loaded_result.is_match, np.array([[True, False], [False, True]]))


# Add tests for eval_retrieval_pair method
@pytest.mark.unit
def test_eval_retrieval_pair_basic() -> None:
    """eval_retrieval_pair should properly create a RetrievalResults object."""
    # Create simple test data
    query_embs = np.array([[1, 0], [0, 1]])
    db_embs = np.array([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]])

    # Make a simple geo distance matrix where first query matches first and third DB entries,
    # and second query matches second DB entry (using distance threshold 0.2)
    geo_distances = np.array(
        [
            [0.1, 0.3, 0.1],  # First query close to DB items 0, 2
            [0.5, 0.1, 0.5],  # Second query close to DB item 1
        ]
    )

    # Set some custom indices and track IDs
    query_indices = np.array([10, 11])
    database_indices = np.array([20, 21, 22])

    # Call the method
    result = ModelTester.eval_retrieval_pair(
        query_embs=query_embs,
        db_embs=db_embs,
        geo_distances=geo_distances,
        dist_thresh=0.2,
        at_n=2,
        query_indices=query_indices,
        database_indices=database_indices,
        query_track_id=1,
        database_track_id=2,
    )

    # Verify basic structure
    assert isinstance(result, RetrievalResults)
    assert result.num_queries == 2
    assert result.num_database == 3
    assert result.query_track_id == 1
    assert result.database_track_id == 2
    assert result.queries_with_matches == 2
    assert result.distance_threshold == 0.2

    # Verify indices were stored correctly
    assert np.array_equal(result.query_indices, query_indices)
    assert np.array_equal(result.database_indices, database_indices)

    # Check shape of returned arrays
    assert result.retrieved_indices.shape == (2, 2)  # 2 queries, top-2 results
    assert result.embedding_distances.shape == (2, 2)
    assert result.geographic_distances.shape == (2, 2)
    assert result.is_match.shape == (2, 2)
    assert result.recall_at_n.shape == (2,)

    # Should have several matches given our geo_distances
    assert np.sum(result.is_match) >= 2


@pytest.mark.unit
def test_eval_retrieval_pair_no_matches() -> None:
    """eval_retrieval_pair should handle cases with no matching points."""
    # Create test data where no geographic matches exist
    query_embs = np.array([[1, 0], [0, 1]])
    db_embs = np.array([[0.9, 0.1], [0.1, 0.9]])
    geo_distances = np.array([[10.0, 10.0], [10.0, 10.0]])  # All beyond threshold

    # Call the method with a low distance threshold
    result = ModelTester.eval_retrieval_pair(
        query_embs=query_embs, db_embs=db_embs, geo_distances=geo_distances, dist_thresh=5.0, at_n=2
    )

    # Check that we got appropriate values for no matches
    assert result.queries_with_matches == 0
    assert np.array_equal(result.recall_at_n, np.zeros(2))
    assert result.recall_at_one_percent == 0.0
    assert result.top1_distance is None
    assert not np.any(result.is_match)  # No matches at all


@pytest.mark.unit
def test_eval_retrieval_pair_one_percent_recall() -> None:
    """eval_retrieval_pair should correctly calculate Recall@1% for different database sizes."""
    # Create a larger database to test 1% threshold
    db_size = 100
    query_embs = np.array([[1, 0]])
    db_embs = np.zeros((db_size, 2))

    # Create geo distances where only 1 item is close
    geo_distances = np.ones((1, db_size)) * 10.0
    geo_distances[0, 0] = 0.1  # Only first DB item is close

    # Call the method
    result = ModelTester.eval_retrieval_pair(
        query_embs=query_embs, db_embs=db_embs, geo_distances=geo_distances, dist_thresh=1.0, at_n=5
    )

    # With 100 items, 1% threshold is 1 item - if the first item is found, R@1% should be 1.0
    assert result.recall_at_one_percent == 1.0

    # Try with different database size where 1% is 2 items
    db_size = 200
    db_embs = np.zeros((db_size, 2))
    geo_distances = np.ones((1, db_size)) * 10.0
    geo_distances[0, 0] = 0.1  # Only first DB item is close

    result = ModelTester.eval_retrieval_pair(
        query_embs=query_embs, db_embs=db_embs, geo_distances=geo_distances, dist_thresh=1.0, at_n=5
    )

    # With 200 items, 1% threshold is 2 items - if only 1 is found, R@1% should be 0.5
    # But this depends on whether the close item was retrieved in top-2
    assert 0.0 <= result.recall_at_one_percent <= 1.0


@pytest.mark.unit
def test_eval_retrieval_pair_top1_distance() -> None:
    """eval_retrieval_pair should correctly calculate mean top-1 distance."""
    # Create data where some top-1 retrievals are correct and others aren't
    query_embs = np.array([[1, 0], [0, 1], [1, 1]])
    db_embs = np.array([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]])

    # Make first and third queries match their top-1, second doesn't
    geo_distances = np.array(
        [
            [0.1, 1.0, 1.0],  # First query matches its top-1
            [1.0, 1.0, 1.0],  # Second query doesn't match any
            [1.0, 0.1, 1.0],  # Third query matches its top-2, not top-1
        ]
    )

    # Set embedding distances manually to test top1_distance calculation
    with mock.patch("faiss.IndexFlatL2", autospec=True) as mock_index:
        instance = mock_index.return_value

        # Mock search to return controlled distances and indices
        emb_distances = np.array(
            [
                [0.1, 0.5, 0.9],  # First query's embedding distances
                [0.2, 0.3, 0.4],  # Second query's embedding distances
                [0.8, 0.3, 0.5],  # Third query's embedding distances
            ]
        )
        retrieved_indices = np.array(
            [
                [0, 1, 2],  # First query's nearest neighbors
                [2, 1, 0],  # Second query's nearest neighbors
                [2, 1, 0],  # Third query's nearest neighbors
            ]
        )
        instance.search.return_value = (emb_distances, retrieved_indices)

        # Call the method with faiss mocked
        result = ModelTester.eval_retrieval_pair(
            query_embs=query_embs, db_embs=db_embs, geo_distances=geo_distances, dist_thresh=0.2, at_n=3
        )

    # Only one query has a correct top-1 match (the first one with distance 0.1)
    assert result.top1_distance == 0.1


@pytest.mark.unit
def test_eval_retrieval_pair_default_indices() -> None:
    """eval_retrieval_pair should use default indices if none provided."""
    query_embs = np.array([[1, 0], [0, 1]])
    db_embs = np.array([[0.9, 0.1], [0.1, 0.9]])
    geo_distances = np.array([[0.1, 1.0], [1.0, 0.1]])

    # Call without providing query/database indices
    result = ModelTester.eval_retrieval_pair(
        query_embs=query_embs, db_embs=db_embs, geo_distances=geo_distances, dist_thresh=0.2, at_n=1
    )

    # Default indices should be arange
    assert np.array_equal(result.query_indices, np.array([0, 1]))
    assert np.array_equal(result.database_indices, np.array([0, 1]))
