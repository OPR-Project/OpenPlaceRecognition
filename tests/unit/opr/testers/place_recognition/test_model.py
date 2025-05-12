"""Test ModelTester class for place recognition."""

from typing import Callable, Iterator
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch

from opr.testers.place_recognition.model import ModelTester


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
    """_eval_pairs should handle basic track pairs correctly."""
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

    # Mock get_recalls to return controlled values
    with mock.patch.object(tester, "get_recalls") as mock_recalls:
        # Return different values for the two track pairs
        mock_recalls.side_effect = [
            (np.array([0.5, 1.0]), 1.0, 0.1),  # Track 0 -> 1
            (np.array([0.7, 0.9]), 0.9, 0.2),  # Track 1 -> 0
        ]

        # Call the method
        recalls_at_n, recalls_at_1p, top1_dists = tester._eval_pairs(embs, queries, databases, geo_dist)

    # Check shapes
    assert recalls_at_n.shape == (2, 2, 2)  # 2 tracks x 2 tracks x at_n=2
    assert recalls_at_1p.shape == (2, 2)
    assert top1_dists.shape == (2, 2)

    # Check values - should have data only at (0,1) and (1,0) positions
    # Track pairs (0,0) and (1,1) aren't computed due to i!=j requirement
    assert np.isnan(recalls_at_n[0, 0, 0])  # No self-comparison
    assert np.isnan(recalls_at_n[1, 1, 0])  # No self-comparison

    assert np.array_equal(recalls_at_n[0, 1], np.array([0.5, 1.0]))
    assert np.array_equal(recalls_at_n[1, 0], np.array([0.7, 0.9]))

    assert recalls_at_1p[0, 1] == 1.0
    assert recalls_at_1p[1, 0] == 0.9

    assert top1_dists[0, 1] == 0.1
    assert top1_dists[1, 0] == 0.2


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

    # Mock get_recalls to return None for top1_distance
    with mock.patch.object(tester, "get_recalls") as mock_recalls:
        mock_recalls.side_effect = [
            (np.array([0.0]), 0.0, None),  # Track 0 -> 1, no top1 matches
            (np.array([0.5]), 0.5, 0.1),  # Track 1 -> 0, has top1 match
        ]

        recalls_at_n, recalls_at_1p, top1_dists = tester._eval_pairs(embs, queries, databases, geo_dist)

    # Check top1_dists values - should have NaN for the first pair
    assert np.isnan(top1_dists[0, 1])
    assert top1_dists[1, 0] == 0.1


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

    # Mock get_recalls
    with mock.patch.object(tester, "get_recalls") as mock_recalls:
        mock_recalls.return_value = (np.zeros(tester.at_n), 0.0, None)

        # This should not raise errors despite empty track
        recalls_at_n, recalls_at_1p, top1_dists = tester._eval_pairs(embs, queries, databases, geo_dist)

    # Check that the method completed
    assert isinstance(recalls_at_n, np.ndarray)
    assert isinstance(recalls_at_1p, np.ndarray)
    assert isinstance(top1_dists, np.ndarray)


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
                    # Mocked evaluation results
                    mock_recalls_n = np.ones((2, 2, tester.at_n)) * 0.5
                    mock_recalls_1p = np.ones((2, 2)) * 0.7
                    mock_top1_dist = np.ones((2, 2)) * 3.0
                    mock_eval.return_value = (mock_recalls_n, mock_recalls_1p, mock_top1_dist)

                    with mock.patch.object(tester, "_aggregate") as mock_agg:
                        # Mocked aggregation results (now scalars, not arrays)
                        expected_recalls_n = np.ones(tester.at_n) * 0.8
                        expected_recalls_1p = 0.9  # Now a scalar
                        expected_top1_dist = 2.5  # Now a scalar
                        mock_agg.return_value = (expected_recalls_n, expected_recalls_1p, expected_top1_dist)

                        # Run the method
                        recalls_n, recalls_1p, top1_dist = tester.run()

    # Check that all methods were called with the right arguments
    mock_extract.assert_called_once()
    mock_group.assert_called_once()
    mock_dist.assert_called_once()
    mock_eval.assert_called_once()
    mock_agg.assert_called_once()

    # Check method call order through argument passing
    mock_eval_args = mock_eval.call_args[0]
    assert np.array_equal(mock_eval_args[0], mock_extract.return_value)
    assert mock_eval_args[1:3] == mock_group.return_value
    assert np.array_equal(mock_eval_args[3], mock_dist.return_value)

    mock_agg_args = mock_agg.call_args[0]
    assert np.array_equal(mock_agg_args[0], mock_recalls_n)
    assert np.array_equal(mock_agg_args[1], mock_recalls_1p)
    assert np.array_equal(mock_agg_args[2], mock_top1_dist)

    # Verify final results are returned correctly
    assert np.array_equal(recalls_n, expected_recalls_n)
    assert recalls_1p == expected_recalls_1p
    assert top1_dist == expected_top1_dist


@pytest.mark.unit
def test_run_end_to_end(mock_model: torch.nn.Module, mock_dataloader: Callable) -> None:
    """run() should execute a complete evaluation workflow with minimal data."""
    # Create a minimal dataset with two tracks
    df = pd.DataFrame({"x": [0, 1, 2, 3], "y": [0, 1, 2, 3], "track": [0, 0, 1, 1]})
    dl = mock_dataloader(df, batch_size=2)

    # We'll use the actual implementation but with a small at_n value
    tester = ModelTester(mock_model, dl, at_n=2)

    # Run the full evaluation
    recalls_n, recalls_1p, top1_dist = tester.run()

    # Check output shapes and types - now we expect scalars for recalls_1p and top1_dist
    assert isinstance(recalls_n, np.ndarray)
    assert recalls_n.shape == (2,)  # 2 values for at_n
    assert np.isscalar(recalls_1p)  # Now a scalar
    assert np.isscalar(top1_dist)  # Now a scalar

    # The mock model returns all-zero embeddings, so distances
    # between all embeddings will be zero. In the geographic space,
    # points are at different locations, so there may or may not
    # be matches depending on dist_thresh. We're mostly checking
    # that the method runs without errors.
