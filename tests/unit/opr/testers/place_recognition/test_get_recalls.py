"""Test get_recalls function in ModelTester class."""

from unittest import mock

import numpy as np
import pytest

from opr.testers.place_recognition.model import ModelTester


@pytest.mark.unit
def test_get_recalls_basic_functionality() -> None:
    """Test get_recalls with a simple case that should have perfect recall."""
    # 3 queries, 4 database items, 2D embeddings
    query_embs = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float32)
    db_embs = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.float32)

    # Distance matrix: first 3 queries match first 3 database items perfectly
    # Values < 1.0 will be considered matches (with dist_thresh=1.0)
    dist_matrix = np.array(
        [
            [0.0, 2.0, 3.0, 4.0],  # First query matches first db
            [2.0, 0.0, 2.0, 3.0],  # Second query matches second db
            [3.0, 2.0, 0.0, 2.0],  # Third query matches third db
        ]
    )

    # With dist_thresh=1.0, only diagonal elements are matches
    with mock.patch("opr.testers.place_recognition.model.faiss_available", False):
        recall_at_n, recall_1p, mean_top1 = ModelTester.get_recalls(
            query_embs, db_embs, dist_matrix, dist_thresh=1.0, at_n=3
        )

    # Recall should be perfect for all N values
    assert np.allclose(recall_at_n, [1.0, 1.0, 1.0])
    assert recall_1p == 1.0
    assert mean_top1 == 0.0  # Perfect embedding match = 0 distance


@pytest.mark.unit
def test_get_recalls_no_matches() -> None:
    """Test when no geographic matches exist."""
    query_embs = np.array([[0, 0], [1, 1]], dtype=np.float32)
    db_embs = np.array([[2, 2], [3, 3]], dtype=np.float32)

    # All geographic distances > dist_thresh
    dist_matrix = np.array(
        [
            [5.0, 6.0],
            [4.0, 5.0],
        ]
    )

    with mock.patch("opr.testers.place_recognition.model.faiss_available", False):
        recall_at_n, recall_1p, mean_top1 = ModelTester.get_recalls(
            query_embs, db_embs, dist_matrix, dist_thresh=3.0, at_n=2
        )

    assert np.allclose(recall_at_n, [0.0, 0.0])
    assert recall_1p == 0.0
    assert mean_top1 is None  # No matches found


@pytest.mark.unit
def test_get_recalls_one_percent_calculation() -> None:
    """Test that one_percent_threshold is calculated correctly."""
    # Create 200 database items - 1% should be 2 items
    query_embs = np.array([[0, 0]], dtype=np.float32)
    db_embs = np.zeros((200, 2), dtype=np.float32)

    # First 3 items are matches
    dist_matrix = np.ones((1, 200)) * 10.0
    dist_matrix[0, :3] = 0.5  # First 3 are matches

    with mock.patch("opr.testers.place_recognition.model.faiss_available", False):
        # Mock KDTree to control the nearest neighbors
        with mock.patch("sklearn.neighbors.KDTree") as mock_kdtree:
            mock_tree = mock.MagicMock()
            mock_tree.query.return_value = (
                np.array([[0.1, 0.2, 0.3]]),  # Distances
                np.array([[0, 1, 2]]),  # Indices (first 3 items)
            )
            mock_kdtree.return_value = mock_tree

            # at_n=1, but one_percent=2 should be used
            recall_at_n, recall_1p, mean_top1 = ModelTester.get_recalls(
                query_embs, db_embs, dist_matrix, dist_thresh=1.0, at_n=1
            )

    # Should have expanded at_n to match one_percent_threshold
    assert len(recall_at_n) == 1  # Still returns requested at_n length
    assert recall_1p > 0.0  # One percent recall should be calculated


@pytest.mark.unit
def test_get_recalls_with_faiss() -> None:
    """Test the FAISS codepath."""
    query_embs = np.array([[0, 0], [1, 1]], dtype=np.float32)
    db_embs = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float32)

    dist_matrix = np.array(
        [
            [0.1, 2.0, 3.0],
            [2.0, 0.1, 2.0],
        ]
    )

    with mock.patch("opr.testers.place_recognition.model.faiss_available", True):
        with mock.patch("faiss.IndexFlatL2") as mock_index:
            mock_instance = mock.MagicMock()
            mock_instance.search.return_value = (
                np.array([[0.0, 2.0], [0.0, 2.0]]),  # Distances
                np.array([[0, 2], [1, 2]]),  # Indices
            )
            mock_index.return_value = mock_instance

            recall_at_n, recall_1p, mean_top1 = ModelTester.get_recalls(
                query_embs, db_embs, dist_matrix, dist_thresh=1.0, at_n=2
            )

    # Should have found matches at position 0
    assert np.allclose(recall_at_n, [1.0, 1.0])
    assert mean_top1 == 0.0


@pytest.mark.unit
def test_get_recalls_partial_matches() -> None:
    """Test when some queries have matches but others don't."""
    query_embs = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float32)
    db_embs = np.array([[0, 0], [3, 3], [4, 4]], dtype=np.float32)

    # Only first query has a geographic match
    dist_matrix = np.array(
        [
            [0.5, 5.0, 6.0],  # Match with first db
            [5.0, 5.0, 6.0],  # No matches
            [6.0, 6.0, 5.0],  # No matches
        ]
    )

    with mock.patch("opr.testers.place_recognition.model.faiss_available", False):
        with mock.patch("sklearn.neighbors.KDTree") as mock_kdtree:
            mock_tree = mock.MagicMock()
            # Perfect retrieval - each query finds its corresponding embedding
            mock_tree.query.return_value = (
                np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.0, 0.0]]),  # Distances
                np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]),  # Indices
            )
            mock_kdtree.return_value = mock_tree

            recall_at_n, recall_1p, mean_top1 = ModelTester.get_recalls(
                query_embs, db_embs, dist_matrix, dist_thresh=1.0, at_n=3
            )

    # Only the first query has a match, and it's the first result
    assert np.allclose(recall_at_n, [1.0, 1.0, 1.0])
    assert recall_1p == 1.0
    assert mean_top1 == 0.0
