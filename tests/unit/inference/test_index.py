"""Unit tests for opr.inference.index.FaissFlatIndex.

These tests validate minimal load/search/meta behavior on a tiny synthetic dataset.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("faiss")


@pytest.mark.unit
def test_faiss_flat_index_load_and_properties(tmp_path: Path) -> None:
    """Test loading and basic properties of FaissFlatIndex."""
    # Arrange: tiny synthetic dataset
    N, D = 5, 3
    rng = np.random.default_rng(0)
    descriptors = rng.standard_normal((N, D), dtype=np.float32)
    poses = [[float(i), float(i + 1), float(i + 2), 0.0, 0.0, 0.0, 1.0] for i in range(N)]
    meta = pd.DataFrame(
        {
            "idx": np.arange(N, dtype=np.int64),
            "pose": poses,
        }
    )
    (tmp_path / "descriptors.npy").write_bytes(b"")  # create file
    np.save(tmp_path / "descriptors.npy", descriptors)
    meta.to_parquet(tmp_path / "meta.parquet")
    schema = {"version": "1", "dim": D, "metric": "l2", "created_at": "", "opr_version": ""}
    (tmp_path / "schema.json").write_text(json.dumps(schema))

    # Act
    from opr.inference.index import FaissFlatIndex, IndexMetric

    index = FaissFlatIndex.load(tmp_path)

    # Assert
    assert index.size() == N
    assert index.dim() == D
    assert index.metric() == IndexMetric.L2


@pytest.mark.unit
def test_faiss_flat_index_search_and_meta(tmp_path: Path) -> None:
    """Test search returns raw distances and row positions that map to idx/pose."""
    # Arrange
    N, D = 6, 4
    rng = np.random.default_rng(42)
    descriptors = rng.normal(size=(N, D)).astype(np.float32)
    poses = [[0.0, 0.0, float(i), 0.0, 0.0, 0.0, 1.0] for i in range(N)]
    meta = pd.DataFrame({"idx": np.arange(100, 100 + N, dtype=np.int64), "pose": poses})
    np.save(tmp_path / "descriptors.npy", descriptors)
    meta.to_parquet(tmp_path / "meta.parquet")
    schema = {"version": "1", "dim": D, "metric": "l2", "created_at": "", "opr_version": ""}
    (tmp_path / "schema.json").write_text(json.dumps(schema))

    from opr.inference.index import FaissFlatIndex

    index = FaissFlatIndex.load(tmp_path)

    # Use two query vectors
    Q = 2
    queries = descriptors[:Q] + 0.01  # small offset to avoid exact self-match ambiguity
    k = 3

    # Act
    inds, dists = index.search(queries, k)
    db_idx, db_pose = index.get_meta(inds[0])  # map first query's top-k

    # Assert shapes and types
    assert inds.shape == (Q, k)
    assert dists.shape == (Q, k)
    assert db_idx.shape == (k,)
    assert db_pose.shape == (k, 7)

    # Verify distance ordering corresponds to squared L2 distances (FAISS L2)
    def squared_l2(a: np.ndarray, b: np.ndarray) -> float:
        diff = a - b
        return float(np.dot(diff, diff))

    expected = np.array(
        [
            [squared_l2(queries[0], descriptors[i]) for i in range(N)],
            [squared_l2(queries[1], descriptors[i]) for i in range(N)],
        ]
    )
    # argsort by ascending distance and compare top-k indices
    expected_topk = np.argsort(expected, axis=1)[:, :k]
    np.testing.assert_array_equal(inds, expected_topk)
