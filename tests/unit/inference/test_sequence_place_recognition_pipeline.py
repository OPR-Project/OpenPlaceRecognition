"""Unit tests for the streaming SequencePlaceRecognitionPipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

pytest.importorskip("faiss")


class _UpdatingStubPRModel(nn.Module):
    """Stub model whose output descriptor can be updated between calls."""

    def __init__(self, descriptor: np.ndarray | None = None) -> None:
        super().__init__()
        if descriptor is not None:
            self._desc = torch.from_numpy(descriptor.astype(np.float32, copy=False))
        else:
            self._desc = None

    def set_descriptor(self, descriptor: np.ndarray) -> None:
        self._desc = torch.from_numpy(descriptor.astype(np.float32, copy=False))

    def forward(self, _: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        assert self._desc is not None, "Descriptor must be set before forward()"
        return {"final_descriptor": self._desc.unsqueeze(0)}


def _build_tiny_index(tmp_path: Path, descriptors: np.ndarray) -> Any:
    N, D = descriptors.shape
    poses = [[float(i), float(i + 1), float(i + 2), 0.0, 0.0, 0.0, 1.0] for i in range(N)]
    meta = pd.DataFrame({"idx": np.arange(1000, 1000 + N, dtype=np.int64), "pose": poses})
    np.save(tmp_path / "descriptors.npy", descriptors)
    meta.to_parquet(tmp_path / "meta.parquet")
    schema = {"version": "1", "dim": D, "metric": "l2", "created_at": "", "opr_version": ""}
    (tmp_path / "schema.json").write_text(json.dumps(schema))

    from opr.inference.index import FaissFlatIndex

    return FaissFlatIndex.load(tmp_path)


@pytest.mark.unit
def test_cold_start_matches_single_frame(tmp_path: Path) -> None:
    """With one frame in the window, results should match single-frame pipeline."""
    rng = np.random.default_rng(42)
    N, D = 8, 6
    db = rng.normal(size=(N, D)).astype(np.float32)
    index = _build_tiny_index(tmp_path, db)

    query = db[0] + 0.01
    model = _UpdatingStubPRModel(query)

    from opr.inference.pipelines import (
        PlaceRecognitionPipeline,
        SequencePlaceRecognitionPipeline,
    )

    single = PlaceRecognitionPipeline(index=index, model=model, device="cpu")
    seq = SequencePlaceRecognitionPipeline(
        index=index, model=model, device="cpu", max_window=5, per_frame_k=5, final_k=5
    )

    res_single = single.infer({}, k=3)
    res_seq = seq.infer({}, k=3)

    np.testing.assert_array_equal(res_seq.indices, res_single.indices)
    np.testing.assert_array_equal(res_seq.distances, res_single.distances)
    np.testing.assert_allclose(res_seq.descriptor, query.astype(np.float32))
    assert res_seq.db_idx.shape == (3,)
    assert res_seq.db_pose.shape == (3, 7)


@pytest.mark.unit
def test_fusion_deduplicates_and_sorts(tmp_path: Path) -> None:
    """Two identical frames cause duplicate candidates; CPF keeps unique, best-first."""
    # Construct a simple 1D axis so nearest neighbors are obvious
    db = np.stack([np.array([i, 0.0], dtype=np.float32) for i in range(10)], axis=0)
    index = _build_tiny_index(tmp_path, db)

    # Both frames query near 1.0 ⇒ top neighbors ~ [1, 0, 2]
    q = np.array([1.0, 0.0], dtype=np.float32)
    model = _UpdatingStubPRModel()

    from opr.inference.pipelines import SequencePlaceRecognitionPipeline

    seq = SequencePlaceRecognitionPipeline(
        index=index, model=model, device="cpu", max_window=5, per_frame_k=3, final_k=3
    )

    model.set_descriptor(q)
    _ = seq.infer({})
    model.set_descriptor(q)
    res, debug = seq.infer({}, return_debug=True)

    # Expected fused indices: [1, 0, 2]
    expected = np.array([1, 0, 2], dtype=np.int64)
    np.testing.assert_array_equal(res.indices, expected)
    assert res.distances.shape == (3,)
    # Debug shapes
    assert debug.per_frame_indices.shape == (2, 3)
    assert debug.per_frame_distances.shape == (2, 3)


@pytest.mark.unit
def test_window_trimming_and_mean_aggregation(tmp_path: Path) -> None:
    """Mean aggregation should reflect only the frames inside the window."""
    rng = np.random.default_rng(7)
    N, D = 12, 4
    db = rng.normal(size=(N, D)).astype(np.float32)
    index = _build_tiny_index(tmp_path, db)

    from opr.inference.pipelines import SequencePlaceRecognitionPipeline

    model = _UpdatingStubPRModel()
    seq = SequencePlaceRecognitionPipeline(
        index=index, model=model, device="cpu", max_window=2, per_frame_k=2, final_k=1, descriptor_agg="mean"
    )

    q1 = rng.normal(size=(D,)).astype(np.float32)
    q2 = rng.normal(size=(D,)).astype(np.float32)
    q3 = rng.normal(size=(D,)).astype(np.float32)

    model.set_descriptor(q1)
    res1 = seq.infer({})
    np.testing.assert_allclose(res1.descriptor, q1, atol=1e-6)

    model.set_descriptor(q2)
    res2 = seq.infer({})
    np.testing.assert_allclose(res2.descriptor, (q1 + q2) / 2.0, atol=1e-6)

    model.set_descriptor(q3)
    res3 = seq.infer({})
    np.testing.assert_allclose(res3.descriptor, (q2 + q3) / 2.0, atol=1e-6)


@pytest.mark.unit
def test_k_override_and_metadata_shapes(tmp_path: Path) -> None:
    """K override should limit output shapes; metadata arrays match k."""
    rng = np.random.default_rng(99)
    N, D = 9, 5
    db = rng.normal(size=(N, D)).astype(np.float32)
    index = _build_tiny_index(tmp_path, db)

    from opr.inference.pipelines import SequencePlaceRecognitionPipeline

    model = _UpdatingStubPRModel(db[3])
    seq = SequencePlaceRecognitionPipeline(
        index=index, model=model, device="cpu", max_window=5, per_frame_k=5, final_k=5
    )

    res = seq.infer({}, k=2)
    assert res.indices.shape == (2,)
    assert res.distances.shape == (2,)
    assert res.db_idx.shape == (2,)
    assert res.db_pose.shape == (2, 7)
