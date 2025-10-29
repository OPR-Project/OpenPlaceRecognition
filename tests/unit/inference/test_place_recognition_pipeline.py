"""Unit tests for the new inference PlaceRecognitionPipeline."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

pytest.importorskip("faiss")


class _StubPRModel(nn.Module):
    """Stub model that returns a fixed descriptor as `final_descriptor`."""

    def __init__(self, descriptor: np.ndarray) -> None:
        super().__init__()
        self._desc = torch.from_numpy(descriptor.astype(np.float32, copy=False))

    def forward(self, _: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"final_descriptor": self._desc.unsqueeze(0)}


@pytest.mark.unit
def test_place_recognition_pipeline_end_to_end(tmp_path: Path) -> None:
    """Test end-to-end: build index, run pipeline, and validate outputs."""
    # Arrange: tiny database
    N, D = 6, 4
    rng = np.random.default_rng(123)
    descriptors = rng.normal(size=(N, D)).astype(np.float32)
    poses = [[float(i), float(i + 1), float(i + 2), 0.0, 0.0, 0.0, 1.0] for i in range(N)]
    meta = pd.DataFrame({"idx": np.arange(200, 200 + N, dtype=np.int64), "pose": poses})
    np.save(tmp_path / "descriptors.npy", descriptors)
    meta.to_parquet(tmp_path / "meta.parquet")
    schema = {"version": "1", "dim": D, "metric": "l2", "created_at": "", "opr_version": ""}
    (tmp_path / "schema.json").write_text(json.dumps(schema))

    # Load index
    from opr.inference.index import FaissFlatIndex

    index = FaissFlatIndex.load(tmp_path)

    # Create stub model producing a query close to the first DB vector
    query_desc = descriptors[0] + 0.01
    model = _StubPRModel(query_desc)

    # Build pipeline
    from opr.inference.pipelines import PlaceRecognitionPipeline

    pipeline = PlaceRecognitionPipeline(index=index, model=model, device="cpu")

    # Act
    result = pipeline.infer(input_data={}, k=3)

    # Assert: shapes
    assert result.descriptor.shape == (D,)
    assert result.indices.shape == (3,)
    assert result.distances.shape == (3,)
    assert result.db_idx.shape == (3,)
    assert result.db_pose.shape == (3, 7)

    # Assert: top-k correctness via L2 distances
    def l2_sq(a: np.ndarray, b: np.ndarray) -> float:
        diff = a - b
        return float(np.dot(diff, diff))

    expected_order = np.argsort([l2_sq(result.descriptor, d) for d in descriptors])[:3]
    np.testing.assert_array_equal(result.indices, expected_order)
    np.testing.assert_array_equal(result.db_idx, meta["idx"].to_numpy()[expected_order])
