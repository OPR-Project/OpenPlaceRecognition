"""Unit tests for localization results JSONL IO utilities."""

import json
from pathlib import Path

import numpy as np
import pytest

from opr.inference.io import (
    iter_localization_results_jsonl,
    load_localization_results_jsonl,
    localization_result_from_dict,
    localization_result_to_dict,
    save_localization_results_jsonl,
)
from opr.inference.pipelines.localization import LocalizationResult, LocalizedCandidate


def _make_candidate(idx: int, path: str | None) -> LocalizedCandidate:
    return LocalizedCandidate(
        idx=idx,
        pr_distance=0.123 * (idx + 1),
        db_pose=np.array([1, 2, 3, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        db_pointcloud_path=path,
        estimated_pose=np.array([4, 5, 6, 0.5, 0.5, 0.5, 0.5], dtype=np.float64),
        registration_confidence=0.9,
    )


def _make_result(version: str = "1", base_idx: int = 10) -> LocalizationResult:
    c1 = _make_candidate(base_idx, "db/pc/0001.pcd")
    c2 = _make_candidate(base_idx + 1, None)
    return LocalizationResult(version=version, candidates=[c1, c2], chosen_idx=c1.idx)


def _assert_results_equal(a: LocalizationResult, b: LocalizationResult) -> None:
    assert a.version == b.version
    assert a.chosen_idx == b.chosen_idx
    assert len(a.candidates) == len(b.candidates)
    for ca, cb in zip(a.candidates, b.candidates):
        assert ca.idx == cb.idx
        assert ca.db_pointcloud_path == cb.db_pointcloud_path
        assert ca.pr_distance == pytest.approx(cb.pr_distance, rel=0, abs=1e-12)
        assert ca.registration_confidence == pytest.approx(cb.registration_confidence, rel=0, abs=1e-12)
        np.testing.assert_allclose(ca.db_pose, cb.db_pose)
        np.testing.assert_allclose(ca.estimated_pose, cb.estimated_pose)


@pytest.mark.unit
def test_localization_result_dict_roundtrip() -> None:
    """Ensure dict conversion is JSON-safe and round-trips."""
    res = _make_result(version="1", base_idx=100)
    data = localization_result_to_dict(res)

    # Ensure JSON-serializable
    s = json.dumps(data, ensure_ascii=False)
    data2 = json.loads(s)

    res_rt = localization_result_from_dict(data2)
    _assert_results_equal(res, res_rt)


@pytest.mark.unit
def test_save_and_load_jsonl(tmp_path: Path) -> None:
    """Save to JSONL and load all results back."""
    res1 = _make_result(version="1", base_idx=1)
    res2 = _make_result(version="2", base_idx=3)
    path = tmp_path / "loc_results.jsonl"

    save_localization_results_jsonl([res1, res2], path)
    loaded = load_localization_results_jsonl(path)

    assert len(loaded) == 2
    _assert_results_equal(res1, loaded[0])
    _assert_results_equal(res2, loaded[1])


@pytest.mark.unit
def test_save_and_iter_jsonl_gz(tmp_path: Path) -> None:
    """Save gzip JSONL and stream results via iterator."""
    res1 = _make_result(version="1", base_idx=10)
    res2 = _make_result(version="1", base_idx=20)
    path = tmp_path / "loc_results.jsonl.gz"

    save_localization_results_jsonl([res1, res2], path)
    iter_loaded = list(iter_localization_results_jsonl(path))

    assert len(iter_loaded) == 2
    _assert_results_equal(res1, iter_loaded[0])
    _assert_results_equal(res2, iter_loaded[1])


@pytest.mark.unit
def test_append_mode_jsonl(tmp_path: Path) -> None:
    """Append to existing JSONL and validate order and content."""
    res1 = _make_result(version="1", base_idx=7)
    res2 = _make_result(version="1", base_idx=8)
    path = tmp_path / "loc_results_append.jsonl"

    save_localization_results_jsonl([res1], path)
    save_localization_results_jsonl([res2], path, append=True)

    loaded = load_localization_results_jsonl(path)
    assert len(loaded) == 2
    _assert_results_equal(res1, loaded[0])
    _assert_results_equal(res2, loaded[1])
