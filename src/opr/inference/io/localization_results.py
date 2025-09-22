"""Utilities to serialize and persist localization results.

This module provides JSON Lines (JSONL) serialization for
`opr.inference.pipelines.localization.LocalizationResult` objects. JSONL stores
one JSON object per line and scales well for large datasets, streaming, and
append-only workflows. Files can be transparently compressed if the path ends
with ``.gz``.

Schema overview (per line):
    {
      "version": str,
      "chosen_idx": int,
      "candidates": [
        {
          "idx": int,
          "pr_distance": float,
          "db_pose": [tx, ty, tz, qx, qy, qz, qw],
          "db_pointcloud_path": str | null,
          "estimated_pose": [tx, ty, tz, qx, qy, qz, qw],
          "registration_confidence": float
        }, ...
      ]
    }

Note:
- ``numpy.ndarray`` fields are serialized to Python lists for portability.
- ``version`` mirrors `LocalizationResult.version` and should be preserved.

Public functions:
- ``save_localization_results_jsonl``: Write an iterable of results to a JSONL file.
- ``load_localization_results_jsonl``: Load all results from a JSONL file.
- ``iter_localization_results_jsonl``: Stream results from a JSONL file as an iterator.
- ``localization_result_to_dict`` / ``localization_result_from_dict``:
  conversion helpers to/from a plain JSON-serializable dictionary.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Mapping

import numpy as np

if TYPE_CHECKING:  # Import only for typing to avoid import cycles at runtime
    from opr.inference.pipelines.localization import (
        LocalizationResult,
        LocalizedCandidate,
    )


def _candidate_to_dict(candidate: "LocalizedCandidate") -> dict[str, Any]:
    """Convert a ``LocalizedCandidate`` to a JSON-serializable dict.

    Args:
        candidate: Candidate object produced by localization.

    Returns:
        Dict containing only JSON-serializable types.
    """
    return {
        "idx": int(candidate.idx),
        "pr_distance": float(candidate.pr_distance),
        "db_pose": np.asarray(candidate.db_pose, dtype=np.float64).tolist(),
        "db_pointcloud_path": candidate.db_pointcloud_path,
        "estimated_pose": np.asarray(candidate.estimated_pose, dtype=np.float64).tolist(),
        "registration_confidence": float(candidate.registration_confidence),
    }


def _candidate_from_dict(payload: Mapping[str, Any]) -> "LocalizedCandidate":
    """Construct a ``LocalizedCandidate`` from a dict.

    Args:
        payload: Mapping with the keys described in the module schema.

    Returns:
        A ``LocalizedCandidate`` instance with ``numpy.ndarray`` fields restored.
    """
    # Local import prevents import cycles at module import time
    from opr.inference.pipelines.localization import LocalizedCandidate

    return LocalizedCandidate(
        idx=int(payload["idx"]),
        pr_distance=float(payload["pr_distance"]),
        db_pose=np.asarray(payload["db_pose"], dtype=np.float64),
        db_pointcloud_path=payload.get("db_pointcloud_path"),
        estimated_pose=np.asarray(payload["estimated_pose"], dtype=np.float64),
        registration_confidence=float(payload["registration_confidence"]),
    )


def localization_result_to_dict(result: "LocalizationResult") -> dict[str, Any]:
    """Convert a ``LocalizationResult`` to a JSON-serializable dict.

    Args:
        result: Result object to convert.

    Returns:
        A dictionary suitable for ``json.dumps``.

    Examples:
        Basic conversion and JSON serialization:

        >>> from opr.inference.io import localization_result_to_dict
        >>> # assume `result` is a LocalizationResult produced by the pipeline
        >>> payload = localization_result_to_dict(result)
        >>> json_string = json.dumps(payload, ensure_ascii=False)
    """
    return {
        "version": str(result.version),
        "chosen_idx": int(result.chosen_idx),
        "candidates": [_candidate_to_dict(c) for c in result.candidates],
    }


def localization_result_from_dict(payload: Mapping[str, Any]) -> "LocalizationResult":
    """Reconstruct a ``LocalizationResult`` from a dictionary.

    Args:
        payload: Mapping produced by :func:`localization_result_to_dict`.

    Returns:
        The reconstructed ``LocalizationResult``.

    Examples:
        Reconstruct from a previously created dictionary:

        >>> from opr.inference.io import (
        ...     localization_result_to_dict, localization_result_from_dict,
        ... )
        >>> # payload was obtained via `localization_result_to_dict(result)`
        >>> result_rt = localization_result_from_dict(payload)
    """
    # Local import prevents import cycles at module import time
    from opr.inference.pipelines.localization import LocalizationResult

    return LocalizationResult(
        version=str(payload["version"]),
        candidates=[_candidate_from_dict(cd) for cd in payload["candidates"]],
        chosen_idx=int(payload["chosen_idx"]),
    )


def save_localization_results_jsonl(
    results: Iterable["LocalizationResult"],
    path: str | Path,
    *,
    append: bool = False,
    compress: bool | None = None,
) -> None:
    """Write localization results to a JSONL file (optionally ``.gz``).

    Each item from ``results`` is converted with
    :func:`localization_result_to_dict` and written as a single JSON object per
    line. The function is streaming and does not require all results to fit into
    memory.

    Compression is enabled automatically when the path ends with ``.gz`` or when
    ``compress=True`` is passed explicitly.

    Args:
        results: Iterable of localization results to write.
        path: Destination file path. If it ends with ``.gz``, gzip compression is
            used automatically.
        append: If True, append to the file instead of overwriting.
        compress: Force gzip compression (True) or disable it (False).

    Examples:
        Save two results to an uncompressed JSONL file:

        >>> from opr.inference.io import save_localization_results_jsonl
        >>> save_localization_results_jsonl([res1, res2], "/tmp/loc.jsonl")

        Save with gzip compression (auto by .gz suffix):

        >>> save_localization_results_jsonl([res1, res2], "/tmp/loc.jsonl.gz")

        Append more results to the same file:

        >>> save_localization_results_jsonl([res3], "/tmp/loc.jsonl", append=True)
    """
    file_path = str(path)
    if compress is None:
        compress = file_path.endswith(".gz")

    opener = gzip.open if compress else open
    mode = "at" if append else "wt"

    with opener(file_path, mode, encoding="utf-8") as f:
        for res in results:
            obj = localization_result_to_dict(res)
            f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")


def iter_localization_results_jsonl(path: str | Path) -> Iterator["LocalizationResult"]:
    """Iterate over a JSONL file with localization results.

    This function streams one result at a time and is suitable for large files.
    Gzip-compressed files are detected by the ``.gz`` suffix.

    Args:
        path: Source file path (``.jsonl`` or ``.jsonl.gz``).

    Yields:
        ``LocalizationResult`` objects reconstructed from each JSON line.

    Examples:
        Stream and process results one by one (works for ``.gz`` too):

        >>> from opr.inference.io import iter_localization_results_jsonl
        >>> for res in iter_localization_results_jsonl("/tmp/loc.jsonl.gz"):
        ...     _ = res.chosen_idx  # process result
    """
    file_path = str(path)
    compress = file_path.endswith(".gz")
    opener = gzip.open if compress else open

    with opener(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            yield localization_result_from_dict(payload)


def load_localization_results_jsonl(path: str | Path) -> list["LocalizationResult"]:
    """Load all localization results from a JSONL file.

    This is a convenience wrapper around
    :func:`iter_localization_results_jsonl` that collects all items into a list.

    Args:
        path: Source file path (``.jsonl`` or ``.jsonl.gz``).

    Returns:
        List of ``LocalizationResult`` objects.

    Examples:
        Load all results into memory:

        >>> from opr.inference.io import load_localization_results_jsonl
        >>> results = load_localization_results_jsonl("/tmp/loc.jsonl")
    """
    return list(iter_localization_results_jsonl(path))
