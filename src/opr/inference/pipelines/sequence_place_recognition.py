"""Streaming sequence Place Recognition pipeline with FIFO window and CPF.

This module implements a streaming sequence-aware place recognition pipeline that
accepts single frames, caches per-frame search results, and fuses candidates via
Candidate Pool Fusion (CPF). It is compatible with `LocalizationPipeline` by
returning a `PlaceRecognitionResult`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Literal

import numpy as np
import torch
from torch import Tensor, nn

from opr.inference.index import Index
from opr.inference.pipelines.place_recognition import PlaceRecognitionResult
from opr.utils import init_model, parse_device


@dataclass
class SequencePRDebug:
    """Optional debug information returned by the sequence pipeline."""

    per_frame_indices: np.ndarray  # [N, per_k]
    per_frame_distances: np.ndarray  # [N, per_k]
    fused_indices: np.ndarray  # [final_k]
    fused_distances: np.ndarray  # [final_k]
    window_size: int
    descriptor_agg: str


def _candidate_pool_fusion(
    distances: np.ndarray,  # [N, per_k]
    indices: np.ndarray,  # [N, per_k]
    final_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse per-frame candidates using Candidate Pool Fusion.

    Args:
        distances: Per-frame raw distances (smaller is better), shape [N, per_k].
        indices: Per-frame internal row ids, shape [N, per_k].
        final_k: Number of final fused candidates to return.

    Returns:
        (fused_distances, fused_indices): Both of shape [final_k] (or shorter if not enough uniques).
    """
    if distances.size == 0 or indices.size == 0:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)

    flat_d = distances.reshape(-1)
    flat_i = indices.reshape(-1)

    order = np.argsort(flat_d)
    sorted_d = flat_d[order]
    sorted_i = flat_i[order]

    # Deduplicate by first occurrence (best distance)
    unique_i, first_pos = np.unique(sorted_i, return_index=True)
    uniq_d = sorted_d[first_pos]

    # Re-sort by distance because np.unique does not preserve order
    re_order = np.argsort(uniq_d)
    fused_i = unique_i[re_order]
    fused_d = uniq_d[re_order]

    if fused_i.shape[0] > final_k:
        fused_i = fused_i[:final_k]
        fused_d = fused_d[:final_k]

    # Ensure dtypes
    fused_d = fused_d.astype(np.float32, copy=False)
    fused_i = fused_i.astype(np.int64, copy=False)
    return fused_d, fused_i


class SequencePlaceRecognitionPipeline:
    """Streaming sequence-aware Place Recognition with a single-frame model.

    Maintains a FIFO window of recent frames. Each call processes one frame,
    caches its descriptor and per-frame top-k, then fuses across the window
    using Candidate Pool Fusion. Descriptor is aggregated across the window.
    """

    def __init__(
        self,
        index: Index,
        model: nn.Module,
        model_weights_path: str | Path | None = None,
        device: str | int | torch.device = "cpu",
        max_window: int = 20,
        per_frame_k: int = 10,
        final_k: int = 10,
        descriptor_agg: Literal["mean", "ema", "last"] = "mean",
        ema_decay: float = 0.9,
        recency_weighting: Literal["none", "linear", "exp"] = "none",
    ) -> None:
        """Initialize the streaming sequence pipeline.

        Args:
            index: Loaded `Index` instance that provides search and metadata.
            model: Single-frame PyTorch model that outputs `{"final_descriptor": Tensor[B,D]}`.
            model_weights_path: Optional path to model weights.
            device: Torch device specification.
            max_window: Maximum number of recent frames kept in the FIFO window.
            per_frame_k: Top-k to retrieve per frame (cached).
            final_k: Final fused top-k to return after CPF.
            descriptor_agg: Aggregation strategy for descriptors across the window.
            ema_decay: EMA decay used when `descriptor_agg="ema"`.
            recency_weighting: Optional recency weighting policy for CPF distances.
        """
        self.index = index
        self.device = parse_device(device)
        self.model = init_model(model, model_weights_path, self.device)
        self.model.eval()

        self.max_window = int(max_window)
        self.per_frame_k = int(per_frame_k)
        self.final_k = int(final_k)
        self.descriptor_agg = descriptor_agg
        self.ema_decay = float(ema_decay)
        self.recency_weighting = recency_weighting

        self._records: Deque[PlaceRecognitionResult] = deque()
        self._running_sum_descriptor: np.ndarray | None = None
        self._ema_descriptor: np.ndarray | None = None

    def reset(self) -> None:
        """Clear the window and aggregation state."""
        self._records.clear()
        self._running_sum_descriptor = None
        self._ema_descriptor = None

    def start_new_sequence(self, session_id: str | None = None) -> None:
        """Alias for reset; kept for future session-aware extensions."""
        self.reset()

    @torch.inference_mode()
    def infer(
        self,
        input_frame: dict[str, Tensor],
        k: int | None = None,
        return_debug: bool = False,
    ) -> PlaceRecognitionResult | tuple[PlaceRecognitionResult, SequencePRDebug]:
        """Process one frame, update the window, and return fused top-k.

        Args:
            input_frame: Single-frame input dict expected by the model.
            k: Optional override for final_k.
            return_debug: If True, also return a SequencePRDebug instance.

        Returns:
            PlaceRecognitionResult (and optionally SequencePRDebug): fused candidates
            and aggregated descriptor for the current window.

        Raises:
            KeyError: If model output does not contain `final_descriptor`.
            ValueError: If the produced descriptor has an unexpected shape.
        """
        final_k = int(k) if k is not None else self.final_k

        # 1) Forward pass: single-frame descriptor
        out = self.model(input_frame)
        if "final_descriptor" not in out:
            raise KeyError("Model output must contain 'final_descriptor'")
        desc_t: Tensor = out["final_descriptor"]
        if desc_t.ndim == 2 and desc_t.shape[0] == 1:
            desc = desc_t[0].detach().cpu().numpy().astype(np.float32, copy=False)
        elif desc_t.ndim == 1:
            desc = desc_t.detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            raise ValueError("Expected descriptor of shape [D] or [1,D]")

        # 2) Per-frame search
        inds, dists = self.index.search(desc.reshape(1, -1), self.per_frame_k)
        inds = inds[0]
        dists = dists[0].astype(np.float32, copy=False)

        # 3) Push to window (cache without metadata to avoid extra lookups)
        pr_res = PlaceRecognitionResult(
            descriptor=desc,
            indices=inds,
            distances=dists,
            db_idx=None,
            db_pose=None,
        )
        self._push_record(pr_res)

        # 4) Gather per-frame arrays for fusion
        per_i = (
            np.stack([r.indices for r in self._records], axis=0)
            if self._records
            else np.empty((0, 0), dtype=np.int64)
        )
        per_d = (
            np.stack([r.distances for r in self._records], axis=0)
            if self._records
            else np.empty((0, 0), dtype=np.float32)
        )

        # Optional recency weighting (off by default)
        if self.recency_weighting != "none" and per_d.size > 0:
            N = per_d.shape[0]
            ages = np.arange(
                N - 1, -1, -1, dtype=np.float32
            )  # oldest .. newest? We want oldest larger weight
            if self.recency_weighting == "linear":
                weights = 1.0 + ages / max(1, N - 1)
            else:  # exp
                # exponential growth with window length; tune base as needed
                base = 1.25
                weights = base ** (ages / max(1, N - 1))
            per_d = per_d * weights[:, None]

        # 5) Fuse
        fused_d, fused_i = _candidate_pool_fusion(per_d, per_i, final_k)

        # 6) Aggregate descriptor across window
        agg_desc = self._aggregate_descriptor()

        # 7) Map metadata for fused indices only
        db_idx, db_pose, _db_pc = (
            self.index.get_meta(fused_i)
            if fused_i.size > 0
            else (np.empty((0,), dtype=np.int64), np.empty((0, 7), dtype=np.float64), None)
        )

        fused_res = PlaceRecognitionResult(
            descriptor=agg_desc,
            indices=fused_i,
            distances=fused_d,
            db_idx=db_idx,
            db_pose=db_pose,
        )

        if not return_debug:
            return fused_res

        debug = SequencePRDebug(
            per_frame_indices=per_i,
            per_frame_distances=per_d,
            fused_indices=fused_i,
            fused_distances=fused_d,
            window_size=len(self._records),
            descriptor_agg=self.descriptor_agg,
        )
        return fused_res, debug

    # --- internal helpers ---

    def _push_record(self, rec: PlaceRecognitionResult) -> None:
        # update aggregates for the descriptor
        if self.descriptor_agg == "mean":
            if self._running_sum_descriptor is None:
                self._running_sum_descriptor = rec.descriptor.astype(np.float32, copy=False).copy()
            else:
                self._running_sum_descriptor += rec.descriptor.astype(np.float32, copy=False)
        elif self.descriptor_agg == "ema":
            if self._ema_descriptor is None:
                self._ema_descriptor = rec.descriptor.astype(np.float32, copy=False).copy()
            else:
                self._ema_descriptor = self.ema_decay * self._ema_descriptor + (
                    1.0 - self.ema_decay
                ) * rec.descriptor.astype(np.float32, copy=False)

        self._records.append(rec)
        if len(self._records) > self.max_window:
            popped = self._records.popleft()
            if self.descriptor_agg == "mean":
                # subtract from running sum
                self._running_sum_descriptor -= popped.descriptor.astype(np.float32, copy=False)
            # EMA does not support exact removal; keep as-is (acceptable approximation)

    def _aggregate_descriptor(self) -> np.ndarray:
        if not self._records:
            return np.empty((0,), dtype=np.float32)
        if self.descriptor_agg == "last":
            return self._records[-1].descriptor.astype(np.float32, copy=False)
        if self.descriptor_agg == "ema":
            # If only one frame, ema == that descriptor
            if self._ema_descriptor is not None:
                return self._ema_descriptor.astype(np.float32, copy=False)
            return self._records[-1].descriptor.astype(np.float32, copy=False)
        # mean
        count = float(len(self._records))
        if self._running_sum_descriptor is None:
            return self._records[-1].descriptor.astype(np.float32, copy=False)
        return (self._running_sum_descriptor / count).astype(np.float32, copy=False)
