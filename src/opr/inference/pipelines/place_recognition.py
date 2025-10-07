"""Top-k Place Recognition pipeline powered by the new Index.

This pipeline performs:
- minimal input preprocessing (expects preprocessed tensors/arrays)
- model forward to compute a `final_descriptor`
- FAISS Flat index search for top-k raw distances
- mapping results to dataset indices and poses

Outputs are simple dictionaries, intended to be replaced by dataclasses later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from opr.inference.index import Index
from opr.utils import init_model, parse_device


@dataclass
class PlaceRecognitionResult:
    """Result of a top-k place recognition query.

    Notes:
        - ``indices`` are internal row ids in the index (shape [k]).
        - ``db_idx`` and ``db_pose`` are optional to allow caching per-frame
          results without performing metadata lookups for each frame. Sequence
          pipelines may fill only for the fused final result.
    """

    descriptor: np.ndarray  # [D]
    indices: np.ndarray  # [k] internal row positions
    distances: np.ndarray  # [k] raw distances
    db_idx: np.ndarray | None = None  # [k] dataset ids
    db_pose: np.ndarray | None = None  # [k,7] poses (tx,ty,tz,qx,qy,qz,qw)


class PlaceRecognitionPipeline:
    """Minimal top-k Place Recognition pipeline using an `Index`.

    The pipeline assumes that the model returns a dict with key `final_descriptor`.
    It returns raw FAISS distances with corresponding dataset indices and poses.
    """

    def __init__(
        self,
        index: Index,
        model: nn.Module,
        model_weights_path: str | Path | None = None,
        device: str | int | torch.device = "cpu",
    ) -> None:
        """Initialize the pipeline.

        Args:
            index: Loaded `Index` instance.
            model: PyTorch model that outputs `{"final_descriptor": Tensor[B,D]}`.
            model_weights_path: Optional path to weights to load.
            device: Torch device spec.
        """
        self.index = index
        self.device = parse_device(device)
        self.model = init_model(model, model_weights_path, self.device)
        self.model.eval()

    @torch.inference_mode()
    def infer(self, input_data: dict[str, Tensor], k: int = 5) -> PlaceRecognitionResult:
        """Run a single-sample inference and top-k search.

        Args:
            input_data: Dict of tensors expected by the model forward.
            k: Number of neighbors to retrieve.

        Returns:
            PlaceRecognitionResult: descriptor, raw distances and mapped metadata.

        Raises:
            KeyError: If model output does not contain the `final_descriptor` key.
            ValueError: If the produced descriptor has an unexpected shape.
        """
        # Forward pass
        out = self.model(input_data)
        if "final_descriptor" not in out:
            raise KeyError("Model output must contain 'final_descriptor'")
        desc_t: Tensor = out["final_descriptor"]
        if desc_t.ndim == 2 and desc_t.shape[0] == 1:
            desc = desc_t[0].detach().cpu().numpy().astype(np.float32, copy=False)
        elif desc_t.ndim == 1:
            desc = desc_t.detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            raise ValueError("Expected descriptor of shape [D] or [1,D]")

        # Search
        inds, dists = self.index.search(desc.reshape(1, -1), k)
        inds = inds[0]
        dists = dists[0]
        db_idx, db_pose, _db_pc = self.index.get_meta(inds)

        return PlaceRecognitionResult(
            descriptor=desc,
            indices=inds,
            distances=dists,
            db_idx=db_idx,
            db_pose=db_pose,
        )
