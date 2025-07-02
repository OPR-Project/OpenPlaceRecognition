from torch import nn, Tensor

from opr.modules.temporal import TemporalAveragePooling


class SequenceLateFusionModel(nn.Module):
    """Meta-model for sequence-based multimodal Place Recognition with late fusion."""

    def __init__(
        self,
        model: nn.Module,
        temporal_fusion_module: nn.Module | None = None,
    ) -> None:
        """Meta-model for sequence-based multimodal Place Recognition with late fusion.

        Args:
            model (nn.Module): Base model for processing individual frames.
            temporal_fusion_module (nn.Module, optional): Module to fuse features across time.
                If None, defaults to a module that takes the average across the sequence.
        """
        super().__init__()
        self.model = model
        self.temporal_fusion_module = temporal_fusion_module or TemporalAveragePooling()

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Process a sequence of frames efficiently by reshaping to batch processing.

        Args:
            batch: Dictionary containing sequence data with shape [B, S, ...]
                  where B is batch size and S is sequence length

        Returns:
            Dictionary with the final descriptor after temporal fusion
        """
        batch_size, seq_len = self._get_batch_and_seq_dims(batch)
        flat_batch = self._reshape_batch_for_processing(batch, batch_size, seq_len)
        flat_output = self.model(flat_batch)
        descriptors = flat_output["final_descriptor"].view(batch_size, seq_len, -1)
        final_descriptor, sequential_descriptors = self.temporal_fusion_module(descriptors)
        return {"final_descriptor": final_descriptor, "sequential_descriptors": sequential_descriptors}

    def _get_batch_and_seq_dims(self, batch: dict[str, Tensor]) -> tuple[int, int]:
        """Extract batch size and sequence length from batch data."""
        for key, value in batch.items():
            if key.startswith("images_"):
                return value.shape[0], value.shape[1]  # B, S from [B, S, C, H, W]
            elif key == "pointclouds_lidar_coords":
                return value.shape[0], value.shape[1]  # B, S from [B, S, N, 3]

        raise ValueError("Could not determine batch size and sequence length from batch")

    def _reshape_batch_for_processing(
            self, batch: dict[str, Tensor], batch_size: int, seq_len: int
        ) -> dict[str, Tensor]:
        """Reshape batch from [B, S, ...] to [B*S, ...] for efficient processing."""
        flat_batch = {}

        for key, value in batch.items():
            if key.startswith("images_"):
                # Reshape image data: [B, S, C, H, W] -> [B*S, C, H, W]
                flat_batch[key] = value.reshape(batch_size * seq_len, *value.shape[2:])

        return flat_batch
