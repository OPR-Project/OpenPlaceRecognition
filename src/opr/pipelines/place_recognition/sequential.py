from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor

from opr.utils import parse_device, init_model


def candidate_pool_fusion(
    distances: np.ndarray, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Candidate Pool Fusion to merge sequence frame votes into final ranking.

    This implements the core "Candidate Pool Fusion" algorithm where each frame in a
    temporal sequence votes independently for its top-K nearest neighbors. All frame
    votes are then pooled, globally sorted by embedding distance, and deduplicated
    to produce the final place recognition ranking.

    The algorithm democratically combines evidence from multiple viewpoints (frames)
    while naturally promoting candidates that are consistently retrieved across frames.

    Args:
        distances (np.ndarray): Embedding distances of shape (query_len, seq_len, at_n).
                               Each query has seq_len frames, each with at_n candidates.
        indices (np.ndarray): Database indices of shape (query_len, seq_len, at_n).
                             Corresponding database indices for each distance.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - fused_distances: shape (query_len, at_n) - final distances after fusion
            - fused_indices: shape (query_len, at_n) - final indices after fusion

    Example:
        Input: 2 frames, top-3 candidates each
        Frame 0: [(db_45, 0.1), (db_23, 0.3), (db_67, 0.5)]
        Frame 1: [(db_23, 0.2), (db_45, 0.4), (db_12, 0.6)]

        Output after fusion: [(db_45, 0.1), (db_23, 0.2), (db_67, 0.5)]
        (db_45 wins with 0.1, db_23 wins with 0.2, db_67 unique from frame 0)
    """
    query_len, _, at_n = distances.shape

    # Initialize output arrays - use np.inf for unfilled distance slots
    merged_distances = np.full((query_len, at_n), np.inf, dtype=distances.dtype)
    merged_indices = np.full((query_len, at_n), -1, dtype=indices.dtype)

    # Process each query independently to avoid memory explosion
    for q in range(query_len):
        # Step 1: Flatten all candidates from all sequence frames for this query
        # This creates the "candidate pool" where each frame votes
        all_distances = distances[q].flatten()  # shape: (seq_len * at_n,)
        all_indices = indices[q].flatten()      # shape: (seq_len * at_n,)

        # Step 2: Sort all candidates globally by embedding distance (ascending)
        # This ensures the best candidates (smallest distances) come first
        sort_order = np.argsort(all_distances)
        sorted_distances = all_distances[sort_order]
        sorted_indices = all_indices[sort_order]

        # Step 3: Remove duplicates, keeping first occurrence (best distance)
        # np.unique with return_index gives us the first occurrence of each unique index
        unique_indices, first_positions = np.unique(
            sorted_indices, return_index=True
        )

        # Extract the corresponding distances for the unique indices
        unique_distances = sorted_distances[first_positions]

        # Step 4: Re-sort by distance since np.unique doesn't preserve order
        # This ensures our final ranking is still distance-ordered
        final_sort_order = np.argsort(unique_distances)
        final_indices = unique_indices[final_sort_order]
        final_distances = unique_distances[final_sort_order]

        # Step 5: Truncate to at_n and store results
        final_count = min(len(final_indices), at_n)
        merged_indices[q, :final_count] = final_indices[:final_count]
        merged_distances[q, :final_count] = final_distances[:final_count]

        # Note: Remaining slots in merged_distances stay as np.inf and
        # merged_indices stay as -1, which should be handled appropriately

    return merged_distances, merged_indices


class SequencePlaceRecognitionPipeline:
    """Basic Place Recognition pipeline."""

    def __init__(
        self,
        database_dir: Path,
        model: nn.Module,
        model_weights_path: Path | None = None,
        device: str | int | torch.device = "cpu",
        use_candidate_pool_fusion: bool = True,
        at_n: int = 5,
    ) -> None:
        """Basic Place Recognition pipeline.

        Args:
            database_dir (Union[str, PathLike]): Path to the database directory. The directory must contain
                "track.csv" and "index.faiss" files.
            model (SequenceLateFusionModel): Model. The forward method must take a dictionary and return a dictionary
                in the predefined format. See the "infer" method for details.
            model_weights_path (Union[str, PathLike], optional): Path to the model weights.
                If None, the weights are not loaded. Defaults to None.
            device (Union[str, int, torch.device]): Device to use. Defaults to "cpu".
            pointcloud_quantization_size (float): Pointcloud quantization size. Defaults to 0.5.
            use_candidate_pool_fusion (bool): Whether to use candidate pool fusion for sequences.
                If False, uses standard single-frame processing. Defaults to False.
            at_n (int): Number of top candidates to retrieve per frame for candidate pool fusion.
                Only used when use_candidate_pool_fusion=True. Defaults to 5.
        """
        self.device = parse_device(device)
        self.model = init_model(model, model_weights_path, self.device)
        self.database_dir = Path(database_dir)
        self._init_database(self.database_dir)
        self.use_candidate_pool_fusion = use_candidate_pool_fusion
        self.at_n = at_n

    def _init_database(self, database_dir: Path) -> None:
        """Initialize database."""
        self.database_df = pd.read_csv(database_dir / "track.csv", index_col=0)
        database_index_filepath = database_dir / "index.faiss"
        if not database_index_filepath.exists():
            raise FileNotFoundError(f"Database index not found: {database_index_filepath}. Create it first.")
        self.database_index = faiss.read_index(str(database_index_filepath))

    def _preprocess_input(self, input_data_sequence: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        """Preprocess input data sequence.

        Args:
            input_data_sequence: List of dictionaries, each representing one frame

        Returns:
            Dictionary with tensors of shape [1, seq_len, ...] where 1 is batch dimension
        """
        if not input_data_sequence:
            raise ValueError("Input data sequence cannot be empty")

        seq_len = len(input_data_sequence)
        out_dict: dict[str, Tensor] = {}

        # Collect data from all frames
        frame_data: dict[str, list[Tensor]] = {}

        for _, input_data in enumerate(input_data_sequence):
            for key in input_data:
                if key.startswith("image_"):
                    new_key = f"images_{key[6:]}"
                    if new_key not in frame_data:
                        frame_data[new_key] = []
                    frame_data[new_key].append(input_data[key].to(self.device))
        # Stack all collected tensors along sequence dimension and add batch dimension
        for key, tensor_list in frame_data.items():
            stacked = torch.stack(tensor_list, dim=0)  # [seq_len, ...]
            out_dict[key] = stacked.unsqueeze(0)  # [1, seq_len, ...]

        return out_dict

    def infer(self, input_data_sequence: list[dict[str, Tensor]]) -> dict[str, np.ndarray]:
        """Single sample inference with optional candidate pool fusion.

        Args:
            input_data_sequence (List[Dict[str, Tensor]]): Sequence of input frames. Each dictionary
                contains keys in the following format:

                "image_{camera_name}" for images from cameras,

                "mask_{camera_name}" for semantic segmentation masks,

                "pointcloud_lidar_coords" for pointcloud coordinates from lidar,

                "pointcloud_lidar_feats" for pointcloud features from lidar,

                "soc" for state-of-charge data.

        Returns:
            Dict[str, np.ndarray]: Inference results. Dictionary with keys:

                "idx" for predicted index in the database,

                "pose" for predicted pose in the format [tx, ty, tz],

                "descriptor" for predicted descriptor.
        """
        input_data = self._preprocess_input(input_data_sequence)
        output = {}

        with torch.no_grad():
            model_output = self.model(input_data)

            if not self.use_candidate_pool_fusion:
                # Standard single descriptor approach
                descriptor = model_output["final_descriptor"].cpu().numpy().reshape(1, -1)
                _, pred_i = self.database_index.search(descriptor, 1)
                pred_i = pred_i[0][0]
            else:
                # Candidate pool fusion approach
                sequential_descriptors = model_output["final_descriptor"]  # [1, seq_len, descriptor_dim]
                seq_descriptors_np = sequential_descriptors.cpu().numpy()  # [1, seq_len, descriptor_dim]

                # Search database for each frame's descriptor
                # Reshape to [seq_len, descriptor_dim] for FAISS search
                seq_descriptors_flat = seq_descriptors_np.reshape(-1, seq_descriptors_np.shape[-1])

                # Get top-K candidates for each frame
                frame_distances, frame_indices = self.database_index.search(seq_descriptors_flat, self.at_n)

                # Reshape results back to [1, seq_len, at_n] for candidate_pool_fusion
                frame_distances = frame_distances.reshape(1, -1, self.at_n)
                frame_indices = frame_indices.reshape(1, -1, self.at_n)

                # Apply candidate pool fusion
                fused_distances, fused_indices = candidate_pool_fusion(frame_distances, frame_indices)

                # Get the best candidate after fusion
                pred_i = fused_indices[0, 0]  # First (best) candidate from first (only) query

        # Get pose for the predicted index
        pred_pose = self.database_df.iloc[pred_i][['tx', 'ty']].to_numpy(
            dtype=float
        )

        # Get final descriptor for output (always use the fused/final descriptor)
        final_descriptor = model_output["final_descriptor"].cpu().numpy()[0]  # Remove batch dimension

        output["idx"] = pred_i
        output["pose"] = pred_pose
        output["descriptor"] = final_descriptor
        return output
