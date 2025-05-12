"""Place Recognition Model Tester."""

import itertools

import numpy as np
import torch
from sklearn.neighbors import KDTree
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import faiss

    faiss_available = True
except ImportError:
    faiss_available = False

from opr.utils import parse_device


class ModelTester:
    """Test a place recognition model.

    This class tests a place recognition model for metrics like Recall@N,
    Recall@1%, and the mean top-1 distance.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        dist_thresh: float = 25.0,
        at_n: int = 25,
        device: str | int | torch.device = "cuda",
        verbose: bool = True,
        batch_size: int = None,  # New parameter for memory-efficient distance calculation
    ) -> None:
        """Initialize ModelTester.

        Args:
            model (torch.nn.Module): A PyTorch model.
            dataloader (DataLoader): A dataloader providing input batches.
            dist_thresh (float): Positive float distance threshold. Defaults to 25.0.
            at_n (int): Positive integer for top-N metric. Defaults to 25.
            device (str | int | torch.device): Device specification (e.g., "cuda"). Defaults to "cuda".
            verbose (bool): Whether to show progress bars. Defaults to True.
            batch_size (int): If specified, compute the distance matrix in batches
                to reduce peak memory usage. Useful for large datasets. Defaults to None.

        Raises:
            ValueError: If `dist_thresh` is not a positive float.
            ValueError: If `at_n` is not a positive integer.
        """
        self.model = model
        self.dataloader = dataloader
        self.dataset_df = self.dataloader.dataset.dataset_df

        # temporary workaround for working with datasets with frame sequences
        # TODO: get rid of 'hasattr', check and use a more elegant solution
        if (
            hasattr(self.dataloader.dataset, "_sequence_indices")
            and self.dataloader.dataset._sequence_indices is not None
        ):
            sequence_indices = [seq[0] for seq in self.dataloader.dataset._sequence_indices]
            self.dataset_df = self.dataset_df.iloc[sequence_indices]
            self.dataset_df = self.dataset_df.reset_index(drop=True)

        self.verbose = verbose

        self.dist_thresh = dist_thresh
        if not isinstance(self.dist_thresh, float) or self.dist_thresh <= 0:
            raise ValueError("dist_thresh must be a positive float value.")

        self.at_n = at_n
        if not isinstance(self.at_n, int) or self.at_n <= 0:
            raise ValueError("at_n must be a positive integer.")

        self.device = parse_device(device)

        self.model.to(self.device)
        self.model.eval()

        self.batch_size = batch_size

        self._coords_columns_names = self._get_coords_columns_names()

    def run(self) -> tuple[np.ndarray, float, float]:
        """Run the full model test.

        This method performs the following steps:
            1. Extract embeddings and related metadata.
            2. Group samples by track to form query and database groups.
            3. Compute the geographic distance matrix.
            4. Evaluate track pairs.
            5. Aggregate and return the final metrics.

        Returns:
            tuple[np.ndarray, float, float]: A tuple containing:
                - Average Recall@N (np.ndarray): Array of recall values for each N up to at_n.
                - Average Recall@1% (float): Average recall at 1% of the database size.
                - Average top-1 distance (float): Average distance to the closest correct match.
        """
        if self.verbose:
            print("Starting place recognition evaluation...")

        embs = self._extract_embeddings()

        if self.verbose:
            print(f"Extracted embeddings: {embs.shape}")

        queries, databases = self._group_by_track()

        if self.verbose:
            print(f"Grouped tracks: {len(queries)} queries, {len(databases)} databases")

        coords = self.dataset_df[self._coords_columns_names].to_numpy()

        if self.verbose:
            print(f"Computing geographic distance matrix for {len(coords)} coordinates...")
            if self.batch_size:
                print(f"Using batch size {self.batch_size} to reduce memory usage")

        geo_dist = self._compute_geo_dist(coords, self.batch_size)

        if self.verbose:
            unique_track_pairs = len(list(itertools.permutations(range(len(queries)), 2)))
            print(f"Evaluating {unique_track_pairs} track pairs...")

        recalls_at_n, recalls_at_one_percent, top1_distances = self._eval_pairs(
            embs, queries, databases, geo_dist
        )

        if self.verbose:
            print("Aggregating metrics...")

        recalls_at_n, recalls_at_one_percent, top1_distances = self._aggregate(
            recalls_at_n, recalls_at_one_percent, top1_distances
        )

        if self.verbose:
            print("Evaluation complete.")

        return recalls_at_n, recalls_at_one_percent, top1_distances

    def _get_coords_columns_names(self) -> list[str]:
        """Retrieve the coordinate columns from the dataset.

        Returns:
            list[str]: A list of column names representing coordinates.

        Raises:
            ValueError: If required coordinate columns are not found or incomplete.
        """
        if "northing" in self.dataset_df.columns:
            if "easting" not in self.dataset_df.columns:
                raise ValueError("'northing' column found, but no 'easting' column in dataset_df.")
            return ["northing", "easting"]
        elif "x" in self.dataset_df.columns:
            if "y" not in self.dataset_df.columns:
                raise ValueError("'x' column found, but no 'y' column in dataset_df.")
            if "z" in self.dataset_df.columns:
                return ["x", "y", "z"]
            return ["x", "y"]
        elif "tx" in self.dataset_df.columns:
            if "ty" not in self.dataset_df.columns:
                raise ValueError("'tx' column found, but no 'ty' column in dataset_df.")
            if "tz" in self.dataset_df.columns:
                return ["tx", "ty", "tz"]
            return ["tx", "ty"]
        else:
            raise ValueError(
                "No valid coordinate columns found in dataset_df. "
                "Expected 'northing'/'easting', 'x'/'y', 'x'/'y'/'z',"
                "'tx'/'ty' or 'tx'/'ty'/'tz'."
            )

    def _extract_embeddings(self) -> np.ndarray:
        """Extract embeddings and associated metadata.

        Runs the model over the dataloader to compute the final descriptors,
        and collects metadata such as coordinates, track IDs, and query flags.

        Returns:
            np.ndarray: Array of descriptors (N_samples x D).
        """
        embs = []
        # Add tqdm progress bar that shows only if verbose is True
        dataloader_iter = tqdm(
            self.dataloader, desc="Extracting embeddings", leave=False, disable=not self.verbose
        )

        for batch in dataloader_iter:
            batch = {k: v.to(self.device) for k, v in batch.items() if k not in ["idxs", "utms"]}
            with torch.no_grad():
                batch_embs = self.model(batch)["final_descriptor"]
            embs.append(batch_embs.cpu().numpy())
        embs = np.concatenate(embs, axis=0)
        return embs

    def _group_by_track(self) -> tuple[list[list[int]], list[list[int]]]:
        """Group samples by track ID.

        Groups dataset samples by their track IDs and separates query samples if specified.

        Returns:
            tuple[list[list[int]], list[list[int]]]: A tuple containing:
                - List of query indices grouped by track
                - List of database indices grouped by track
        """
        queries = []
        databases = []
        for _, group in self.dataset_df.groupby("track"):
            db_indices = group.index.to_list()
            databases.append(db_indices)
            if "in_query" in group.columns:
                query_indices = group[group["in_query"]].index.to_list()
                queries.append(query_indices)
            else:
                queries.append(db_indices)
        return queries, databases

    def _compute_geo_dist(self, coords: np.ndarray, batch_size: int = None) -> np.ndarray:
        """Compute pairwise L2 distances over coordinates.

        Args:
            coords (np.ndarray): Coordinate array of shape (N_samples, coord_dim).
            batch_size (int): If specified, compute the distance matrix in batches
                to reduce peak memory usage. Defaults to None.

        Returns:
            np.ndarray: Distance matrix of shape (N_samples, N_samples).
        """
        if batch_size is None:
            # Original implementation - all at once (higher memory usage)
            return np.linalg.norm(coords[:, np.newaxis] - coords[np.newaxis, :], axis=2)
        else:
            # Batched implementation to reduce peak memory usage
            n = coords.shape[0]
            dist_matrix = np.zeros((n, n), dtype=np.float32)

            # Process in batches to reduce memory usage
            for i in tqdm(
                range(0, n, batch_size), desc="Computing geographic distances", disable=not self.verbose
            ):
                end_idx = min(i + batch_size, n)
                batch = coords[i:end_idx]
                # Calculate distances for this batch to all points
                dist_matrix[i:end_idx] = np.sqrt(
                    ((batch[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2).sum(axis=2)
                )
            return dist_matrix

    def _eval_pairs(
        self,
        embs: np.ndarray,
        queries: list[list[int]],
        databases: list[list[int]],
        geo_dist: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate track pairs and compute recall metrics.

        For each ordered pair of tracks (i != j):
          - Extract query/database embeddings and geo-distances
          - Call get_recalls(...) to compute metrics
          - Store per-pair Recall@N, Recall@1%, top1 distance values

        Args:
            embs (np.ndarray): Embeddings array for all samples.
            queries (list[list[int]]): List of query indices grouped by track.
            databases (list[list[int]]): List of database indices grouped by track.
            geo_dist (np.ndarray): Geographic distance matrix.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                - recalls_at_n: (T x T x at_n) array of Recall@N values for each track pair
                - recalls_at_one_percent: (T x T) array of Recall@1% values for each track pair
                - top1_distances: (T x T) array of top-1 distances for each track pair
        """
        recalls_at_n = np.zeros((len(queries), len(databases), self.at_n))
        recalls_at_one_percent = np.zeros((len(queries), len(databases)))
        top1_distances = np.zeros((len(queries), len(databases)))

        recalls_at_n.fill(np.nan)
        recalls_at_one_percent.fill(np.nan)
        top1_distances.fill(np.nan)

        ij_permutations = list(itertools.permutations(range(len(queries)), 2))

        # Updated progress bar with more information
        progress_bar = tqdm(
            ij_permutations,
            desc="Evaluating track pairs",
            leave=False,
            disable=not self.verbose,
            total=len(ij_permutations),
        )

        for i, j in progress_bar:
            query = queries[i]
            database = databases[j]

            if len(query) == 0 or len(database) == 0:
                if self.verbose:
                    progress_bar.set_description(f"Track pair ({i},{j}): Empty tracks, skipping")
                continue

            # Update progress description if verbose
            if self.verbose:
                progress_bar.set_description(
                    f"Track pair ({i},{j}): {len(query)} queries, {len(database)} references"
                )

            query_embs = embs[query]
            database_embs = embs[database]

            distances = geo_dist[query][:, database]

            recalls, one_percent_recall, top1_distance = self.get_recalls(
                query_embs, database_embs, distances, at_n=self.at_n, dist_thresh=self.dist_thresh
            )

            recalls_at_n[i, j] = recalls
            recalls_at_one_percent[i, j] = one_percent_recall

            if top1_distance is not None:
                top1_distances[i, j] = top1_distance

        return recalls_at_n, recalls_at_one_percent, top1_distances

    def _aggregate(self, Rn: np.ndarray, R1p: np.ndarray, T1: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Aggregate per-pair metrics into final averages.

        Args:
            Rn (np.ndarray): Recall@N matrix of shape (T, T, at_n).
            R1p (np.ndarray): Recall@1% matrix of shape (T, T).
            T1 (np.ndarray): Top-1 distance matrix of shape (T, T).

        Returns:
            tuple[np.ndarray, float, float]:
                - Mean Recall@N values across all track pairs (array of length at_n)
                - Mean Recall@1% value (scalar)
                - Mean top-1 distance value (scalar)
        """
        mean_Rn = np.nanmean(Rn, axis=(0, 1))
        mean_R1p = np.nanmean(R1p, axis=(0, 1))
        mean_T1 = np.nanmean(T1, axis=(0, 1))

        return mean_Rn, mean_R1p, mean_T1

    @staticmethod
    def get_recalls(
        query_embs: np.ndarray,
        db_embs: np.ndarray,
        dist_matrix: np.ndarray,
        dist_thresh: float = 25.0,
        at_n: int = 25,
    ) -> tuple[np.ndarray, float, float | None]:
        """Calculate Recall@N, Recall@1% and mean top-1 distance for the given query and db embeddings.

        Args:
            query_embs (np.ndarray): Query embeddings array.
            db_embs (np.ndarray): Database embeddings array.
            dist_matrix (np.ndarray): Geographic distance matrix of shape (query_len, db_len).
            dist_thresh (float): Geographic distance threshold for positive match. Defaults to 25.0.
            at_n (int): The maximum N value for the Recall@N metric. Defaults to 25.

        Returns:
            tuple[np.ndarray, float, float | None]:
                - Recall@N: Array of recall values for each N up to at_n
                - Recall@1%: Recall at 1% of the database size
                - Mean top-1 distance: Average distance to closest match, or None if no matches found
        """
        db_len = db_embs.shape[0]
        orig_at_n = at_n
        one_percent_threshold = max(1, int(np.ceil(db_len * 0.01)))
        at_n = max(at_n, one_percent_threshold)

        # Geographic distance binary mask - indicates true positives
        positives_mask = dist_matrix < dist_thresh
        queries_with_matches = np.sum(np.any(positives_mask, axis=1))

        if queries_with_matches == 0:
            # No matches found - early return
            return np.zeros(orig_at_n), 0.0, None

        # Get nearest neighbors using embedding distances (not geographic)
        if faiss_available:
            # Use FAISS for faster nearest neighbor search
            index = faiss.IndexFlatL2(db_embs.shape[1])
            index.add(db_embs)
            distances, indices = index.search(query_embs, at_n)
        else:
            # Fall back to KDTree
            tree = KDTree(db_embs)
            distances, indices = tree.query(query_embs, k=at_n)

        # Initialize recall array
        recall_at_n = np.zeros(at_n, dtype=float)
        top1_distances = []

        # For each query, check if the retrieved neighbors are geographic matches
        for query_i, closest_inds in enumerate(indices):
            # Check which retrieved neighbors are geographic matches
            query_gt_matches_mask = positives_mask[query_i][closest_inds]

            # Store top-1 distance if it's a match
            if query_gt_matches_mask[0]:
                top1_distances.append(distances[query_i][0])

            # Update recall counts
            recall_at_n += np.cumsum(query_gt_matches_mask, axis=0, dtype=bool)

        # Normalize and finalize metrics
        recall_at_n = recall_at_n / queries_with_matches
        one_percent_recall = recall_at_n[one_percent_threshold - 1]
        mean_top1_distance = np.mean(top1_distances) if len(top1_distances) > 0 else None

        return recall_at_n[:orig_at_n], one_percent_recall, mean_top1_distance
