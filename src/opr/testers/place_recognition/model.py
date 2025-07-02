"""Place Recognition Model Tester."""

import itertools
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

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


@dataclass
class RetrievalResults:
    """Detailed results of place recognition retrieval for a single query-database track pair.

    This dataclass stores comprehensive information about the retrieval process between
    one query track and one database track. It captures both per-query sample details
    and aggregated metrics for the entire track pair.

    Each query track contains multiple individual samples, and this class stores
    the retrieval results for each of those samples against the database track.
    The arrays are structured such that the first dimension represents individual
    query samples, while the second dimension (where applicable) represents the
    top-k retrievals for each query.

    Example:
        For a query track with 50 samples, retrieving top-10 matches from a database track
        with 200 samples:
        - retrieved_indices would have shape (50, 10)
        - each row contains indices of the 10 closest database samples for one query
    """

    # Original indices
    query_indices: np.ndarray  # Original indices of query samples
    database_indices: np.ndarray  # Original indices of database samples

    # Per-query retrieval details
    retrieved_indices: np.ndarray  # Shape: (num_queries, k) - Indices of retrieved neighbors
    embedding_distances: np.ndarray  # Shape: (num_queries, k) - L2 distances in embedding space
    geographic_distances: np.ndarray  # Shape: (num_queries, k) - Geographic distances
    is_match: np.ndarray  # Shape: (num_queries, k) - Boolean mask of correct matches

    # Aggregated metrics
    recall_at_n: np.ndarray  # Recall@N values
    recall_at_one_percent: float  # Recall@1% value
    top1_distance: float | None  # Mean top-1 distance (can be None)

    # Additional metadata
    num_queries: int  # Total number of individual samples in the query track
    num_database: int  # Total number of individual samples in the database track
    distance_threshold: float  # Geographic distance threshold used
    queries_with_matches: int  # Number of query samples that have at least one match

    # Track information
    query_track_id: int = None  # Track ID for the query
    database_track_id: int = None  # Track ID for the database


@dataclass
class RetrievalResultsCollection:
    """Collection of retrieval results from multiple query-database track pairs.

    This class stores and manages multiple RetrievalResults objects, providing
    methods for aggregation, filtering, and analysis. It can be used to accumulate
    results across all evaluated track pairs for more comprehensive analysis.

    The collection maintains all detailed retrieval information, allowing for
    post-processing, visualization, and deeper analysis than the standard
    aggregate metrics alone.
    """

    results: list[RetrievalResults] = field(default_factory=list)

    def __len__(self) -> int:
        """Get the number of results in the collection.

        Returns:
            int: The number of results in the collection.
        """
        return len(self.results)

    def append(self, result: RetrievalResults) -> None:
        """Add a RetrievalResults object to the collection.

        Args:
            result (RetrievalResults): The retrieval results to add.
        """
        self.results.append(result)

    def extend(self, results: Iterable[RetrievalResults]) -> None:
        """Add multiple RetrievalResults objects to the collection.

        Args:
            results (Iterable[RetrievalResults]): The retrieval results to add.
        """
        self.results.extend(results)

    @property
    def num_pairs(self) -> int:
        """Get the number of query-database track pairs in the collection.

        Returns:
            int: The number of track pairs.
        """
        return len(self.results)

    @property
    def num_queries(self) -> int:
        """Get the total number of query samples across all track pairs.

        Returns:
            int: The total number of query samples.
        """
        return sum(res.num_queries for res in self.results)

    @property
    def num_tracks(self) -> tuple[int, int]:
        """Get the number of unique query and database tracks in the collection.

        Returns:
            tuple[int, int]: A tuple containing (unique_query_tracks, unique_database_tracks)
        """
        query_tracks = set(res.query_track_id for res in self.results if res.query_track_id is not None)
        db_tracks = set(res.database_track_id for res in self.results if res.database_track_id is not None)
        return len(query_tracks), len(db_tracks)

    def aggregate_metrics(self) -> dict[str, Any]:
        """Calculate aggregate metrics across all results in the collection.

        Returns:
            dict[str, Any]: Dictionary containing the following aggregate metrics:
                - 'recall_at_n': Mean Recall@N values
                - 'recall_at_one_percent': Mean Recall@1% value
                - 'top1_distance': Mean top-1 distance value
                - 'overall_accuracy': Percentage of queries with correct top-1 match
                - 'queries_with_matches': Total number of queries with matches
                - 'total_queries': Total number of queries
        """
        # Calculate recall metrics
        recalls_at_n = []
        recalls_at_one_percent = []
        top1_distances = []
        correct_top1_matches = 0
        total_with_matches = 0

        for res in self.results:
            if res.queries_with_matches > 0:
                recalls_at_n.append(res.recall_at_n)
                recalls_at_one_percent.append(res.recall_at_one_percent)
                if res.top1_distance is not None:
                    top1_distances.append(res.top1_distance)

                # Count correct top-1 matches
                correct_top1_matches += np.sum(res.is_match[:, 0])
                total_with_matches += res.queries_with_matches

        # Compute means
        mean_recall_at_n = np.mean(recalls_at_n, axis=0) if recalls_at_n else np.array([])
        mean_recall_at_one_percent = np.mean(recalls_at_one_percent) if recalls_at_one_percent else 0.0
        mean_top1_distance = np.mean(top1_distances) if top1_distances else None
        overall_accuracy = correct_top1_matches / total_with_matches if total_with_matches > 0 else 0.0

        return {
            "recall_at_n": mean_recall_at_n,
            "recall_at_one_percent": mean_recall_at_one_percent,
            "top1_distance": mean_top1_distance,
            "overall_accuracy": overall_accuracy,
            "queries_with_matches": total_with_matches,
            "total_queries": self.num_queries,
        }

    def filter_by_track(
        self, query_track_id: int | None = None, database_track_id: int | None = None
    ) -> "RetrievalResultsCollection":
        """Filter results by query and/or database track IDs.

        Args:
            query_track_id (Optional[int]): If provided, only keep results with this query track ID.
            database_track_id (Optional[int]): If provided, only keep results with this database track ID.

        Returns:
            RetrievalResultsCollection: A new collection containing only the filtered results.
        """
        filtered_results = []

        for res in self.results:
            if (query_track_id is None or res.query_track_id == query_track_id) and (
                database_track_id is None or res.database_track_id == database_track_id
            ):
                filtered_results.append(res)

        collection = RetrievalResultsCollection()
        collection.extend(filtered_results)
        return collection

    def get_difficult_queries(self, top_k: int = 10) -> list[tuple[int, int]]:
        """Identify query samples that consistently fail to retrieve correct matches."""
        raise NotImplementedError(
            "This method is not implemented. Please implement it in a subclass or provide a custom implementation."
        )

    def save(self, path: str) -> None:
        """Save the collection to disk in JSON format for later analysis.

        Args:
            path (str): Path to save the collection.
        """

        def convert(obj: Any) -> Any:
            """Convert numpy types to native Python types for JSON serialization."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            return obj

        data = [{k: convert(v) for k, v in asdict(result).items()} for result in self.results]

        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls: type["RetrievalResultsCollection"], path: str) -> "RetrievalResultsCollection":
        """Load a previously saved collection in JSON format.

        Args:
            path (str): Path to the saved collection.

        Returns:
            RetrievalResultsCollection: The loaded collection.
        """
        with open(path, "r") as f:
            data = json.load(f)

        collection = RetrievalResultsCollection()

        # Convert loaded data back to proper types and create RetrievalResults objects
        for item in data:
            # Convert lists back to numpy arrays
            for key in [
                "query_indices",
                "database_indices",
                "retrieved_indices",
                "embedding_distances",
                "geographic_distances",
                "is_match",
                "recall_at_n",
            ]:
                if key in item:
                    item[key] = np.array(item[key])

            # Create RetrievalResults object using unpacked dictionary
            collection.results.append(RetrievalResults(**item))

        return collection


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

    def run(self) -> RetrievalResultsCollection:
        """Run the full model test.

        This method performs the following steps:
            1. Extract embeddings and related metadata.
            2. Group samples by track to form query and database groups.
            3. Compute the geographic distance matrix.
            4. Evaluate track pairs.
            5. Return a collection of detailed retrieval results.

        Returns:
            RetrievalResultsCollection: Collection of detailed retrieval results for all track pairs.
                Contains all metrics, per-query details and can be used for further analysis.
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

        # Get the results collection
        results_collection = self._eval_pairs(embs, queries, databases, geo_dist)

        if self.verbose:
            print(f"Collected {len(results_collection)} track pair results")
            print("Evaluation complete.")

        return results_collection

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
    ) -> RetrievalResultsCollection:
        """Evaluate track pairs and compute recall metrics.

        For each ordered pair of tracks (i != j):
          - Extract query/database embeddings and geo-distances
          - Call eval_retrieval_pair(...) to compute metrics and retrieval details
          - Store results in a collection

        Args:
            embs (np.ndarray): Embeddings array for all samples.
            queries (list[list[int]]): List of query indices grouped by track.
            databases (list[list[int]]): List of database indices grouped by track.
            geo_dist (np.ndarray): Geographic distance matrix.

        Returns:
            RetrievalResultsCollection: Collection of detailed retrieval results for all track pairs.
        """
        # Create a collection to store detailed results
        results_collection = RetrievalResultsCollection()

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

            # Call the renamed method which returns a RetrievalResults object
            results = self.eval_retrieval_pair(
                query_embs,
                database_embs,
                distances,
                at_n=self.at_n,
                dist_thresh=self.dist_thresh,
                query_indices=np.array(query),
                database_indices=np.array(database),
                query_track_id=i,
                database_track_id=j,
            )

            # Store the results in our collection
            results_collection.append(results)

        return results_collection

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
    def eval_retrieval_pair(
        query_embs: np.ndarray,
        db_embs: np.ndarray,
        geo_distances: np.ndarray,
        dist_thresh: float = 25.0,
        at_n: int = 25,
        query_indices: np.ndarray = None,
        database_indices: np.ndarray = None,
        query_track_id: int = None,
        database_track_id: int = None,
    ) -> RetrievalResults:
        """Evaluate retrieval performance for a query-database track pair.

        Performs nearest neighbor search in embedding space and evaluates results
        against geographic ground truth. Returns comprehensive retrieval details
        and performance metrics.

        Args:
            query_embs (np.ndarray): Query embeddings array.
            db_embs (np.ndarray): Database embeddings array.
            geo_distances (np.ndarray): Geographic distance matrix of shape (query_len, db_len).
            dist_thresh (float): Geographic distance threshold for positive match. Defaults to 25.0.
            at_n (int): The maximum N value for the Recall@N metric. Defaults to 25.
            query_indices (np.ndarray): Original indices of query samples. Defaults to None.
            database_indices (np.ndarray): Original indices of database samples. Defaults to None.
            query_track_id (int): ID of the query track. Defaults to None.
            database_track_id (int): ID of the database track. Defaults to None.

        Returns:
            RetrievalResults: Detailed retrieval results including metrics and per-query details.
        """
        db_len = db_embs.shape[0]
        query_len = query_embs.shape[0]
        orig_at_n = at_n
        # Use original calculation method for backward compatibility
        # TODO: think which way is more correct - rounding or truncating
        one_percent_threshold = max(int(round(db_len / 100.0)), 1)
        at_n = max(at_n, one_percent_threshold)

        # Use default indices if not provided
        if query_indices is None:
            query_indices = np.arange(query_len)
        if database_indices is None:
            database_indices = np.arange(db_len)

        # Geographic distance binary mask - indicates true positives
        positives_mask = geo_distances <= dist_thresh
        queries_with_matches = np.sum(np.any(positives_mask, axis=1))

        # Get nearest neighbors using embedding distances (not geographic)
        if faiss_available:
            # Use FAISS for faster nearest neighbor search
            index = faiss.IndexFlatL2(db_embs.shape[1])
            index.add(db_embs)
            emb_distances, retrieved_indices = index.search(query_embs, at_n)
        else:
            # Fall back to KDTree
            tree = KDTree(db_embs)
            emb_distances, retrieved_indices = tree.query(query_embs, k=at_n)

        # Initialize arrays for retrieval data
        is_match = np.zeros((query_len, at_n), dtype=bool)
        retrieved_geo_distances = np.zeros((query_len, at_n), dtype=np.float32)

        # Initialize recall array and top1 distances list
        recall_at_n = np.zeros(at_n, dtype=float)
        top1_distances = []

        # For each query, check if the retrieved neighbors are geographic matches
        for query_i, closest_inds in enumerate(retrieved_indices):
            # Get geographic match status for retrieved neighbors
            query_gt_matches_mask = positives_mask[query_i][closest_inds]
            is_match[query_i] = query_gt_matches_mask

            # Get geographic distances to retrieved neighbors
            retrieved_geo_distances[query_i] = geo_distances[query_i][closest_inds]

            # Store top-1 distance if it's a match (for traditional metric)
            if query_gt_matches_mask[0]:
                top1_distances.append(emb_distances[query_i][0])

            # Update recall counts (for traditional metric)
            recall_at_n += np.cumsum(query_gt_matches_mask, axis=0, dtype=bool)

        # Normalize and finalize metrics
        if queries_with_matches > 0:
            recall_at_n = recall_at_n / queries_with_matches
            one_percent_recall = recall_at_n[one_percent_threshold - 1]
            mean_top1_distance = np.mean(top1_distances) if len(top1_distances) > 0 else None
        else:
            recall_at_n = np.zeros(at_n)
            one_percent_recall = 0.0
            mean_top1_distance = None

        # Create and return the comprehensive results object
        return RetrievalResults(
            query_indices=query_indices,
            database_indices=database_indices,
            retrieved_indices=retrieved_indices,
            embedding_distances=emb_distances,
            geographic_distances=retrieved_geo_distances,
            is_match=is_match,
            recall_at_n=recall_at_n[:orig_at_n],
            recall_at_one_percent=one_percent_recall,
            top1_distance=mean_top1_distance,
            num_queries=query_len,
            num_database=db_len,
            distance_threshold=dist_thresh,
            queries_with_matches=queries_with_matches,
            query_track_id=query_track_id,
            database_track_id=database_track_id,
        )
