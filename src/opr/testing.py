"""Testing functions implementation."""
import itertools
from typing import Optional, Tuple, Union

import numpy as np
import torch
from pytorch_metric_learning.distances import LpDistance
from sklearn.neighbors import KDTree
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from opr.utils import parse_device


def get_recalls(
    query_embs: np.ndarray,
    db_embs: np.ndarray,
    dist_matrix: np.ndarray,
    dist_thresh: float = 25.0,
    at_n: int = 25,
) -> Tuple[np.ndarray, float, Optional[float]]:
    """Calculate Recall@N, Recall@1% and mean top-1 distance for the given query and db embeddings.

    Args:
        query_embs (np.ndarray): Query embeddings array.
        db_embs (np.ndarray): Database embeddings array.
        dist_matrix (np.ndarray): Distance matrix of shape (query_len, db_len).
        dist_thresh (float): Distance threshold for positive match. Defaults to 25.0.
        at_n (int): The maximum N value for the Recall@N metric. Defaults to 25.

    Returns:
        Tuple[np.ndarray, float, Optional[float]]: (Recall@N, Recall@1%, mean top-1 distance).
            The 'mean top-1 distance' metric may be `None` if Recall@1 = 0.
    """
    database_tree = KDTree(db_embs)

    positives_mask = dist_matrix <= dist_thresh
    queries_with_matches = int((positives_mask.sum(axis=1) > 0).sum())
    queries_with_matches_indices = np.argwhere(positives_mask.sum(axis=1) > 0)
    assert queries_with_matches_indices.shape[0] == queries_with_matches

    recall_at_n = np.zeros((at_n,), dtype=float)

    top1_distances = []
    errors_query = []
    errors_db = []
    gt_db = []
    errors_dists = []
    one_percent_threshold = max(int(round(len(db_embs) / 100.0)), 1)

    distances, indices = database_tree.query(query_embs, k=at_n)

    for query_i, closest_inds in enumerate(indices):
        query_gt_matches_mask = positives_mask[query_i][closest_inds]
        if query_gt_matches_mask[0]:
            top1_distances.append(distances[query_i][0])
        elif query_i in queries_with_matches_indices:
            errors_dists.append(distances[query_i][0])
            errors_query.append(query_i)
            errors_db.append(closest_inds[0])
            gt_db.append(dist_matrix[query_i].argmin())
        recall_at_n += np.cumsum(query_gt_matches_mask, axis=0, dtype=bool)

    recall_at_n = recall_at_n / queries_with_matches
    one_percent_recall = recall_at_n[one_percent_threshold - 1]
    if len(top1_distances) > 0:
        mean_top1_distance = np.mean(top1_distances)
    else:
        mean_top1_distance = None

    return recall_at_n, one_percent_recall, mean_top1_distance


def test(
    model: nn.Module,
    dataloader: DataLoader,
    distance_threshold: float = 25.0,
    device: Union[str, int, torch.device] = "cuda",
) -> Tuple[np.ndarray, float, float]:
    """Evaluates the model on the test set.

    Args:
        model (nn.Module): The model to test.
        dataloader (DataLoader): The data loader for the test set.
        distance_threshold (float): The distance threshold for a correct match. Defaults to 25.0.
        device (Union[str, int, torch.device]): Device ("cpu" or "cuda"). Defaults to "cuda".

    Returns:
        Tuple[np.ndarray, float, float]: Array of AverageRecall@N (N from 1 to 25), AverageRecall@1%
            and mean top-1 distance.

    Raises:
        ValueError: If the required coordinate columns are not found in the dataset.
    """
    device = parse_device(device)

    model = model.to(device)

    with torch.no_grad():
        embeddings_list = []
        for batch in tqdm(dataloader, desc="Calculating test set descriptors", leave=False):
            batch = {e: batch[e].to(device) for e in batch}
            embeddings = model(batch)["final_descriptor"]
            embeddings_list.append(embeddings.cpu().numpy())
            torch.cuda.empty_cache()
        test_embeddings = np.vstack(embeddings_list)

    test_df = dataloader.dataset.dataset_df

    # temporary workaround for working with datasets with frame sequences
    if hasattr(dataloader.dataset, "_sequence_indices") and dataloader.dataset._sequence_indices is not None:
        sequence_indices = [seq[0] for seq in dataloader.dataset._sequence_indices]
        test_df = test_df.iloc[sequence_indices]
        test_df = test_df.reset_index(drop=True)

    queries = []
    databases = []

    for _, group in test_df.groupby("track"):
        databases.append(group.index.to_list())

        if "in_query" in group.columns:
            selected_queries = group[group["in_query"]]
            queries.append(selected_queries.index.to_list())
        else:
            queries.append(group.index.to_list())

    if "northing" in test_df.columns and "easting" in test_df.columns:
        coords_columns = ["northing", "easting"]
    elif "x" in test_df.columns and "y" in test_df.columns:
        coords_columns = ["x", "y"]
    elif "tx" in test_df.columns and "ty" in test_df.columns:
        coords_columns = ["tx", "ty"]
    else:
        raise ValueError(
            "Required coordinate columns ('northing'/'easting', 'x'/'y', or 'tx'/'ty') not found in the dataset"
        )

    utms = torch.tensor(test_df[coords_columns].to_numpy())
    dist_fn = LpDistance(normalize_embeddings=False)
    dist_utms = dist_fn(utms).numpy()

    n = 50
    recalls_at_n = np.zeros((len(queries), len(databases), n))
    recalls_at_one_percent = np.zeros((len(queries), len(databases), 1))
    top1_distances = np.zeros((len(queries), len(databases), 1))
    ij_permutations = list(itertools.permutations(range(len(queries)), 2))
    count_r_at_1 = 0

    for i, j in tqdm(ij_permutations, desc="Calculating metrics", leave=False):
        query = queries[i]
        database = databases[j]
        query_embs = test_embeddings[query]
        database_embs = test_embeddings[database]

        distances = dist_utms[query][:, database]
        (
            recalls_at_n[i, j],
            recalls_at_one_percent[i, j],
            top1_distance,
        ) = get_recalls(query_embs, database_embs, distances, at_n=n, dist_thresh=distance_threshold)

        if top1_distance:
            count_r_at_1 += 1
            top1_distances[i, j] = top1_distance
    mean_recall_at_n = recalls_at_n.sum(axis=(0, 1)) / len(ij_permutations)
    mean_recall_at_one_percent = recalls_at_one_percent.sum(axis=(0, 1)).squeeze() / len(ij_permutations)
    mean_top1_distance = top1_distances.sum(axis=(0, 1)).squeeze() / len(ij_permutations)

    return mean_recall_at_n, mean_recall_at_one_percent, mean_top1_distance
