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

    queries = []
    databases = []

    for _, group in test_df.groupby("track"):
        databases.append(group.index.to_list())
        selected_queries = group[group["in_query"]]
        queries.append(selected_queries.index.to_list())

    utms = torch.tensor(test_df[["northing", "easting"]].to_numpy())
    dist_fn = LpDistance(normalize_embeddings=False)
    dist_utms = dist_fn(utms).numpy()

    n = 25
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


def get_recalls_with_reranking(
    front_query_embs: np.ndarray,
    back_query_embs: np.ndarray,
    db_embs: np.ndarray,
    dist_matrix: np.ndarray,
    dist_thresh: float = 25.0,
    at_n: int = 25,
) -> Tuple[np.ndarray, float, Optional[float]]:
    """Calculate Recall@N, Recall@1% and mean top-1 distance for the given query and db embeddings.

    Args:
        front_query_embs (np.ndarray): Query front image embeddings array.
        back_query_embs (np.ndarray): Query back image embeddings array.
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

    front_distances, front_indices = database_tree.query(front_query_embs, k=at_n)
    back_distances, back_indices = database_tree.query(back_query_embs, k=at_n)

    distances = np.hstack([front_distances, back_distances])
    indices = np.hstack([front_indices, back_indices]) % (len(db_embs) // 2)
    sorted_order = np.argsort(distances, axis=1)[:, :at_n]

    for query_i, (closest_inds, order) in enumerate(zip(indices, sorted_order)):
        closest_inds = closest_inds[order]
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


def test_image_model_with_reranking(
    model: nn.Module,
    dataloader: DataLoader,
    dist_thresh: float = 25.0,
    device: str = "cuda",
) -> Tuple[np.ndarray, float, float]:
    """Test Place Recognition Average Recall@N metric performance.

    Args:
        model (nn.Module): The model to test.
        dataloader (DataLoader): Test dataloader object.
        dist_thresh (float): Distance threshold for positive match. Defaults to 25.0.
        device (str): Device ("cpu" or "cuda"). Defaults to "cuda".

    Returns:
        Tuple[np.ndarray, float, float]: Array of AverageRecall@N (N from 1 to 25), AverageRecall@1%
            and mean top-1 distance.
    """
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        front_test_embeddings_list = []
        back_test_embeddings_list = []
        for data in tqdm(dataloader, desc="Calculating test set descriptors"):
            batch_full, _, _ = data
            batch = {"images": batch_full["images_cam5"].to(device)}
            batch_embeddings = model(batch)
            front_test_embeddings_list.append(batch_embeddings["image"].cpu().numpy())
            batch = {"images": batch_full["images_cam2"].to(device)}
            batch_embeddings = model(batch)
            back_test_embeddings_list.append(batch_embeddings["image"].cpu().numpy())
        front_test_embeddings = np.vstack(front_test_embeddings_list)
        back_test_embeddings = np.vstack(back_test_embeddings_list)

    # TODO: resolve mypy typing here
    test_df = dataloader.dataset.dataset_df  # type: ignore

    queries = []
    databases = []

    for _, group in test_df.groupby("track"):
        databases.append(group.index.to_list())
        selected_queries = group[group["in_query"]]
        queries.append(selected_queries.index.to_list())

    utms = torch.tensor(test_df[["northing", "easting"]].to_numpy())
    dist_fn = LpDistance(normalize_embeddings=False)
    dist_utms = dist_fn(utms).numpy()

    n = 25
    recalls_at_n = np.zeros((len(queries), len(databases), n))
    recalls_at_one_percent = np.zeros((len(queries), len(databases), 1))
    top1_distances = np.zeros((len(queries), len(databases), 1))
    ij_permutations = list(itertools.permutations(range(len(queries)), 2))
    count_r_at_1 = 0

    for i, j in tqdm(ij_permutations, desc="Calculating metrics"):
        query = queries[i]
        database = databases[j]
        front_query_embs = front_test_embeddings[query]
        back_query_embs = back_test_embeddings[query]
        database_embs = np.vstack([front_test_embeddings[database], back_test_embeddings[database]])

        distances = dist_utms[query][:, database]
        (
            recalls_at_n[i, j],
            recalls_at_one_percent[i, j],
            top1_distance,
        ) = get_recalls_with_reranking(
            front_query_embs, back_query_embs, database_embs, distances, at_n=n, dist_thresh=dist_thresh
        )

        if top1_distance:
            count_r_at_1 += 1
            top1_distances[i, j] = top1_distance
    mean_recall_at_n = recalls_at_n.sum(axis=(0, 1)) / len(ij_permutations)
    mean_recall_at_one_percent = recalls_at_one_percent.sum(axis=(0, 1)).squeeze() / len(ij_permutations)
    mean_top1_distance = top1_distances.sum(axis=(0, 1)).squeeze() / len(ij_permutations)
    print(f"Mean Recall@N:\n{mean_recall_at_n}")
    print(f"Mean Recall@1% = {mean_recall_at_one_percent}")
    print(f"Mean top-1 distance = {mean_top1_distance}")

    return mean_recall_at_n, mean_recall_at_one_percent, mean_top1_distance
