from collections import defaultdict
import itertools
from typing import Optional, Tuple, Union, List

import numpy as np
import torch
from pytorch_metric_learning.distances import LpDistance
from sklearn.neighbors import KDTree
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import faiss

from opr.utils import parse_device

import time


def get_recalls(
    query_indices: List[int],
    query_embs: np.ndarray,
    query_segment_to_index: List[int],
    db_indices: List[int],
    db_embs: np.ndarray,
    db_segment_to_index: List[int],
    dist_matrix: np.ndarray,
    dist_thresh: float = 25.0,
    at_n: int = 25,
    aggregation: str = "max",
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
    positives_mask = dist_matrix <= dist_thresh
    query_matches = positives_mask.sum(axis=1) > 0
    queries_with_matches = np.count_nonzero(query_matches)
    assert queries_with_matches != 0

    recall_at_n = np.zeros((at_n,), dtype=float)

    top1_distances, errors_query, errors_db, gt_db, errors_dists = [], [], [], [], []

    one_percent_threshold = max(int(round(len(db_indices) / 100.0)), 1)
    if one_percent_threshold >= at_n:
        raise ValueError(f"at_n is required to be greater than one percent threshold: {one_percent_threshold}")

    db_embs = db_embs / np.linalg.norm(db_embs, axis=1, keepdims=True)
    query_embs = query_embs / np.linalg.norm(query_embs, axis=1, keepdims=True)

    index = faiss.IndexFlatL2(db_embs.shape[1])
    index.add(db_embs)
    distances, indices = index.search(query_embs, k=30)
    distances = (distances - distances.min()) / (distances.max() - distances.min())
    similarities = 1 - distances

    db_idx_to_pos = {db_ind: i for i, db_ind in enumerate(db_indices)}
    query_idx_to_pos = {query_ind: i for i, query_ind in enumerate(query_indices)}
    segment_sim_accumulator = np.zeros((len(indices), len(db_indices)), dtype=np.float32)

    sim_values = defaultdict(list)

    for q_seg_idx, db_seg_indices in enumerate(indices):
        q_img_idx = query_segment_to_index[q_seg_idx]
        q_pos = query_idx_to_pos[q_img_idx]
        for i, db_seg_idx in enumerate(db_seg_indices):
            db_img_idx = db_segment_to_index[db_seg_idx]
            d_pos = db_idx_to_pos[db_img_idx]
            sim = similarities[q_seg_idx, i]
            segment_sim_accumulator[q_seg_idx, d_pos] = max(sim, segment_sim_accumulator[q_seg_idx, d_pos])
        
        checked_d_pos = set()
        for db_seg_idx in db_seg_indices:
            db_img_idx = db_segment_to_index[db_seg_idx]
            d_pos = db_idx_to_pos[db_img_idx]
            if d_pos in checked_d_pos:
                continue
            checked_d_pos.add(d_pos)
            sim = segment_sim_accumulator[q_seg_idx, d_pos]
            sim_values[(q_pos, d_pos)].append(sim)

    sim_accumulator = np.zeros((len(query_indices), len(db_indices)), dtype=np.float32)

    for (q_pos, d_pos), sims in sim_values.items():
        if aggregation == "max":
            sim_accumulator[q_pos, d_pos] = max(sims)
        elif aggregation == "sum":
            sim_accumulator[q_pos, d_pos] = sum(sims)
        elif aggregation == "mean":
            sim_accumulator[q_pos, d_pos] = np.mean(sims)
        elif aggregation.startswith("topk"):
            try:
                k = int(aggregation[4:])
            except ValueError:
                raise ValueError(f"Invalid top-k format: {aggregation}")
            top_k = sorted(sims, reverse=True)[:k]
            sim_accumulator[q_pos, d_pos] = np.mean(top_k)
        elif aggregation == "softmax":
            exp_sims = np.exp(sims)
            weights = exp_sims / np.sum(exp_sims)
            sim_accumulator[q_pos, d_pos] = np.sum(np.array(sims) * weights)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

    top_n_indices = np.argpartition(-sim_accumulator, at_n, axis=1)[:, :at_n]
    top_n_sorted = np.take_along_axis(top_n_indices,
                                      np.argsort(-np.take_along_axis(sim_accumulator, top_n_indices, axis=1)), 
                                      axis=1)

    for q_i, top_dbs in enumerate(top_n_sorted):
        match_mask = positives_mask[q_i][top_dbs]
        if match_mask[0]:
            #top1_distances.append(distances[query_i][0])
            top1_distances.append(0)
        elif query_matches[q_i]:
            #errors_dists.append(distances[query_i][0])
            errors_dists.append(0)
            errors_query.append(q_i)
            errors_db.append(top_dbs[0])
            gt_db.append(dist_matrix[q_i].argmin())
        recall_at_n += np.cumsum(match_mask, dtype=bool)

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
    distance_threshold: float = 5.0,
    device: Union[str, int, torch.device] = "cuda",
    aggregation: str = "max",
    at_n: int = 10,
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
    segments_cnt = 0
    start = time.perf_counter()
    with torch.no_grad():
        embeddings_list = []
        image_segments = {}
        image_index = 0
        for batch in tqdm(dataloader, desc="Calculating test set descriptors", leave=False):
            for key in batch:
                if key.startswith("images_"): # SegBoQ needs to make crops and then transform them, so images need to be on CPU
                    continue
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)
                elif isinstance(batch[key], list) and isinstance(batch[key][0], torch.Tensor):
                    batch[key] = [t.to(device, non_blocking=True) for t in batch[key]]
            embeddings = model(batch)["final_descriptor"]

            batch_embs = []
            batch_segments = []

            for segments_list in embeddings:
                segment_indices = []
                for segment_emb in segments_list:
                    segment_indices.append(segments_cnt)
                    batch_embs.append(segment_emb)
                    segments_cnt += 1
                batch_segments.append(segment_indices)
                image_index += 1

            embeddings_list.extend(batch_embs)
            for idx, segs in enumerate(batch_segments):
                image_segments[image_index - len(batch_segments) + idx] = segs
        test_embeddings = torch.stack(embeddings_list).cpu().numpy()

    end = time.perf_counter()

    print(f"Elapsed time for embeddings: {end - start:.4f} seconds")

    test_df = dataloader.dataset.dataset_df
    print('Segments count:', segments_cnt)

    queries = []
    databases = []

    for _, group in test_df.groupby("track"):
        databases.append(group.index.to_list())
        selected_queries = group[group["in_query"]]
        queries.append(selected_queries.index.to_list())

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

    recalls_at_n = np.zeros((len(queries), len(databases), at_n))
    recalls_at_one_percent = np.zeros((len(queries), len(databases), 1))
    top1_distances = np.zeros((len(queries), len(databases), 1))
    ij_permutations = list(itertools.permutations(range(len(queries)), 2))
    count_r_at_1 = 0

    recalls_time = 0.0

    for i, j in tqdm(ij_permutations, desc="Calculating metrics", leave=False):
        query = queries[i]
        database = databases[j]

        query_segment_indices = [ind for image_ind in query for ind in image_segments[image_ind]]
        query_segment_to_image = [image_ind for image_ind in query for ind in image_segments[image_ind]]
        query_embs = test_embeddings[query_segment_indices]

        database_segment_indices = [ind for image_ind in database for ind in image_segments[image_ind]]
        database_segment_to_image = [image_ind for image_ind in database for ind in image_segments[image_ind]]
        database_embs = test_embeddings[database_segment_indices]

        start = time.perf_counter()

        distances = dist_utms[query][:, database]
        (
            recalls_at_n[i, j],
            recalls_at_one_percent[i, j],
            top1_distance,
        ) = get_recalls(query, query_embs, query_segment_to_image, database, database_embs, database_segment_to_image,
                        distances, at_n=at_n, dist_thresh=distance_threshold, aggregation=aggregation)
        
        end = time.perf_counter()

        recalls_time += end - start

        if top1_distance:
            count_r_at_1 += 1
            top1_distances[i, j] = top1_distance

    print(f"Elapsed time for recalls: {recalls_time / 2:.4f} seconds")
    mean_recall_at_n = recalls_at_n.sum(axis=(0, 1)) / len(ij_permutations)
    mean_recall_at_one_percent = recalls_at_one_percent.sum(axis=(0, 1)).squeeze() / len(ij_permutations)
    mean_top1_distance = top1_distances.sum(axis=(0, 1)).squeeze() / len(ij_permutations)

    return mean_recall_at_n, mean_recall_at_one_percent, mean_top1_distance
