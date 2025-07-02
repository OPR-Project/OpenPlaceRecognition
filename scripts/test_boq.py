from copy import copy
from pathlib import Path
from time import time

import faiss
import wandb
from tqdm import tqdm
from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation
import torch
from torch.utils.data import DataLoader
import pandas as pd
import hydra

from opr.pipelines.place_recognition import PlaceRecognitionPipeline


def pose_to_matrix(pose):
    """From the 6D poses in the [tx ty tz qx qy qz qw] format to 4x4 pose matrices."""
    position = pose[:3]
    pose_matrix = np.eye(4)
    pose_matrix[:3,3] = position
    return pose_matrix


def compute_error(estimated_pose, gt_pose):
    """For the 6D poses in the [tx ty tz qx qy qz qw] format."""
    estimated_pose = pose_to_matrix(estimated_pose)
    gt_pose = pose_to_matrix(gt_pose)
    error_pose = np.linalg.inv(estimated_pose) @ gt_pose
    dist_error = np.sum(error_pose[:3, 3]**2) ** 0.5
    return dist_error


@hydra.main(version_base=None, config_path="../configs", config_name="test_nclt")
def main(cfg: DictConfig) -> None:
    # Initialize wandb
    # name = f"{cfg.model.dino_model}_pr{cfg.pr_threshold}_{cfg.dataset.subset}_pool-{cfg.model.pooling}_facet-{cfg.model.facet}_cls-{cfg.model.use_cls}_layer-{cfg.model.layer}"
    # name = "theia"
    name = "dinov2"
    wandb.init(project="place_recognition", name=name)
    dataset_root = cfg.dataset.dataset_root
    weights_path = cfg.model.weights_path
    print('HELLOOOOO!!:: ', cfg.dataset.dataset_root)

    track_list = sorted([str(subdir.name) for subdir in Path(dataset_root).iterdir() if subdir.is_dir()])
    print(f"Found {len(track_list)} tracks")
    print(track_list)

    model = instantiate(cfg.model)
    print(f"cfg.model: {cfg.model}")
    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path))
    model = model.to(cfg.device)
    model.eval()

    print(f"model instantiated")

    dataset = instantiate(cfg.dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=dataset.collate_fn,
    )

    descriptors = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = {k: v.to(cfg.device) for k, v in batch.items()}
            # print(f"batch.dtype: {batch.dtype}")
            out = model(batch)
            
            # print(f"out.shape: {out.shape}")
            final_descriptor = model(batch)["final_descriptor"]

            print(f"final_descriptor.shape: {final_descriptor.shape}")
            
            descriptors.append(final_descriptor.squeeze(1).detach().cpu().numpy())

    descriptors = np.concatenate(descriptors, axis=0)
    print(f"descriptors.shape: {descriptors.shape}")

    # exit()

    dataset_df = dataset.dataset_df

    for track, indices in dataset_df.groupby("track").groups.items():
        track_descriptors = descriptors[indices]
        track_index = faiss.IndexFlatL2(track_descriptors.shape[1])
        print(f"track_descriptors.shape: {track_descriptors.shape}")
        track_index.add(track_descriptors)
        faiss.write_index(track_index, f"{dataset_root}/{track}/index.faiss")
        print(f"Saved index {dataset_root}/{track}/index.faiss")

    test_csv = pd.read_csv(Path(dataset_root) / "test.csv", index_col=0)

    all_recalls = []
    all_mean_dist_errors = []
    all_mean_angle_errors = []
    all_median_dist_errors = []
    all_median_angle_errors = []
    all_times = []

    for db_track in track_list:
        pipe = PlaceRecognitionPipeline(
            database_dir=Path(dataset_root) / db_track,
            model=model,
            model_weights_path=weights_path,
            device=cfg.device,
        )
        for query_track in track_list:
            if db_track == query_track:
                continue
            query_dataset = copy(dataset)
            query_dataset.dataset_df = query_dataset.dataset_df[query_dataset.dataset_df["track"] == query_track]
            query_df = pd.read_csv(Path(dataset_root) / query_track / "track.csv", index_col=0)

            # filter out only test subset
            query_df = query_df[query_df['image'].isin(query_dataset.dataset_df['image'])].reset_index(drop=True)
            # and do not forget to change the database_df in the pipeline
            pipe.database_df = pipe.database_df[pipe.database_df['image'].isin(test_csv['image'])].reset_index(drop=True)

            pr_matches = []
            dist_errors = []
            angle_errors = []
            times = []

            true_pairs = []
            false_pairs = []

            for q_i, query in tqdm(enumerate(query_dataset)):
                # print(query_df.columns)
                query["pose"] = query_df.iloc[q_i][['northing', 'easting', 'down']].to_numpy()
                t = time()
                output = pipe.infer(query)
                times.append(time() - t)
                dist_error = compute_error(output["pose"], query["pose"])
                pr_matches.append(dist_error < cfg.pr_threshold)
                dist_errors.append(dist_error)
                if dist_error < 10:
                    true_pairs.append((q_i, output["idx"]))
                elif dist_error > 100:
                    false_pairs.append((q_i, output["idx"]))

            all_recalls.append(np.mean(pr_matches))
            all_mean_dist_errors.append(np.mean(dist_errors))
            all_median_dist_errors.append(np.median(dist_errors))
            all_times.extend(times[1:]) # drop the first iteration cause it is always slower

            # Log pair results to wandb
            wandb.log({
                f"pair/{db_track}_{query_track}/recall": np.mean(pr_matches),
                f"pair/{db_track}_{query_track}/mean_dist_error": np.mean(dist_errors),
                f"pair/{db_track}_{query_track}/median_dist_error": np.median(dist_errors),
                f"pair/{db_track}_{query_track}/mean_time": np.mean(times[1:])
            })

    # Calculate and log overall results
    overall_results = {
        "overall/recall": np.mean(all_recalls)*100,
        "overall/mean_dist_error": np.mean(all_mean_dist_errors),
        "overall/median_dist_error": np.mean(all_median_dist_errors),
        "overall/mean_time": np.mean(all_times)
    }
    wandb.log(overall_results)
    print(overall_results)
    results_str = f"""Average Recall@1: {overall_results['overall/recall']:.2f}
Average mean dist error: {overall_results['overall/mean_dist_error']:.2f}
Average mean angle error: {overall_results['overall/mean_angle_error']:.2f}
Average median dist error: {overall_results['overall/median_dist_error']:.2f}
Average median angle error: {overall_results['overall/median_angle_error']:.2f}
"""
    print(results_str)
    wandb.finish()


if __name__ == "__main__":
    main()
    # Example hydra multirun command:
    # python test.py -m dataset=nclt model=resnet18fpn,resnet50fpn pr_threshold=5,10,15

    # This will run the test script with:
    # - Two different models (resnet18fpn and resnet50fpn)
    # - Three different PR thresholds (5m, 10m, 15m)
    # - Using the NCLT dataset config
    # Resulting in 6 total runs (2 models x 3 thresholds)

    # You can also override specific config values:
    # python test.py -m dataset.positive_threshold=5,10 dataset.negative_threshold=25,50
