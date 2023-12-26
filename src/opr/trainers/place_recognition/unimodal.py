"""Pointcloud Place Recognition trainer."""
import itertools
import logging
from os import PathLike
from pathlib import Path
from time import time
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import wandb
from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.utils import common_functions as c_f
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger

from opr.testing import get_recalls
from opr.utils import (
    accumulate_dict,
    compute_epoch_stats_mean,
    flatten_dict,
    parse_device,
)

c_f.COLLECT_STATS = True


# TODO: Think about naming
class UnimodalPlaceRecognitionTrainer:
    """Single modality Place Recognition trainer."""

    def __init__(
        self,
        checkpoints_dir: Union[str, PathLike],
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[Any] = None,
        batch_expansion_threshold: Optional[int] = None,
        wandb_log: bool = False,
        device: Union[str, int, torch.device] = "cuda",
    ) -> None:
        """Single modality Place Recognition trainer.

        Args:
            checkpoints_dir (Union[str, PathLike]): Path to save checkpoints.
            model (nn.Module): Model to train. The forward method should have a defined interface.
                See the documentation for details.
            loss_fn (nn.Module): Loss function. Should take in embeddings, positives_mask and negatives_mask
                and return a loss and stats dictionary.
            optimizer (Optimizer): Optimizer to use.
            scheduler (optional): Scheduler to use. Defaults to None.
            batch_expansion_threshold (int, optional): Batch expansion threshold if dynamic batch sizing
                strategy is used. Defaults to None.
            wandb_log (bool): Whether to use wandb for remote logging. Defaults to False.
            device (Union[str, int, torch.device]): Device to train on. Defaults to "cuda".

        Raises:
            ValueError: If CUDA device is set but not available.
        """
        # logger = logging.getLogger(self.__class__.__name__)

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.checkpoints_dir = Path(checkpoints_dir)
        self.batch_expansion_threshold = batch_expansion_threshold

        self.wandb_log = wandb_log
        self.device = parse_device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device is not available.")

        self.model = self.model.to(self.device)

        self._stats: Dict[str, Dict[str, float]] = {"train": {}}
        self.best_recall_at_1 = 0.0

    @property
    def stats(self) -> Dict[str, Any]:
        """Dictionary with statistics for the last epoch."""
        return self._stats

    def train(
        self,
        epochs: int,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        test_dataloader: Optional[DataLoader] = None,
        test_every_n_epochs: int = 1,
    ) -> None:
        """Trains the single modal Place Recognition model for the specified number of epochs.

        Args:
            epochs (int): The number of epochs to train for.
            train_dataloader (DataLoader): The data loader for the training set.
            val_dataloader (Optional[DataLoader]): The data loader for the validation set.
            test_dataloader (Optional[DataLoader]): The data loader for the test set.
            test_every_n_epochs (int): The frequency (in epochs) at which to evaluate the model on the test set.
        """
        for epoch in range(epochs):
            self._stats = {"train": {}}
            logger.info(f"=====> Epoch: {epoch+1:3d}/{epochs}:")

            # === Train-Val stage ===
            self._loop_epoch(train_dataloader, val_dataloader)
            self._stats["train"]["batch_size"] = train_dataloader.batch_sampler.batch_size
            if val_dataloader:
                self._stats["val"]["batch_size"] = val_dataloader.batch_sampler.batch_size

            if self.scheduler:
                self._stats["train"]["lr"] = self.scheduler.get_last_lr()[0]
                self.scheduler.step()
            else:
                self._stats["train"]["lr"] = self.optimizer.param_groups[0]["lr"]

            if (
                self.batch_expansion_threshold is not None
                and self._stats["train"]["non_zero_rate"] < self.batch_expansion_threshold
            ):
                logger.info(
                    f"Non-zero rate is below threshold: {self._stats['train']['non_zero_rate']:.03f} < "
                    f"{self.batch_expansion_threshold}."
                )
                train_dataloader.batch_sampler.expand_batch()

            # === Test stage ===
            if test_dataloader and epoch % test_every_n_epochs == 0:
                self._stats["test"] = {}
                self.test(test_dataloader)

            # === Checkpointing ===
            checkpoint_dict = {
                "epoch": epoch + 1,
                "stats_dict": self._stats,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            }
            if self.scheduler:
                checkpoint_dict["scheduler_state_dict"] = self.scheduler.state_dict()

            torch.save(checkpoint_dict, self.checkpoints_dir / "last.pth")

            if self.wandb_log:
                wandb.log(data=flatten_dict(self._stats), step=epoch)
                wandb.save(str(self.checkpoints_dir / "last.pth"))

            if "test" in self._stats and self._stats["test"]["mean_recall_at_1"] > self.best_recall_at_1:
                logger.info("Recall@1 improved!")
                torch.save(checkpoint_dict, self.checkpoints_dir / "best.pth")
                self.best_recall_at_1 = self._stats["test"]["mean_recall_at_1"]
                if self.wandb_log:
                    wandb.save(str(self.checkpoints_dir / "best.pth"))

    def test(self, dataloader: DataLoader, distance_threshold: float = 25.0) -> None:
        """Evaluates the model on the test set.

        Args:
            dataloader (DataLoader): The data loader for the test set.
            distance_threshold (float): The distance threshold for a correct match. Defaults to 25.0.
        """
        logger.info("=> Test stage:")
        start_t = time()
        self.model.eval()
        with torch.no_grad():
            embeddings_list = []
            for batch in tqdm(dataloader, desc="Calculating test set descriptors", leave=False):
                batch = {e: batch[e].to(self.device) for e in batch}
                embeddings = self.model(batch)["final_descriptor"]
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
        
        
        logger.debug(f"Test embeddings: {test_embeddings.shape}")

        utms = torch.tensor(test_df[["tx", "ty"]].to_numpy())
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
        elapsed_t = time() - start_t
        minutes, seconds = divmod(int(elapsed_t), 60)
        logger.info(f"Test time: {int(minutes):02d}:{int(seconds):02d}")
        logger.info(f"Mean Recall@N:\n{mean_recall_at_n}")
        logger.info(f"Mean Recall@1% = {mean_recall_at_one_percent}")
        logger.info(f"Mean top-1 distance = {mean_top1_distance}")
        self._stats["test"]["mean_recall_at_1"] = mean_recall_at_n[0]
        self._stats["test"]["mean_recall_at_3"] = mean_recall_at_n[2]
        self._stats["test"]["mean_recall_at_5"] = mean_recall_at_n[4]
        self._stats["test"]["mean_recall_at_10"] = mean_recall_at_n[9]
        self._stats["test"]["mean_recall_at_1%"] = mean_recall_at_one_percent
        self._stats["test"]["mean_top1_distance"] = mean_top1_distance

    def _loop_epoch(self, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None) -> None:
        dataloaders = {"train": train_dataloader}
        if val_dataloader:
            dataloaders["val"] = val_dataloader
        for stage, dataloader in dataloaders.items():
            logger.info(f"=> {stage.capitalize()} stage:")
            start_t = time()
            self.model.train(stage == "train")
            accumulated_stats = {}
            for batch in tqdm(
                dataloader,
                desc=stage.capitalize(),
                total=len(dataloader),
                dynamic_ncols=True,
                leave=False,
                position=0,
            ):
                idxs = batch["idxs"]
                positives_mask = dataloader.dataset.positives_mask[idxs][:, idxs]
                negatives_mask = dataloader.dataset.negatives_mask[idxs][:, idxs]
                batch = {e: batch[e].to(self.device) for e in batch if e not in ["idxs", "utms"]}

                with torch.set_grad_enabled(stage == "train"):
                    embeddings = self.model(batch)["final_descriptor"]
                    loss, stats = self.loss_fn(embeddings, positives_mask, negatives_mask)

                if stage == "train":
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                accumulated_stats = accumulate_dict(accumulated_stats, stats)

                torch.cuda.empty_cache()

            epoch_stats = compute_epoch_stats_mean(accumulated_stats)
            elapsed_t = time() - start_t
            minutes, seconds = divmod(int(elapsed_t), 60)
            logger.info(f"{stage.capitalize()} time: {int(minutes):02d}:{int(seconds):02d}")
            logger.info(f"{stage.capitalize()} stats: {epoch_stats}")
            self._stats[stage] = epoch_stats
