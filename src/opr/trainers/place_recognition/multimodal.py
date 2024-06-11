"""Multimodal Place Recognition Trainer."""
from time import time
from typing import Optional

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from opr.trainers.place_recognition.unimodal import UnimodalPlaceRecognitionTrainer
from opr.utils import accumulate_dict, compute_epoch_stats_mean


class MultimodalPlaceRecognitionTrainer(UnimodalPlaceRecognitionTrainer):
    """Multimodal Place Recognition Trainer."""

    def __init__(self, modalities_weights: dict[str, float], *args, **kwargs) -> None:  # noqa: ANN002,ANN003
        """Initialize the MultimodalTrainer object.

        Args:
            modalities_weights (dict): A dict of weights for each modality.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.modalities_weights = modalities_weights

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
                    stats = {}
                    losses = {}
                    embeddings = self.model(batch)
                    # self.modalities_weights instead of hardcoded list
                    for modality in ["image", "cloud", "semantic", "text", "final_descriptor"]:
                        if modality in embeddings:
                            mod_loss, mod_stats = self.loss_fn(
                                embeddings[modality], positives_mask, negatives_mask
                            )
                            stats[modality] = mod_stats
                            losses[modality] = mod_loss

                    non_zero_rate = np.mean([i["non_zero_rate"] for i in stats.values()])
                    stats["non_zero_rate"] = non_zero_rate

                if stage == "train":
                    # Sum the losses with weights
                    loss = sum(losses[modality] * self.modalities_weights[modality] for modality in losses)
                    # stats["total_loss"] = loss

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
