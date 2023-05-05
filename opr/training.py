"""Training functions implementation."""
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from opr.losses import MultimodalTripletMarginLoss
from opr.models.base_models import ComposedModel
from opr.utils import accumulate_dict, compute_epoch_stats_mean, merge_nested_dicts


def step(
    data: Tuple[Dict[str, Tensor], Tensor, Tensor],
    model: ComposedModel,
    loss_fn: MultimodalTripletMarginLoss,
    optimizer: torch.optim.Optimizer,
    phase: str = "train",
    device: str = "cuda",
) -> Dict[str, Any]:
    """Train/val step function.

    Args:
        data (Tuple[Dict[str, Tensor], Tensor, Tensor]): Input batch data.
        model (ComposedModel): Model object.
        loss_fn (MultimodalTripletMarginLoss): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer function.
        phase (str): Current phase ("train" or "val").. Defaults to "train".
        device (str): Device ("cpu" or "cuda"). Defaults to "cuda".

    Returns:
        Dict[str, Any]: Stats dictionary.
    """
    assert phase in ["train", "val"]

    batch, positives_mask, negatives_mask = data
    batch = {e: batch[e].to(device) for e in batch}

    if phase == "train":
        model.train()
    else:
        model.eval()

    optimizer.zero_grad()

    with torch.set_grad_enabled(phase == "train"):
        output = model(batch)
        loss, loss_stats, miner_stats = loss_fn(
            model_output=output,
            positives_mask=positives_mask,
            negatives_mask=negatives_mask,
        )
        if phase == "train":
            loss.backward()
            optimizer.step()

        stats = merge_nested_dicts(loss_stats, miner_stats)

    return stats


def epoch_loop(
    dataloader: DataLoader,
    model: ComposedModel,
    loss_fn: MultimodalTripletMarginLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.MultiStepLR] = None,
    phase: str = "train",
    device: str = "cuda",
) -> Tuple[Dict[str, Any], Optional[float]]:
    """Performs one full epoch.

    Args:
        dataloader (DataLoader): Dataloader for current phase.
        model (ComposedModel): Model object.
        loss_fn (MultimodalTripletMarginLoss): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer function.
        scheduler (torch.optim.lr_scheduler.MultiStepLR, optional): Scheduler function. Defaults to None.
        phase (str): Current phase ("train" or "val"). Defaults to "train".
        device (str): Device ("cpu" or "cuda"). Defaults to "cuda".

    Returns:
        Tuple[Dict[str, Any], Optional[float]]: Stats dictionary and optional mean non-zero-rate
            value (if dynamic batch sizing implemented).
    """
    assert phase in ["train", "val"]
    step_fn = step
    total_steps = len(dataloader)

    if phase == "train":
        model.train()
    else:
        model.eval()

    epoch_stats: Dict[str, Any] = {}
    # loss_stats_bar = tqdm(total=0, position=1, bar_format="{desc}")
    # miner_stats_bar = tqdm(total=0, position=2, bar_format="{desc}")

    for _, batch_data in tqdm(
        enumerate(dataloader),
        desc=phase,
        total=total_steps,
        dynamic_ncols=True,
        leave=True,
        position=0,
    ):
        step_stats = step_fn(batch_data, model, loss_fn, optimizer, phase, device)
        epoch_stats = accumulate_dict(epoch_stats, step_stats)
        torch.cuda.empty_cache()

    if scheduler is not None and phase == "train":
        scheduler.step()

    final_stats = compute_epoch_stats_mean(epoch_stats)

    non_zero_rates = []
    for modality, modality_value in final_stats.items():
        if modality in ("image", "cloud", "semantic", "fusion"):
            if "non_zero_rate" in modality_value.keys():  # TODO: works only with HardTripletMarginLoss
                non_zero_rates.append(final_stats[modality]["non_zero_rate"])
    if len(non_zero_rates) > 0:
        mean_non_zero_rate = float(np.mean(non_zero_rates))
        final_stats["mean_non_zero_rate"] = mean_non_zero_rate
    else:
        mean_non_zero_rate = None

    return final_stats, mean_non_zero_rate
