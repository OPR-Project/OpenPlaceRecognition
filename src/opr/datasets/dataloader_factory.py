"""Functions to create PyTorch DataLoaders for different datasets."""
from typing import Dict, Optional

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from opr.samplers.batch_sampler import DistributedBatchSamplerWrapper


def make_dataloaders(
    dataset_cfg: DictConfig,
    batch_sampler_cfg: DictConfig,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """Function to create DataLoader objects from given dataset and sampler configs.

    Args:
        dataset_cfg (DictConfig): Dataset configuration.
        batch_sampler_cfg (DictConfig): Batch sampler configuration.
        num_workers (int): Number of workers for DataLoader. Defaults to 0.

    Returns:
        Dict[str, DataLoader]: Dictionary with DataLoaders.
    """
    dataset = {}
    for subset in ["train", "val", "test"]:
        dataset[subset] = instantiate(dataset_cfg, subset=subset)

    batch_split_size: Dict[str, Optional[int]] = {}
    if "batch_split_size" not in batch_sampler_cfg:
        batch_split_size["train"] = None
        batch_split_size["val"] = None
    else:
        batch_split_size["train"] = batch_sampler_cfg.batch_split_size
        batch_split_size["val"] = batch_sampler_cfg.batch_split_size

    sampler = {}
    sampler["train"] = instantiate(batch_sampler_cfg, dataset=dataset["train"])
    if "val_batch_size" in batch_sampler_cfg and batch_sampler_cfg.val_batch_size is not None:
        val_batch_size = batch_sampler_cfg.val_batch_size
        sampler["val"] = instantiate(
            batch_sampler_cfg,
            dataset=dataset["val"],
            batch_size=val_batch_size,
            batch_size_limit=None,
            batch_expansion_rate=None,
        )
        batch_split_size["val"] = None
    elif "batch_size_limit" not in batch_sampler_cfg or batch_sampler_cfg.batch_size_limit is None:
        val_batch_size = batch_sampler_cfg.batch_size
        sampler["val"] = instantiate(batch_sampler_cfg, dataset=dataset["val"])
    else:
        val_batch_size = batch_sampler_cfg.batch_size_limit
        sampler["val"] = instantiate(
            batch_sampler_cfg,
            dataset=dataset["val"],
            batch_size=val_batch_size,
            batch_size_limit=None,
            batch_expansion_rate=None,
        )

    dataloaders = {}
    for subset in ["train", "val"]:
        dataloaders[subset] = DataLoader(
            dataset=dataset[subset],
            batch_sampler=sampler[subset],
            collate_fn=dataset[subset].collate_fn,
            num_workers=num_workers,
            pin_memory=True,
        )
    dataloaders["test"] = DataLoader(
        dataset=dataset["test"],
        batch_size=val_batch_size,
        collate_fn=dataset[subset].collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return dataloaders


def make_distributed_dataloaders(
    dataset_cfg: DictConfig,
    batch_sampler_cfg: DictConfig,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """Function to create DataLoader objects from given dataset and sampler configs.

    Args:
        dataset_cfg (DictConfig): Dataset configuration.
        batch_sampler_cfg (DictConfig): Batch sampler configuration.
        num_workers (int): Number of workers for DataLoader. Defaults to 0.

    Returns:
        Dict[str, DataLoader]: Dictionary with DataLoaders.
    """
    dataset = {}
    for subset in ["train", "val", "test"]:
        dataset[subset] = instantiate(dataset_cfg, subset=subset)

    batch_split_size: Dict[str, Optional[int]] = {}
    if "batch_split_size" not in batch_sampler_cfg:
        batch_split_size["train"] = None
        batch_split_size["val"] = None
    else:
        batch_split_size["train"] = batch_sampler_cfg.batch_split_size
        batch_split_size["val"] = batch_sampler_cfg.batch_split_size

    sampler = {}
    sampler["train"] = instantiate(batch_sampler_cfg, dataset=dataset["train"])
    if "val_batch_size" in batch_sampler_cfg and batch_sampler_cfg.val_batch_size is not None:
        val_batch_size = batch_sampler_cfg.val_batch_size
        sampler["val"] = instantiate(
            batch_sampler_cfg,
            dataset=dataset["val"],
            batch_size=val_batch_size,
            batch_size_limit=None,
            batch_expansion_rate=None,
        )
        batch_split_size["val"] = None
    elif "batch_size_limit" not in batch_sampler_cfg or batch_sampler_cfg.batch_size_limit is None:
        val_batch_size = batch_sampler_cfg.batch_size
        sampler["val"] = instantiate(batch_sampler_cfg, dataset=dataset["val"])
    else:
        val_batch_size = batch_sampler_cfg.batch_size_limit
        sampler["val"] = instantiate(
            batch_sampler_cfg,
            dataset=dataset["val"],
            batch_size=val_batch_size,
            batch_size_limit=None,
            batch_expansion_rate=None,
        )

    dataloaders = {}
    for subset in ["train", "val"]:
        dataloaders[subset] = DataLoader(
            dataset=dataset[subset],
            batch_sampler=DistributedBatchSamplerWrapper(sampler[subset]),
            collate_fn=dataset[subset].distributed_collate_fn,
            num_workers=num_workers,
            pin_memory=True,
        )
    dataloaders["test"] = DataLoader(
        dataset=dataset["test"],
        batch_size=val_batch_size // torch.distributed.get_world_size(),  # TODO: may it be done better?
        collate_fn=dataset[subset].collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return dataloaders
