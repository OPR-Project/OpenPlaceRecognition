import argparse
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from hydra import compose, initialize
from hydra.utils import instantiate
from opr.testing import test
from opr.datasets.dataloader_factory import make_collate_fn


def parse_args() -> Tuple[Path, Path]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the checkpoint file.")
    parser.add_argument(
        "--config", type=Path, help="path to the model config file (if config not saved in checkpoint)."
    )
    args = parser.parse_args()
    return args.checkpoint, args.config


if __name__ == "__main__":
    ckpt_path, cfg_path = parse_args()
    checkpoint = torch.load(ckpt_path)

    print("\n=> Loading config")
    if cfg_path is not None:
        print("Loading config from given path...")
        with initialize(config_path=str(cfg_path.parent), job_name="test_model", version_base=None):
            cfg = compose(config_name=cfg_path.name)
    elif "config" in checkpoint:
        print("Loading config from checkpoint...")
        cfg = checkpoint["config"]
    else:
        raise ValueError(
            "There is no config saved in checkpoint file, provide explicit path in '--config' argument"
        )

    print("\n=> Instantiating model")
    model = instantiate(cfg.model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(cfg.general.device)
    model.eval()

    print("\n=> Instantiating dataloader")
    dataset = instantiate(cfg.dataset.dataset, subset="test")
    if "batch_size_limit" not in cfg.dataset.sampler:
        batch_size = cfg.dataset.sampler.batch_size
    else:
        batch_size = cfg.dataset.sampler.batch_size_limit
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=make_collate_fn(dataset, batch_split_size=None),
        num_workers=cfg.general.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print("\n=> Testing:\n")
    recall_at_n, recall_at_one_percent, mean_top1_distance = test(
        model=model,
        descriptor_key=cfg.general.test_modality,
        dataloader=dataloader,
        device=cfg.general.device,
    )
