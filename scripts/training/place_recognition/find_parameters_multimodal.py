"""Script to train a multi-modal Place Recognition model."""
import logging
import pprint
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Literal

import hydra
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from opr.datasets.dataloader_factory import make_dataloaders
from opr.models.place_recognition.base import LateFusionModel
from opr.trainers.place_recognition import MultimodalPlaceRecognitionTrainer
from opr.utils import set_seed

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

logger = logging.getLogger(__name__)


@hydra.main(config_path="../../../configs", config_name="find_parameters_multimodal", version_base=None)
def main(cfg: DictConfig) -> None:
    """Training code.

    Args:
        cfg (DictConfig): config to train with
    """
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    logger.info(f"Config:\n{pprint.pformat(config_dict, compact=True)}")

    if not cfg.debug and not cfg.wandb.disabled:
        config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb.init(
            dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
            name=cfg.exp_name,
            project=cfg.wandb.project,
            settings=wandb.Settings(start_method="thread"),
            config=config_dict,
        )
        logger.debug(f"Initialized wandb run with name: {wandb.run.name}")

    logger.info(f"Output directory: {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")
    checkpoints_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) / "checkpoints"
    if not checkpoints_dir.exists():
        checkpoints_dir.mkdir(parents=True)

    set_seed(seed=cfg.seed, make_deterministic=False)
    logger.info(f"=> Seed: {cfg.seed}")

    logger.debug("=> Instantiating model...")
    model = instantiate(cfg.model)

    logger.debug("=> Instantiating loss...")
    loss_fn = instantiate(cfg.loss)

    logger.debug("=> Making dataloaders...")
    dataloaders: Dict[Literal["train", "val", "test"], DataLoader] = make_dataloaders(
        dataset_cfg=cfg.dataset,
        batch_sampler_cfg=cfg.sampler,
        num_workers=cfg.num_workers,
    )

    logger.debug("=> Instantiating optimizer...")
    params_l = []
    if isinstance(model, LateFusionModel):
        # Different LR for image feature extractor (pretrained ResNet)
        if model.image_module is not None:
            params_l.append({"params": model.image_module.parameters(), "lr": cfg.optimizer.image_lr})
        if model.cloud_module is not None:
            params_l.append({"params": model.cloud_module.parameters(), "lr": cfg.optimizer.cloud_lr})
        if model.fusion_module is not None:
            params_l.append({"params": model.fusion_module.parameters(), "lr": cfg.optimizer.fusion_lr})
    else:
        # All parameters use the same lr
        params_l.append({"params": model.parameters(), "lr": cfg.optimizer.lr})
    if cfg.optimizer._target_ == "torch.optim.Adam":
        optimizer = torch.optim.Adam(params_l, weight_decay=cfg.optimizer.weight_decay)
    elif cfg.optimizer._target_ == "torch.optim.AdamW":
        optimizer = torch.optim.AdamW(params_l, weight_decay=cfg.optimizer.weight_decay)

    logger.debug("=> Instantiating scheduler...")
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    logger.debug("=> Instantiating trainer...")

    modalities_weights = cfg.modalities_weights
    print(modalities_weights, type(modalities_weights))

    trainer = MultimodalPlaceRecognitionTrainer(
        checkpoints_dir=checkpoints_dir,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        batch_expansion_threshold=cfg.batch_expansion_threshold,
        wandb_log=(not cfg.debug and not cfg.wandb.disabled),
        device=cfg.device,
        modalities_weights=modalities_weights,
    )

    logger.info(f"=====> {trainer.__class__.__name__} is ready, starting training for {cfg.epochs} epochs.")

    trainer.train(
        epochs=cfg.epochs,
        train_dataloader=dataloaders["train"],
        val_dataloader=dataloaders["val"],
        test_dataloader=dataloaders["test"],
    )

    logger.info("Training completed. Testing the best model on the test set.")
    best_ckpt = torch.load(checkpoints_dir / "best.pth")
    trainer.model.load_state_dict(best_ckpt["model_state_dict"])
    trainer.test(dataloader=dataloaders["test"])
    logger.info("Testing completed.")
    wandb.finish()


if __name__ == "__main__":
    run_dir = REPO_ROOT / "outputs" / (r"${exp_name}" + f"_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")
    sys.argv.append(f"hydra.run.dir={run_dir}")
    main()
