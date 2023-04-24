"""Sample training script: trains MinkLoc++ model on Oxford RobotCar dataset."""
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from opr.datasets.dataloader_factory import make_dataloaders
from opr.testing import test
from opr.training import epoch_loop

EPOCHS = 60

IMAGE_LR = 0.0001
CLOUD_LR = 0.001
FUSION_LR = 0.001
TEXT_LR = 0.001
WEIGHT_DECAY = 0.0001


SCHEDULER_GAMMA = 0.1
SCHEDULER_STEPS = [40]

DEVICE = "cuda"
BATCH_EXPANSION_TH = 0.7
CHECKPOINTS_DIR = Path("checkpoints")


if __name__ == "__main__":
    # ==========> INITIALIZATION:

    model_config = OmegaConf.load("configs/models/text_model.yaml")
    model = instantiate(model_config)

    loss_cfg = OmegaConf.load("configs/losses/triplet_margin_text.yaml")
    loss_fn = instantiate(loss_cfg)

    dataset_cfg = OmegaConf.load("configs/datasets/phystech_campus_text.yaml")
    dataloaders = make_dataloaders(
        dataset_cfg=dataset_cfg.dataset,
        batch_sampler_cfg=dataset_cfg.sampler,
        num_workers=dataset_cfg.num_workers,
    )

    params_list = []
    if model.image_module is not None and IMAGE_LR is not None:
        params_list.append({"params": model.image_module.parameters(), "lr": IMAGE_LR})
    if model.cloud_module is not None and CLOUD_LR is not None:
        params_list.append({"params": model.cloud_module.parameters(), "lr": CLOUD_LR})
    if model.fusion_module is not None and FUSION_LR is not None:
        params_list.append({"params": model.fusion_module.parameters(), "lr": FUSION_LR})
    if model.text_module is not None and TEXT_LR is not None:
        params_list.append({"params": model.text_module.parameters(), "lr": IMAGE_LR})
    optimizer = Adam(params_list, weight_decay=WEIGHT_DECAY)
    scheduler = MultiStepLR(optimizer, milestones=SCHEDULER_STEPS, gamma=SCHEDULER_GAMMA)

    if not CHECKPOINTS_DIR.exists():
        CHECKPOINTS_DIR.mkdir(parents=True)

    model = model.to(DEVICE)

    # ==========> TRAIN LOOP:

    best_recall_at_1 = 0.0

    for epoch in range(EPOCHS):
        print(f"\n\n=====> Epoch {epoch+1}:")
        # TODO: resolve mypy typing here
        train_batch_size = dataloaders["train"].batch_sampler.batch_size  # type: ignore

        print("\n=> Training:\n")

        train_stats, train_rate_non_zero = epoch_loop(
            dataloader=dataloaders["train"],
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            phase="train",
            device=DEVICE,
        )

        print(f"\ntrain_rate_non_zero = {train_rate_non_zero}")

        if BATCH_EXPANSION_TH is not None:
            if BATCH_EXPANSION_TH == 1.0:
                print("Batch expansion rate is set to every epoch. Increasing batch size.")
                # TODO: resolve mypy typing here
                dataloaders["train"].batch_sampler.expand_batch()  # type: ignore
            elif train_rate_non_zero is None:
                print(
                    "\nWARNING: 'BATCH_EXPANSION_TH' was set, but 'train_rate_non_zero' is None. ",
                    "The batch size was not expanded.",
                )
            elif train_rate_non_zero < BATCH_EXPANSION_TH:
                print(
                    "Average non-zero triplet ratio is less than threshold: ",
                    f"{train_rate_non_zero} < {BATCH_EXPANSION_TH}",
                )
                # TODO: resolve mypy typing here
                dataloaders["train"].batch_sampler.expand_batch()  # type: ignore

        print("\n=> Validating:\n")

        val_stats, val_rate_non_zero = epoch_loop(
            dataloader=dataloaders["val"],
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            phase="val",
            device=DEVICE,
        )

        print(f"\nval_rate_non_zero = {val_rate_non_zero}")

        print("\n=> Testing:\n")

        recall_at_n, recall_at_one_percent, mean_top1_distance = test(
            model=model,
            descriptor_key="text",
            dataloader=dataloaders["test"],
            device=DEVICE,
        )

        stats_dict = {}
        stats_dict["test"] = {
            "mean_top1_distance": mean_top1_distance,
            "recall_at_1%": recall_at_one_percent,
            "recall_at_1": recall_at_n[0],
            "recall_at_3": recall_at_n[2],
            "recall_at_5": recall_at_n[4],
            "recall_at_10": recall_at_n[9],
        }
        stats_dict["train"] = train_stats
        stats_dict["train"]["batch_size"] = train_batch_size
        stats_dict["val"] = val_stats

        # saving checkpoints
        checkpoint_dict = {
            "epoch": epoch + 1,
            "stats_dict": stats_dict,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(checkpoint_dict, CHECKPOINTS_DIR / f"epoch_{epoch+1}.pth")
        if recall_at_n[0] > best_recall_at_1:
            print("Recall@1 improved!")
            torch.save(checkpoint_dict, CHECKPOINTS_DIR / "best.pth")
            best_recall_at_1 = recall_at_n[0]
