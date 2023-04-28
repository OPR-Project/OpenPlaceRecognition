"""Sample training script: trains MinkLoc++ model on Oxford RobotCar dataset."""
from pathlib import Path

import torch
import hydra
from hydra.utils import instantiate
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from opr.datasets.dataloader_factory import make_dataloaders
from opr.testing import test
from opr.training import epoch_loop

from omegaconf import DictConfig


@hydra.main(config_path="configs", config_name="config")
def train(cfg: DictConfig):
    model = instantiate(cfg.model)

    loss_fn = instantiate(cfg.loss)

    dataloaders = make_dataloaders(
        dataset_cfg=cfg.dataset.dataset,
        batch_sampler_cfg=cfg.dataset.sampler,
        num_workers=cfg.dataset.num_workers,
    )

    params_list = []
    for modality in cfg.general.modalities:
        params_list.append(
            {
                "params": getattr(model, f"{modality}_module").parameters(),
                "lr": cfg.optimizer[f"{modality}_lr"],
            }
        )
    optimizer = Adam(params_list, weight_decay=cfg.optimizer.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=cfg.scheduler.steps, gamma=cfg.scheduler.gamma)

    checkpoints_dir = Path(cfg.general.checkpoints_dir)
    if not checkpoints_dir.exists():
        checkpoints_dir.mkdir(parents=True)

    model = model.to(cfg.general.device)

    # ==========> TRAIN LOOP:

    best_recall_at_1 = 0.0

    for epoch in range(cfg.general.epochs):
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
            device=cfg.general.device,
        )

        print(f"\ntrain_rate_non_zero = {train_rate_non_zero}")

        batch_expansion_th = cfg.general.batch_expansion_th
        if batch_expansion_th is not None:
            if batch_expansion_th == 1.0:
                print("Batch expansion rate is set to every epoch. Increasing batch size.")
                # TODO: resolve mypy typing here
                dataloaders["train"].batch_sampler.expand_batch()  # type: ignore
            elif train_rate_non_zero is None:
                print(
                    "\nWARNING: 'batch_expansion_th' was set, but 'train_rate_non_zero' is None. ",
                    "The batch size was not expanded.",
                )
            elif train_rate_non_zero < batch_expansion_th:
                print(
                    "Average non-zero triplet ratio is less than threshold: ",
                    f"{train_rate_non_zero} < {batch_expansion_th}",
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
            device=cfg.general.device,
        )

        print(f"\nval_rate_non_zero = {val_rate_non_zero}")

        print("\n=> Testing:\n")

        recall_at_n, recall_at_one_percent, mean_top1_distance = test(
            model=model,
            descriptor_key="fusion",
            dataloader=dataloaders["test"],
            device=cfg.general.device,
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
        torch.save(checkpoint_dict, checkpoints_dir / f"epoch_{epoch+1}.pth")
        if recall_at_n[0] > best_recall_at_1:
            print("Recall@1 improved!")
            torch.save(checkpoint_dict, checkpoints_dir / "best.pth")
            best_recall_at_1 = recall_at_n[0]


if __name__ == "__main__":
    train()
