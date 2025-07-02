import argparse
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor, nn


class BoQModel(nn.Module):
    def __init__(self, backbone_name: Literal["dinov2", "resnet50"] = "dinov2"):
        super().__init__()

        out_dims = {
            "dinov2": 12288,
            "resnet50": 16384,
        }

        self.model = torch.hub.load(
            repo_or_dir="amaralibey/bag-of-queries",
            model="get_trained_boq",
            backbone_name=backbone_name,
            output_dim=out_dims[backbone_name],
        )

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Forward pass of the BoQ model.

        Args:
            batch (dict[str, Tensor]): Input batch containing images and other data.

        Returns:
            dict[str, Tensor]: Output dictionary containing the final descriptor.
        """
        key = next((k for k in batch if k.startswith("images_")), None)
        if key is None:
            raise KeyError("No key starting with 'images_' found in the batch.")
        images = batch[key]
        descriptor, _ = self.model(images)
        output = {"final_descriptor": descriptor}
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone_name", type=str, default="dinov2", choices=["dinov2", "resnet50"], help="Backbone name"
    )
    parser.add_argument("--save_checkpoint", type=Path, default=None, help="Path to save the checkpoint")

    args = parser.parse_args()

    model = BoQModel(backbone_name=args.backbone_name)
    if args.save_checkpoint:
        torch.save(model.state_dict(), args.save_checkpoint)
        print(f"Model checkpoint saved to {args.save_checkpoint}")
    else:
        print("No checkpoint path provided. Model not saved.")
