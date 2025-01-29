"""Adaptation of PointMamba model for Place Recognition."""
from time import time

import torch
from easydict import EasyDict
from loguru import logger
from torch import Tensor

from opr.modules import SeqGeM

try:
    from pointmamba.models import PointMamba

    pointmamba_available = True
except ImportError:
    logger.warning("The 'pointmamba' package is not installed. Please install it manually if neccessary.")
    pointmamba_available = False
    PointMamba = object


class PointMambaPR(PointMamba):
    """Adaptation of PointMamba model for Place Recognition.

    Paper: https://arxiv.org/abs/2402.10739
    Adapted from original repository: https://github.com/LMD0311/PointMamba
    License: Apache-2.0
    """

    def __init__(
        self,
        pooling: str = "gem",
        normalize_output: bool = False,
        trans_dim: int = 256,
        depth: int = 12,
        group_size: int = 32,
        num_group: int = 512,
        encoder_dims: int = 256,
        drop_out: float = 0.0,
        drop_path: float = 0.1,
        rms_norm: bool = False,
        drop_out_in_block: float = 0.0,
    ) -> None:
        """Initialize the PointMamba model for Place Recognition.

        Args:
            pooling (str): Pooling method to use. Defaults to "gem".
            normalize_output (bool): Whether to normalize the output. Defaults to False.
            trans_dim (int): Dimension of the transformer (argument from original PointMamba). Defaults to 256.
            depth (int): Depth of the transformer (argument from original PointMamba). Defaults to 12.
            group_size (int): Size of the points group (argument from original PointMamba). Defaults to 32.
            num_group (int): Number of groups (argument from original PointMamba). Defaults to 512.
            encoder_dims (int): Dimension of the encoder (argument from original PointMamba). Defaults to 256.
            drop_out (float): Dropout rate (argument from original PointMamba). Defaults to 0.0.
            drop_path (float): Drop path rate (argument from original PointMamba). Defaults to 0.1.
            rms_norm (bool): Whether to use RMS normalization (argument from original PointMamba). Defaults to False.
            drop_out_in_block (float): Dropout rate in the block (argument from original PointMamba). Defaults to 0.0.

        Raises:
            ImportError: If the 'pointmamba' package is not installed.
            NotImplementedError: If an unknown pooling method is provided.
        """
        if not pointmamba_available:
            raise ImportError("The 'pointmamba' package is not installed. Please install it manually.")

        self.config = EasyDict(
            trans_dim=trans_dim,
            depth=depth,
            group_size=group_size,
            num_group=num_group,
            encoder_dims=encoder_dims,
            drop_out=drop_out,
            drop_path=drop_path,
            rms_norm=rms_norm,
            drop_out_in_block=drop_out_in_block,
            cls_dim=1,
        )
        super().__init__(self.config)

        if hasattr(self, "cls_head_finetune"):
            delattr(self, "cls_head_finetune")

        if hasattr(self, "loss_ce"):
            delattr(self, "loss_ce")

        if pooling == "gem":
            self.head = SeqGeM()
        else:
            raise NotImplementedError("Unknown pooling method: {}".format(pooling))

        self.normalize_output = normalize_output

        self.stats = {
            "total": [],
            "group_divider": [],
            "encoder": [],
            "pos_embed": [],
            "transformer": [],
            "reordering": [],
        }

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:  # noqa: D102
        total_t_s = time()

        pts = batch["pointclouds_lidar_coords"]

        t_s = time()
        neighborhood, center = self.group_divider(pts)
        torch.cuda.current_stream().synchronize()
        self.stats["group_divider"].append(time() - t_s)

        t_s = time()
        group_input_tokens = self.encoder(neighborhood)  # B G N
        torch.cuda.current_stream().synchronize()
        self.stats["encoder"].append(time() - t_s)

        t_s = time()
        pos = self.pos_embed(center)
        torch.cuda.current_stream().synchronize()
        self.stats["pos_embed"].append(time() - t_s)

        t_s = time()
        # reordering strategy
        center_x = center[:, :, 0].argsort(dim=-1)[:, :, None]
        center_y = center[:, :, 1].argsort(dim=-1)[:, :, None]
        center_z = center[:, :, 2].argsort(dim=-1)[:, :, None]
        group_input_tokens_x = group_input_tokens.gather(
            dim=1, index=torch.tile(center_x, (1, 1, group_input_tokens.shape[-1]))
        )
        group_input_tokens_y = group_input_tokens.gather(
            dim=1, index=torch.tile(center_y, (1, 1, group_input_tokens.shape[-1]))
        )
        group_input_tokens_z = group_input_tokens.gather(
            dim=1, index=torch.tile(center_z, (1, 1, group_input_tokens.shape[-1]))
        )
        pos_x = pos.gather(dim=1, index=torch.tile(center_x, (1, 1, pos.shape[-1])))
        pos_y = pos.gather(dim=1, index=torch.tile(center_y, (1, 1, pos.shape[-1])))
        pos_z = pos.gather(dim=1, index=torch.tile(center_z, (1, 1, pos.shape[-1])))
        group_input_tokens = torch.cat(
            [group_input_tokens_x, group_input_tokens_y, group_input_tokens_z], dim=1
        )
        pos = torch.cat([pos_x, pos_y, pos_z], dim=1)

        x = group_input_tokens
        torch.cuda.current_stream().synchronize()
        self.stats["reordering"].append(time() - t_s)

        t_s = time()
        # transformer
        x = self.drop_out(x)
        x = self.blocks(x, pos)
        x = self.norm(x)
        torch.cuda.current_stream().synchronize()
        self.stats["transformer"].append(time() - t_s)

        x = self.head(x.permute(0, 2, 1))
        if self.normalize_output:
            x = torch.nn.functional.normalize(x, dim=-1)
        out_dict: dict[str, Tensor] = {"final_descriptor": x}

        self.stats["total"].append(time() - total_t_s)

        return out_dict
