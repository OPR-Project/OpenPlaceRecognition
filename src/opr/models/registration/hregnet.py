"""Implementation of HRegNet point cloud registration model."""
from argparse import Namespace
from typing import Any, Dict

import torch
from hregnet.models import HRegNet as _HRegNet
from torch import Tensor


class HRegNet(_HRegNet):
    """HRegNet: A Hierarchical Network for Large-scale Outdoor LiDAR Point Cloud Registration.

    Paper: https://arxiv.org/abs/2107.11992
    Code is adopted from the original repository: https://github.com/ispc-lab/HRegNet, MIT License
    """

    def __init__(
        self,
        num_reg_steps: int = 3,
        use_fps: bool = True,
        use_weights: bool = True,
        freeze_detector: bool = False,
        freeze_feats: bool = False,
    ) -> None:
        """HRegNet: A Hierarchical Network for Large-scale Outdoor LiDAR Point Cloud Registration.

        Paper: https://arxiv.org/abs/2107.11992
        Code is adopted from the original repository: https://github.com/ispc-lab/HRegNet, MIT License

        Args:
            num_reg_steps (int): Number of registration steps. Must be in [1, 2, 3]. Less steps are faster
                but less accurate. Defaults to 3.
            use_fps (bool): Whether to use farthest point sampling (FPS) for keypoint detection. Defaults to True.
            use_weights (bool): Whether to use weights for keypoint detection. Defaults to True.
            freeze_detector (bool): Whether to freeze the detector. Defaults to False.
            freeze_feats (bool): Whether to freeze the features. Defaults to False.
        """
        args = Namespace(
            use_fps=use_fps,
            use_weights=use_weights,
            freeze_detector=freeze_detector,
            freeze_feats=freeze_feats,
        )
        super().__init__(args, num_reg_steps)

    def forward(self, query_pc: Tensor, db_pc: Tensor) -> Dict[str, Any]:  # noqa: D102
        if query_pc.dim() == 2:
            query_pc = query_pc.unsqueeze(0)
        if db_pc.dim() == 2:
            db_pc = db_pc.unsqueeze(0)
        _out_dict = super().forward(src_points=db_pc, dst_points=query_pc)
        rotation_matrix = _out_dict["rotation"][-1].detach()
        translation = _out_dict["translation"][-1].detach()
        transform = torch.eye(4, device=query_pc.device)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = translation
        output_dict = {}
        # estimated transform should be a 4x4 matrix
        output_dict["estimated_transform"] = transform
        return output_dict
