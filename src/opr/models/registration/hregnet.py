"""Implementation of HRegNet point cloud registration model."""
from argparse import Namespace
from time import time
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
        use_sim: bool = True,
        use_neighbor: bool = True,
        use_fps: bool = True,
        use_weights: bool = True,
        light_feats: bool = False,
        freeze_detector: bool = False,
        freeze_feats: bool = False,
    ) -> None:
        """HRegNet: A Hierarchical Network for Large-scale Outdoor LiDAR Point Cloud Registration.

        Paper: https://arxiv.org/abs/2107.11992
        Code is adopted from the original repository: https://github.com/ispc-lab/HRegNet, MIT License

        Args:
            num_reg_steps (int): Number of registration steps. Must be in [1, 2, 3]. Less steps are faster
                but less accurate. Defaults to 3.
            use_sim (bool): Whether to use original similarity features. Defaults to True.
            use_neighbor (bool): Whether to use neighbor-aware similarity featuress. Defaults to True.
            use_fps (bool): Whether to use farthest point sampling (FPS) for keypoint detection. Defaults to True.
            use_weights (bool): Whether to use weights for keypoint detection. Defaults to True.
            light_feats (bool): Whether to use light feature extractor. Defaults to False.
            freeze_detector (bool): Whether to freeze the detector. Defaults to False.
            freeze_feats (bool): Whether to freeze the features. Defaults to False.
        """
        args = Namespace(
            use_fps=use_fps,
            use_weights=use_weights,
            freeze_detector=freeze_detector,
            freeze_feats=freeze_feats,
        )
        super().__init__(
            args=args,
            num_reg_steps=num_reg_steps,
            use_sim=use_sim,
            use_neighbor=use_neighbor,
            model_version=("light" if light_feats else "original"),
        )

    def forward(  # noqa: D102
        self, query_pc: Tensor, db_pc: Tensor | None = None, db_pc_feats: Dict[str, Tensor] | None = None
    ) -> Dict[str, Any]:
        if db_pc is None and db_pc_feats is None:
            raise ValueError("Either db_pc or db_pc_feats must be provided.")
        if db_pc is not None and db_pc_feats is not None:
            raise ValueError("Only one of db_pc or db_pc_feats must be provided.")

        if query_pc.dim() == 2:
            query_pc = query_pc.unsqueeze(0)
        if db_pc is not None and db_pc.dim() == 2:
            db_pc = db_pc.unsqueeze(0)

        _out_dict = self._forward(src_points=db_pc, src_feats=db_pc_feats, dst_points=query_pc)
        rotation_matrix = _out_dict["rotation"][-1].detach()
        translation = _out_dict["translation"][-1].detach()
        transform = torch.eye(4, device=query_pc.device)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = translation
        output_dict = {}
        # estimated transform should be a 4x4 matrix
        output_dict["estimated_transform"] = transform
        return output_dict

    def extract_features(self, pc: Tensor) -> Dict[str, Tensor]:  # noqa: D102
        if pc.dim() == 2:
            pc = pc.unsqueeze(0)
        return self.feature_extraction(pc)

    def _forward(
        self, dst_points: Tensor, src_points: Tensor | None = None, src_feats: Dict[str, Tensor] | None = None
    ) -> Dict[str, Any]:
        if src_points is None and src_feats is None:
            raise ValueError("Either src_points or src_feats must be provided.")
        if src_points is not None and src_feats is not None:
            raise ValueError("Only one of src_points or src_feats must be provided.")
        # Feature extraction
        t_s = time()
        if src_feats is None:
            src_feats = self.feature_extraction(src_points)
        self.stats_history["src_feats_time"].append(time() - t_s)
        t_s = time()
        dst_feats = self.feature_extraction(dst_points)
        self.stats_history["dst_feats_time"].append(time() - t_s)

        # Coarse registration
        t_s = time()
        src_xyz_corres_3, src_dst_weights_3 = self.coarse_corres(
            src_feats["xyz_3"],
            src_feats["desc_3"],
            dst_feats["xyz_3"],
            dst_feats["desc_3"],
            src_feats["sigmas_3"],
            dst_feats["sigmas_3"],
        )

        R3, t3 = self.svd_head(src_feats["xyz_3"], src_xyz_corres_3, src_dst_weights_3)

        corres_dict = {}
        corres_dict["src_xyz_corres_3"] = src_xyz_corres_3
        corres_dict["src_dst_weights_3"] = src_dst_weights_3

        ret_dict = {}
        ret_dict["rotation"] = [R3]
        ret_dict["translation"] = [t3]
        ret_dict["src_feats"] = src_feats
        ret_dict["dst_feats"] = dst_feats

        self.stats_history["coarse_reg_time"].append(time() - t_s)

        if self.num_reg_steps == 1:
            return ret_dict

        # Fine registration: Layer 2
        t_s = time()
        src_xyz_2_trans = torch.matmul(R3, src_feats["xyz_2"].permute(0, 2, 1).contiguous()) + t3.unsqueeze(2)
        src_xyz_2_trans = src_xyz_2_trans.permute(0, 2, 1).contiguous()
        src_xyz_corres_2, src_dst_weights_2 = self.fine_corres_2(
            src_xyz_2_trans,
            src_feats["desc_2"],
            dst_feats["xyz_2"],
            dst_feats["desc_2"],
            src_feats["sigmas_2"],
            dst_feats["sigmas_2"],
        )
        R2_, t2_ = self.svd_head(src_xyz_2_trans, src_xyz_corres_2, src_dst_weights_2)
        T3 = torch.zeros(R3.shape[0], 4, 4).cuda()
        T3[:, :3, :3] = R3
        T3[:, :3, 3] = t3
        T3[:, 3, 3] = 1.0
        T2_ = torch.zeros(R2_.shape[0], 4, 4).cuda()
        T2_[:, :3, :3] = R2_
        T2_[:, :3, 3] = t2_
        T2_[:, 3, 3] = 1.0
        T2 = torch.matmul(T2_, T3)
        R2 = T2[:, :3, :3]
        t2 = T2[:, :3, 3]

        corres_dict["src_xyz_corres_2"] = src_xyz_corres_2
        corres_dict["src_dst_weights_2"] = src_dst_weights_2

        ret_dict["rotation"].append(R2)
        ret_dict["translation"].append(t2)

        self.stats_history["fine_reg_2_time"].append(time() - t_s)

        if self.num_reg_steps == 2:
            return ret_dict

        # Fine registration: Layer 1
        t_s = time()
        src_xyz_1_trans = torch.matmul(R2, src_feats["xyz_1"].permute(0, 2, 1).contiguous()) + t2.unsqueeze(2)
        src_xyz_1_trans = src_xyz_1_trans.permute(0, 2, 1).contiguous()
        src_xyz_corres_1, src_dst_weights_1 = self.fine_corres_1(
            src_xyz_1_trans,
            src_feats["desc_1"],
            dst_feats["xyz_1"],
            dst_feats["desc_1"],
            src_feats["sigmas_1"],
            dst_feats["sigmas_1"],
        )
        R1_, t1_ = self.svd_head(src_xyz_1_trans, src_xyz_corres_1, src_dst_weights_1)
        T1_ = torch.zeros(R1_.shape[0], 4, 4).cuda()
        T1_[:, :3, :3] = R1_
        T1_[:, :3, 3] = t1_
        T1_[:, 3, 3] = 1.0

        T1 = torch.matmul(T1_, T2)
        R1 = T1[:, :3, :3]
        t1 = T1[:, :3, 3]

        corres_dict["src_xyz_corres_1"] = src_xyz_corres_1
        corres_dict["src_dst_weights_1"] = src_dst_weights_1

        ret_dict["rotation"].append(R1)
        ret_dict["translation"].append(t1)

        self.stats_history["fine_reg_1_time"].append(time() - t_s)

        return ret_dict
