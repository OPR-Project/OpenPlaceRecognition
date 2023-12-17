"""GeoTransformer model for registration.

Paper: https://arxiv.org/abs/2202.06688

Code is adopted from original repository: https://github.com/qinzheng93/GeoTransformer, MIT License
"""
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor, nn

try:
    from geotransformer.modules.geotransformer import (
        GeometricTransformer,
        LocalGlobalRegistration,
        SuperPointMatching,
        SuperPointTargetGenerator,
    )
    from geotransformer.modules.kpconv import (
        ConvBlock,
        LastUnaryBlock,
        ResidualBlock,
        UnaryBlock,
        nearest_upsample,
    )
    from geotransformer.modules.ops import index_select, point_to_node_partition
    from geotransformer.modules.registration import get_node_correspondences
    from geotransformer.modules.sinkhorn import LearnableLogOptimalTransport
    from geotransformer.utils.data import (
        calibrate_neighbors_stack_mode,
        registration_collate_fn_stack_mode,
    )
    from geotransformer.utils.torch import to_cuda
except ImportError as err:
    raise ImportError(
        "To use the GeoTransformer model, please install the geotransformer package first."
    ) from err


class KPConvFPN(nn.Module):
    """Feature Pyramid Network with KPConv backbone."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        init_dim: int,
        kernel_size: int,
        init_radius: float,
        init_sigma: float,
        group_norm: int,
    ) -> None:
        """Feature Pyramid Network with KPConv backbone.

        Args:
            input_dim: The input feature dimension.
            output_dim: The output feature dimension.
            init_dim: The initial feature dimension.
            kernel_size: The kernel size of KPConv.
            init_radius: The initial radius of KPConv.
            init_sigma: The initial sigma of KPConv.
            group_norm: The number of groups in group normalization.
        """
        super().__init__()

        self.encoder1_1 = ConvBlock(input_dim, init_dim, kernel_size, init_radius, init_sigma, group_norm)
        self.encoder1_2 = ResidualBlock(
            init_dim, init_dim * 2, kernel_size, init_radius, init_sigma, group_norm
        )

        self.encoder2_1 = ResidualBlock(
            init_dim * 2, init_dim * 2, kernel_size, init_radius, init_sigma, group_norm, strided=True
        )
        self.encoder2_2 = ResidualBlock(
            init_dim * 2, init_dim * 4, kernel_size, init_radius * 2, init_sigma * 2, group_norm
        )
        self.encoder2_3 = ResidualBlock(
            init_dim * 4, init_dim * 4, kernel_size, init_radius * 2, init_sigma * 2, group_norm
        )

        self.encoder3_1 = ResidualBlock(
            init_dim * 4,
            init_dim * 4,
            kernel_size,
            init_radius * 2,
            init_sigma * 2,
            group_norm,
            strided=True,
        )
        self.encoder3_2 = ResidualBlock(
            init_dim * 4, init_dim * 8, kernel_size, init_radius * 4, init_sigma * 4, group_norm
        )
        self.encoder3_3 = ResidualBlock(
            init_dim * 8, init_dim * 8, kernel_size, init_radius * 4, init_sigma * 4, group_norm
        )

        self.encoder4_1 = ResidualBlock(
            init_dim * 8,
            init_dim * 8,
            kernel_size,
            init_radius * 4,
            init_sigma * 4,
            group_norm,
            strided=True,
        )
        self.encoder4_2 = ResidualBlock(
            init_dim * 8, init_dim * 16, kernel_size, init_radius * 8, init_sigma * 8, group_norm
        )
        self.encoder4_3 = ResidualBlock(
            init_dim * 16, init_dim * 16, kernel_size, init_radius * 8, init_sigma * 8, group_norm
        )

        self.encoder5_1 = ResidualBlock(
            init_dim * 16,
            init_dim * 16,
            kernel_size,
            init_radius * 8,
            init_sigma * 8,
            group_norm,
            strided=True,
        )
        self.encoder5_2 = ResidualBlock(
            init_dim * 16, init_dim * 32, kernel_size, init_radius * 16, init_sigma * 16, group_norm
        )
        self.encoder5_3 = ResidualBlock(
            init_dim * 32, init_dim * 32, kernel_size, init_radius * 16, init_sigma * 16, group_norm
        )

        self.decoder4 = UnaryBlock(init_dim * 48, init_dim * 16, group_norm)
        self.decoder3 = UnaryBlock(init_dim * 24, init_dim * 8, group_norm)
        self.decoder2 = LastUnaryBlock(init_dim * 12, output_dim)

    def forward(self, feats: Tensor, data_dict: Dict[str, List]) -> List[Tensor]:  # noqa: D102
        feats_list = []

        points_list = data_dict["points"]
        neighbors_list = data_dict["neighbors"]
        subsampling_list = data_dict["subsampling"]
        upsampling_list = data_dict["upsampling"]

        feats_s1 = feats
        feats_s1 = self.encoder1_1(feats_s1, points_list[0], points_list[0], neighbors_list[0])
        feats_s1 = self.encoder1_2(feats_s1, points_list[0], points_list[0], neighbors_list[0])

        feats_s2 = self.encoder2_1(feats_s1, points_list[1], points_list[0], subsampling_list[0])
        feats_s2 = self.encoder2_2(feats_s2, points_list[1], points_list[1], neighbors_list[1])
        feats_s2 = self.encoder2_3(feats_s2, points_list[1], points_list[1], neighbors_list[1])

        feats_s3 = self.encoder3_1(feats_s2, points_list[2], points_list[1], subsampling_list[1])
        feats_s3 = self.encoder3_2(feats_s3, points_list[2], points_list[2], neighbors_list[2])
        feats_s3 = self.encoder3_3(feats_s3, points_list[2], points_list[2], neighbors_list[2])

        feats_s4 = self.encoder4_1(feats_s3, points_list[3], points_list[2], subsampling_list[2])
        feats_s4 = self.encoder4_2(feats_s4, points_list[3], points_list[3], neighbors_list[3])
        feats_s4 = self.encoder4_3(feats_s4, points_list[3], points_list[3], neighbors_list[3])

        feats_s5 = self.encoder5_1(feats_s4, points_list[4], points_list[3], subsampling_list[3])
        feats_s5 = self.encoder5_2(feats_s5, points_list[4], points_list[4], neighbors_list[4])
        feats_s5 = self.encoder5_3(feats_s5, points_list[4], points_list[4], neighbors_list[4])

        latent_s5 = feats_s5
        feats_list.append(feats_s5)

        latent_s4 = nearest_upsample(latent_s5, upsampling_list[3])
        latent_s4 = torch.cat([latent_s4, feats_s4], dim=1)
        latent_s4 = self.decoder4(latent_s4)
        feats_list.append(latent_s4)

        latent_s3 = nearest_upsample(latent_s4, upsampling_list[2])
        latent_s3 = torch.cat([latent_s3, feats_s3], dim=1)
        latent_s3 = self.decoder3(latent_s3)
        feats_list.append(latent_s3)

        latent_s2 = nearest_upsample(latent_s3, upsampling_list[1])
        latent_s2 = torch.cat([latent_s2, feats_s2], dim=1)
        latent_s2 = self.decoder2(latent_s2)
        feats_list.append(latent_s2)

        feats_list.reverse()

        return feats_list


class GeoTransformer(nn.Module):
    """GeoTransformer model for registration."""

    def __init__(
        self,
        model: DictConfig,
        backbone: DictConfig,
        geotransformer: DictConfig,
        coarse_matching: DictConfig,
        fine_matching: DictConfig,
    ) -> None:
        """Geotransformer model for registration.

        Args:
            model: The model configuration.
            backbone: The backbone configuration.
            geotransformer: The geotransformer configuration.
            coarse_matching: The coarse matching configuration.
            fine_matching: The fine matching configuration.
        """
        super().__init__()
        self.num_points_in_patch = model.num_points_in_patch
        self.matching_radius = model.ground_truth_matching_radius

        backbone.init_radius = backbone.base_radius * backbone.init_voxel_size
        backbone.init_sigma = backbone.base_sigma * backbone.init_voxel_size

        self.backbone_cfg = backbone

        self.backbone = KPConvFPN(
            backbone.input_dim,
            backbone.output_dim,
            backbone.init_dim,
            backbone.kernel_size,
            backbone.init_radius,
            backbone.init_sigma,
            backbone.group_norm,
        )

        self.transformer = GeometricTransformer(
            geotransformer.input_dim,
            geotransformer.output_dim,
            geotransformer.hidden_dim,
            geotransformer.num_heads,
            geotransformer.blocks,
            geotransformer.sigma_d,
            geotransformer.sigma_a,
            geotransformer.angle_k,
            reduction_a=geotransformer.reduction_a,
        )

        self.coarse_target = SuperPointTargetGenerator(
            coarse_matching.num_targets, coarse_matching.overlap_threshold
        )

        self.coarse_matching = SuperPointMatching(
            coarse_matching.num_correspondences, coarse_matching.dual_normalization
        )

        self.fine_matching = LocalGlobalRegistration(
            fine_matching.topk,
            fine_matching.acceptance_radius,
            mutual=fine_matching.mutual,
            confidence_threshold=fine_matching.confidence_threshold,
            use_dustbin=fine_matching.use_dustbin,
            use_global_score=fine_matching.use_global_score,
            correspondence_threshold=fine_matching.correspondence_threshold,
            correspondence_limit=fine_matching.correspondence_limit,
            num_refinement_steps=fine_matching.num_refinement_steps,
        )

        self.optimal_transport = LearnableLogOptimalTransport(model.num_sinkhorn_iterations)

    @property
    def _is_cuda(self) -> bool:
        for param in self.parameters():
            if param.is_cuda:
                return True
        return False

    def _preprocess_input(
        self, query_pc: Tensor, db_pc: Tensor, gt_transform: Optional[Tensor] = None
    ) -> Dict[str, Any]:
        data_dict = {}
        data_dict["ref_points"] = db_pc
        data_dict["src_points"] = query_pc
        data_dict["ref_feats"] = torch.ones((query_pc.shape[0], 1), dtype=torch.float32)
        data_dict["src_feats"] = torch.ones((db_pc.shape[0], 1), dtype=torch.float32)
        if gt_transform:
            data_dict["transform"] = gt_transform
        else:
            data_dict["transform"] = torch.eye(4, dtype=torch.float32)
        neighbor_limits = calibrate_neighbors_stack_mode(
            [data_dict],
            registration_collate_fn_stack_mode,
            self.backbone_cfg.num_stages,
            self.backbone_cfg.init_voxel_size,
            self.backbone_cfg.init_radius,
        )
        data_dict = registration_collate_fn_stack_mode(
            [data_dict],
            self.backbone_cfg.num_stages,
            self.backbone_cfg.init_voxel_size,
            self.backbone_cfg.init_radius,
            neighbor_limits,
        )
        if self._is_cuda:
            data_dict = to_cuda(data_dict)
        return data_dict

    def forward(  # noqa: D102
        self, query_pc: Tensor, db_pc: Tensor, gt_transform: Optional[Tensor] = None
    ) -> Dict[str, Any]:
        data_dict = self._preprocess_input(query_pc, db_pc, gt_transform)
        output_dict = {}

        # Downsample point clouds
        feats = data_dict["features"].detach()
        transform = data_dict["transform"].detach()

        ref_length_c = data_dict["lengths"][-1][0].item()
        ref_length_f = data_dict["lengths"][1][0].item()
        # ref_length = data_dict["lengths"][0][0].item()
        points_c = data_dict["points"][-1].detach()
        points_f = data_dict["points"][1].detach()
        # points = data_dict["points"][0].detach()

        ref_points_c = points_c[:ref_length_c]
        src_points_c = points_c[ref_length_c:]
        ref_points_f = points_f[:ref_length_f]
        src_points_f = points_f[ref_length_f:]
        # ref_points = points[:ref_length]
        # src_points = points[ref_length:]

        # output_dict["ref_points_c"] = ref_points_c
        # output_dict["src_points_c"] = src_points_c
        # output_dict["ref_points_f"] = ref_points_f
        # output_dict["src_points_f"] = src_points_f
        # output_dict["ref_points"] = ref_points
        # output_dict["src_points"] = src_points

        # 1. Generate ground truth node correspondences
        _, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = point_to_node_partition(
            ref_points_f, ref_points_c, self.num_points_in_patch
        )
        _, src_node_masks, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(
            src_points_f, src_points_c, self.num_points_in_patch
        )

        ref_padded_points_f = torch.cat([ref_points_f, torch.zeros_like(ref_points_f[:1])], dim=0)
        src_padded_points_f = torch.cat([src_points_f, torch.zeros_like(src_points_f[:1])], dim=0)
        ref_node_knn_points = index_select(ref_padded_points_f, ref_node_knn_indices, dim=0)
        src_node_knn_points = index_select(src_padded_points_f, src_node_knn_indices, dim=0)

        gt_node_corr_indices, gt_node_corr_overlaps = get_node_correspondences(
            ref_points_c,
            src_points_c,
            ref_node_knn_points,
            src_node_knn_points,
            transform,
            self.matching_radius,
            ref_masks=ref_node_masks,
            src_masks=src_node_masks,
            ref_knn_masks=ref_node_knn_masks,
            src_knn_masks=src_node_knn_masks,
        )

        # output_dict["gt_node_corr_indices"] = gt_node_corr_indices
        # output_dict["gt_node_corr_overlaps"] = gt_node_corr_overlaps

        # 2. KPFCNN Encoder
        feats_list = self.backbone(feats, data_dict)

        feats_c = feats_list[-1]
        feats_f = feats_list[0]

        # 3. Conditional Transformer
        ref_feats_c = feats_c[:ref_length_c]
        src_feats_c = feats_c[ref_length_c:]
        ref_feats_c, src_feats_c = self.transformer(
            ref_points_c.unsqueeze(0),
            src_points_c.unsqueeze(0),
            ref_feats_c.unsqueeze(0),
            src_feats_c.unsqueeze(0),
        )
        ref_feats_c_norm = F.normalize(ref_feats_c.squeeze(0), p=2, dim=1)
        src_feats_c_norm = F.normalize(src_feats_c.squeeze(0), p=2, dim=1)

        # output_dict["ref_feats_c"] = ref_feats_c_norm
        # output_dict["src_feats_c"] = src_feats_c_norm

        # 5. Head for fine level matching
        ref_feats_f = feats_f[:ref_length_f]
        src_feats_f = feats_f[ref_length_f:]
        # output_dict["ref_feats_f"] = ref_feats_f
        # output_dict["src_feats_f"] = src_feats_f

        # 6. Select topk nearest node correspondences
        with torch.no_grad():
            ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_matching(
                ref_feats_c_norm, src_feats_c_norm, ref_node_masks, src_node_masks
            )

            # output_dict["ref_node_corr_indices"] = ref_node_corr_indices
            # output_dict["src_node_corr_indices"] = src_node_corr_indices

            # 7 Random select ground truth node correspondences during training
            if self.training:
                ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_target(
                    gt_node_corr_indices, gt_node_corr_overlaps
                )

        # 7.2 Generate batched node points & feats
        ref_node_corr_knn_indices = ref_node_knn_indices[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_indices = src_node_knn_indices[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_masks = ref_node_knn_masks[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_masks = src_node_knn_masks[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_points = ref_node_knn_points[ref_node_corr_indices]  # (P, K, 3)
        src_node_corr_knn_points = src_node_knn_points[src_node_corr_indices]  # (P, K, 3)

        ref_padded_feats_f = torch.cat([ref_feats_f, torch.zeros_like(ref_feats_f[:1])], dim=0)
        src_padded_feats_f = torch.cat([src_feats_f, torch.zeros_like(src_feats_f[:1])], dim=0)
        ref_node_corr_knn_feats = index_select(
            ref_padded_feats_f, ref_node_corr_knn_indices, dim=0
        )  # (P, K, C)
        src_node_corr_knn_feats = index_select(
            src_padded_feats_f, src_node_corr_knn_indices, dim=0
        )  # (P, K, C)

        # output_dict["ref_node_corr_knn_points"] = ref_node_corr_knn_points
        # output_dict["src_node_corr_knn_points"] = src_node_corr_knn_points
        # output_dict["ref_node_corr_knn_masks"] = ref_node_corr_knn_masks
        # output_dict["src_node_corr_knn_masks"] = src_node_corr_knn_masks

        # 8. Optimal transport
        matching_scores = torch.einsum(
            "bnd,bmd->bnm", ref_node_corr_knn_feats, src_node_corr_knn_feats
        )  # (P, K, K)
        matching_scores = matching_scores / feats_f.shape[1] ** 0.5
        matching_scores = self.optimal_transport(
            matching_scores, ref_node_corr_knn_masks, src_node_corr_knn_masks
        )

        # output_dict["matching_scores"] = matching_scores

        # 9. Generate final correspondences during testing
        with torch.no_grad():
            if not self.fine_matching.use_dustbin:
                matching_scores = matching_scores[:, :-1, :-1]

            ref_corr_points, src_corr_points, corr_scores, estimated_transform = self.fine_matching(
                ref_node_corr_knn_points,
                src_node_corr_knn_points,
                ref_node_corr_knn_masks,
                src_node_corr_knn_masks,
                matching_scores,
                node_corr_scores,
            )

            # output_dict["ref_corr_points"] = ref_corr_points
            # output_dict["src_corr_points"] = src_corr_points
            # output_dict["corr_scores"] = corr_scores
            output_dict["estimated_transform"] = estimated_transform

        return output_dict
