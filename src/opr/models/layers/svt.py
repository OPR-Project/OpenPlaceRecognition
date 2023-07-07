from typing import Tuple

import torch
from torch import nn
import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck

from opr.models.layers.eca import MinkECABasicBlock
from opr.models.minkloc import MinkResNetBase


class ASVT(nn.Module):
    """ASVT - Atom-Based Sparse Voxel Transformer."""

    def __init__(self, in_dim: int, reduction: int = 8) -> None:
        super().__init__()
        self.q_conv = ME.MinkowskiConvolution(in_dim, in_dim // reduction, 1, dimension=3, bias=False)
        self.k_conv = ME.MinkowskiConvolution(in_dim, in_dim // reduction, 1, dimension=3, bias=False)

        self.v_conv = ME.MinkowskiConvolution(in_dim, in_dim, 1, dimension=3, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        self.trans_conv = ME.MinkowskiConvolution(in_dim, in_dim, 1, dimension=3, bias=False)
        self.after_norm = ME.MinkowskiBatchNorm(in_dim)
        self.act = ME.MinkowskiReLU(inplace=True)

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        x_q = self.q_conv(x)
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)

        bath_size = torch.max(x.C[:, 0], 0)[0] + 1
        start_id = 0
        x_feat = list()
        for i in range(bath_size):
            end_id = start_id + torch.sum(x.C[:, 0] == i)
            dq = x_q.F[start_id:end_id, :]  # N*C
            dk = x_k.F[start_id:end_id, :].T  # C*N
            dv = x_v.F[start_id:end_id, :]  # N*C
            de = torch.matmul(dq, dk)  # N*N
            da = self.softmax(de)  # N*N
            # da = da / (1e-9 + da.sum(dim=1, keepdim=True))
            dr = torch.matmul(da, dv)  # N*C
            x_feat.append(dr)
            start_id = end_id
        x_r = torch.cat(x_feat, dim=0)
        x_r = ME.SparseTensor(
            # coordinates=x.coordinates,
            features=x_r,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )

        x_r = x + self.act(self.after_norm(self.trans_conv(x_r)))
        return x_r


class CSVT(nn.Module):
    """CSVT - Cluster-Based Sparse Voxel Transformer."""

    def __init__(self, channels, num_tokens=16):
        super().__init__()

        # layers for generate tokens
        self.q_conv = ME.MinkowskiConvolution(channels, channels, 1, dimension=3, bias=False)
        self.k_conv = ME.MinkowskiConvolution(channels, num_tokens, 1, dimension=3, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        # layers for tranformer
        self.convvalues = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.convkeys = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.convquries = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.embedding1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        # layers for projector
        self.p_conv = ME.MinkowskiConvolution(channels, channels, 1, dimension=3, bias=False)
        self.T_conv = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        # hidden state
        self.trans_conv = ME.MinkowskiConvolution(channels, channels, 1, dimension=3, bias=False)
        self.after_norm = ME.MinkowskiBatchNorm(channels)
        self.act = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        # generate tokens
        x_q = self.q_conv(x)
        x_k = self.k_conv(x)

        bath_size = torch.max(x.C[:, 0], 0)[0] + 1
        start_id = 0
        x_feat = list()
        for i in range(bath_size):
            end_id = start_id + torch.sum(x.C[:, 0] == i)
            dq = x_q.F[start_id:end_id, :]  # N*C
            dk = x_k.F[start_id:end_id, :].T  # num_tokens*N
            dk = self.softmax(dk)  # N*num_tokens

            de = torch.matmul(dk, dq).T  # C*num_tokens
            # da = da / (1e-9 + da.sum(dim=1, keepdim=True))
            de = torch.unsqueeze(de, dim=0)
            x_feat.append(de)
            start_id = end_id
        tokens = torch.cat(x_feat, dim=0)  # B*C*num_tokens

        # visul transormers on multi tokens
        vt_values = self.convvalues(tokens)
        vt_keys = self.convkeys(tokens)  # B*C*num_tokens
        vt_quires = self.convquries(tokens)  # B*C*num_tokens
        vt_map = torch.matmul(vt_keys.transpose(1, 2), vt_quires)  # B*num_tokens*num_tokens
        vt_map = self.softmax(vt_map)  # B*num_tokens*num_tokens
        T_middle = torch.matmul(vt_map, vt_values.transpose(1, 2)).transpose(1, 2)  # B*C*num_tokens
        # T_out = tokens + self.actembedding1(self.bnembedding1(self.embedding1(T_middle)))                    # B*C*num_tokens
        T_out = tokens + self.embedding1(T_middle)

        # projector
        x_p = self.p_conv(x)
        T_P = self.T_conv(T_out)

        start_id = 0
        x_feat2 = list()
        for i in range(bath_size):
            end_id = start_id + torch.sum(x.C[:, 0] == i)
            dp = x_p.F[start_id:end_id, :]  # N*C
            dt = T_P[i]  # C*num_tokens

            dm = torch.matmul(dp, dt)  # N*num_tokens
            dm = self.softmax(dm)  # N*num_tokens

            df = torch.matmul(dm, dt.T)  # N*C
            x_feat2.append(df)
            start_id = end_id
        x_r = torch.cat(x_feat2, dim=0)

        x_r = ME.SparseTensor(
            # coordinates=x.coordinates,
            features=x_r,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )
        x_r = x + self.act(self.after_norm(self.trans_conv(x_r)))
        return x_r


# TODO: move the code to parent dir
class SVTNet(MinkResNetBase):
    sparse = True

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 256,
        conv0_kernel_size: int = 5,
        block: str = "ECABasicBlock",
        asvt: bool = True,
        csvt: bool = True,
        layers: Tuple[int, ...] = (1, 1, 1),
        planes: Tuple[int, ...] = (32, 64, 64),
    ) -> None:
        assert len(layers) == len(planes)
        assert 1 <= len(layers)
        self.num_bottom_up = len(layers)
        self.conv0_kernel_size = conv0_kernel_size
        self.block = self._create_resnet_block(block_name=block)
        self.layers = layers
        self.planes = planes
        self.lateral_dim = out_channels
        self.init_dim = planes[0]
        self.is_asvt = asvt
        self.is_csvt = csvt
        MinkResNetBase.__init__(self, in_channels, out_channels, dimension=3)

    def _create_resnet_block(self, block_name: str) -> nn.Module:
        if block_name == "BasicBlock":
            block_module = BasicBlock
        elif block_name == "Bottleneck":
            block_module = Bottleneck
        elif block_name == "ECABasicBlock":
            block_module = MinkECABasicBlock
        else:
            raise NotImplementedError(f"Unsupported network block: {block_name}")

        return block_module

    def _network_initialization(self, in_channels, out_channels, D):
        assert len(self.layers) == len(self.planes)
        assert len(self.planes) == self.num_bottom_up

        self.convs = nn.ModuleList()  # Bottom-up convolutional blocks with stride=2
        self.bn = nn.ModuleList()  # Bottom-up BatchNorms
        self.blocks = nn.ModuleList()  # Bottom-up blocks
        self.tconvs = nn.ModuleList()  # Top-down tranposed convolutions
        self.conv1x1 = nn.ModuleList()  # 1x1 convolutions in lateral connections

        # The first convolution is special case, with kernel size = 5
        self.inplanes = self.planes[0]
        self.conv0 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=self.conv0_kernel_size, dimension=D
        )
        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        for plane, layer in zip(self.planes, self.layers):
            self.convs.append(
                ME.MinkowskiConvolution(self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
            )
            self.bn.append(ME.MinkowskiBatchNorm(self.inplanes))
            self.blocks.append(self._make_layer(self.block, plane, layer))

        self.conv1x1.append(
            ME.MinkowskiConvolution(self.inplanes, self.lateral_dim, kernel_size=1, stride=1, dimension=D)
        )

        # before_lateral_dim=plane
        after_reduction = max(self.lateral_dim / 8, 8)
        reduction = int(self.lateral_dim // after_reduction)

        if self.is_asvt:
            self.asvt = ASVT(self.lateral_dim, reduction)
        if self.is_csvt:
            self.csvt = CSVT(self.lateral_dim, 8)

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        # *** BOTTOM-UP PASS ***
        # First bottom-up convolution is special (with bigger stride)
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)

        # BOTTOM-UP PASS
        for ndx, (conv, bn, block) in enumerate(zip(self.convs, self.bn, self.blocks)):
            x = conv(x)  # Decreases spatial resolution (conv stride=2)
            x = bn(x)
            x = self.relu(x)
            x = block(x)

        x = self.conv1x1[0](x)

        if self.is_csvt:
            x_csvt = self.csvt(x)
        if self.is_asvt:
            x_asvt = self.asvt(x)

        if self.is_csvt and self.is_asvt:
            x = x_csvt + x_asvt
        elif self.is_csvt:
            x = x_csvt
        elif self.is_asvt:
            x = x_asvt

        return x
