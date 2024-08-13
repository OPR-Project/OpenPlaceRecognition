"""Implementation of OverlapTransformer model."""
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing import Dict

from opr.modules import NetVLAD


class OverlapTransformer(nn.Module):
    """OverlapTransformer: An Efficient and Yaw-Angle-Invariant Transformer Network for LiDAR-Based Place Recognition.

    Paper: https://arxiv.org/abs/2203.03397
    Adapted from original repository: https://github.com/haomo-ai/OverlapTransformer
    """

    def __init__(
        self,
        height: int = 64,
        width: int = 900,
        channels: int = 1,
        norm_layer: nn.Module = None,
        use_transformer: bool = True,
    ) -> None:
        """Initialize the OverlapTransformer model.

        Args:
            height (int): Height of the input tensor. Defaults to 64.
            width (int): Width of the input tensor. Defaults to 900.
            channels (int): Number of channels in the input tensor. Defaults to 1.
            norm_layer (nn.Module): Normalization layer to use. Defaults to None.
            use_transformer (bool): Whether to use the transformer encoder. Defaults to True.
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.use_transformer = use_transformer

        self.conv1 = nn.Conv2d(channels, 16, kernel_size=(5, 1), stride=(1, 1), bias=False)
        self.bn1 = norm_layer(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 1), stride=(2, 1), bias=False)
        self.bn2 = norm_layer(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 1), stride=(2, 1), bias=False)
        self.bn3 = norm_layer(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 1), stride=(2, 1), bias=False)
        self.bn4 = norm_layer(64)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(2, 1), stride=(2, 1), bias=False)
        self.bn5 = norm_layer(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(2, 1), bias=False)
        self.bn6 = norm_layer(128)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(2, 1), bias=False)
        self.bn7 = norm_layer(128)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(2, 1), bias=False)
        self.bn8 = norm_layer(128)
        self.conv9 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(2, 1), bias=False)
        self.bn9 = norm_layer(128)
        self.conv10 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(2, 1), bias=False)
        self.bn10 = norm_layer(128)
        self.conv11 = nn.Conv2d(128, 128, kernel_size=(1, 1), stride=(2, 1), bias=False)
        self.bn11 = norm_layer(128)
        self.relu = nn.ReLU(inplace=True)

        """
            MHSA
            num_layers=1 is suggested in our work.
        """
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=4, dim_feedforward=1024, activation="relu", batch_first=False, dropout=0.0
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.convLast1 = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bnLast1 = norm_layer(256)
        self.convLast2 = nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bnLast2 = norm_layer(1024)

        self.linear = nn.Linear(128 * 900, 256)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

        self.net_vlad = NetVLAD(num_clusters=64, dim=1024)  # TODO: implement with 'NetVLADLoupe'?

        self.linear1 = nn.Linear(1 * 256, 256)
        self.bnl1 = norm_layer(256)
        self.linear2 = nn.Linear(1 * 256, 256)
        self.bnl2 = norm_layer(256)
        self.linear3 = nn.Linear(1 * 256, 256)
        self.bnl3 = norm_layer(256)

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:  # noqa: D102
        for key, value in batch.items():
            if key.startswith("range_image"):
                x_l = value
                out_l = self.relu(self.conv1(x_l))
                out_l = self.relu(self.conv2(out_l))
                out_l = self.relu(self.conv3(out_l))
                out_l = self.relu(self.conv4(out_l))
                out_l = self.relu(self.conv5(out_l))
                out_l = self.relu(self.conv6(out_l))
                out_l = self.relu(self.conv7(out_l))
                out_l = self.relu(self.conv8(out_l))
                out_l = self.relu(self.conv9(out_l))
                out_l = self.relu(self.conv10(out_l))
                out_l = self.relu(self.conv11(out_l))
                out_l_1 = out_l.permute(0, 1, 3, 2)
                out_l_1 = self.relu(self.convLast1(out_l_1))
                # Using transformer needs to decide whether batch_size first
                if self.use_transformer:
                    out_l = out_l_1.squeeze(3)
                    out_l = out_l.permute(2, 0, 1)
                    out_l = self.transformer_encoder(out_l)
                    out_l = out_l.permute(1, 2, 0)
                    out_l = out_l.unsqueeze(3)
                    out_l = torch.cat((out_l_1, out_l), dim=1)
                    out_l = self.relu(self.convLast2(out_l))
                    out_l = F.normalize(out_l, dim=1)
                    out_l = self.net_vlad(out_l)
                    out_l = F.normalize(out_l, dim=1)
                else:
                    out_l = torch.cat((out_l_1, out_l_1), dim=1)
                    out_l = F.normalize(out_l, dim=1)
                    out_l = self.net_vlad(out_l)
                    out_l = F.normalize(out_l, dim=1)
        out_dict: Dict[str, Tensor] = {"final_descriptor": out_l}
        return out_dict
