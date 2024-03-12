"""MixVPR: Feature Mixing for Visual Place Recognition.

Source: https://github.com/amaralibey/MixVPR/blob/main/models/aggregators/mixvpr.py
"""
from torch import Tensor, nn
from torch.nn import functional as F


class FeatureMixerLayer(nn.Module):
    """Feature Mixer Layer."""

    def __init__(self, in_dim: int, mlp_ratio: float = 1.0) -> None:
        """Feature Mixer Layer.

        Args:
            in_dim (int): Input dimension.
            mlp_ratio (float): Ratio of the mid projection layer in the mixer block. Defaults to 1.0.s
        """
        super().__init__()
        self.mix = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(in_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(in_dim * mlp_ratio), in_dim),
        )

        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        return x + self.mix(x)


class MixVPR(nn.Module):
    """MixVPR aggregation layer.

    Source: https://github.com/amaralibey/MixVPR/blob/main/models/aggregators/mixvpr.py
    """

    def __init__(
        self,
        in_channels: int = 256,
        in_h: int = 10,
        in_w: int = 18,
        out_channels: int = 128,
        mix_depth: int = 4,
        mlp_ratio: float = 1,
        out_rows: int = 2,
    ) -> None:
        """Aggregation layer from the MixVPR paper.

        Args:
            in_channels (int): Depth of input feature maps. Defaults to 256.
            in_h (int): Height of input feature maps. Defaults to 10.
            in_w (int): Width of input feature maps. Defaults to 18.
            out_channels (int): Depth wise projection dimension. Defaults to 128.
            mix_depth (int): The number of stacked FeatureMixers. Defaults to 4.
            mlp_ratio (float): Ratio of the mid projection layer in the mixer block. Defaults to 1.
            out_rows (int): Row wise projection dimesion. Defaults to 2.
        """
        super().__init__()

        self.in_h = in_h  # height of input feature maps
        self.in_w = in_w  # width of input feature maps
        self.in_channels = in_channels  # depth of input feature maps

        self.out_channels = out_channels  # depth wise projection dimension
        self.out_rows = out_rows  # row wise projection dimesion

        self.mix_depth = mix_depth  # L the number of stacked FeatureMixers
        self.mlp_ratio = mlp_ratio  # ratio of the mid projection layer in the mixer block

        hw = in_h * in_w
        self.mix = nn.Sequential(
            *[FeatureMixerLayer(in_dim=hw, mlp_ratio=mlp_ratio) for _ in range(self.mix_depth)]
        )
        self.channel_proj = nn.Linear(in_channels, out_channels)
        self.row_proj = nn.Linear(hw, out_rows)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        x = x.flatten(2)
        x = self.mix(x)
        x = x.permute(0, 2, 1)
        x = self.channel_proj(x)
        x = x.permute(0, 2, 1)
        x = self.row_proj(x)
        x = F.normalize(x.flatten(1), p=2, dim=-1)
        return x
