"""CosPlace aggregation layer implementation."""
from torch import Tensor, nn
from torch.nn import functional as F

from .gem import GeM


class CosPlace(nn.Module):
    """CosPlace aggregation layer.

    As implemented in https://github.com/gmberton/CosPlace/blob/main/model/network.py
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        """Aggregation layer as implemented in CosPlace method.

        Args:
            in_dim (int): Input dimension.
            out_dim (int): Output dimension.
        """
        super().__init__()
        self.gem = GeM()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        x = F.normalize(x, p=2, dim=1)
        x = self.gem(x)
        x = x.flatten(1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x
