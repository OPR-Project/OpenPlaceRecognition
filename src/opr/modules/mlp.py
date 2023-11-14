"""2-layer MLP module implementation."""
from functools import partial
from typing import Optional

from torch import Tensor, nn


class MLP(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: str = "gelu",
        bias: bool = True,
        drop: float = 0.0,
        use_conv: bool = False,
    ) -> None:
        """MLP as used in Vision Transformer, MLP-Mixer and related networks.

        Args:
            in_features (int): Number of input features.
            hidden_features (int, optional): Number of hidden features. Defaults to None.
            out_features (int, optional): Number of output features. Defaults to None.
            act_layer (str): Activation function. Defaults to "gelu".
            bias (bool): Whether to use bias. Defaults to True.
            drop (float): Dropout probability. Defaults to 0.0.
            use_conv (bool): Whether to use Conv2d instead of Linear. Defaults to False.

        Raises:
            ValueError: Unsupported activation function.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias)
        if act_layer.lower() == "gelu":
            self.act = nn.GELU()
        elif act_layer.lower() == "relu":
            self.act = nn.ReLU()
        elif act_layer.lower() == "leakyrelu":
            self.act = nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation function: {act_layer}")
        self.drop1 = nn.Dropout(drop)
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
