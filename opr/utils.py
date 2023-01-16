"""Package-level utility functions."""
import random

import numpy as np
import torch


def set_seed(seed: int = 0, make_deterministic: bool = False) -> None:
    """Set the random seed for the `random`, `numpy`, and `torch` libraries and enables deterministic mode.

    Args:
        seed (int): The random seed to use. Defaults to 0.
        make_deterministic (bool): Whether to make PyTorch deterministic. If True,
            disables PyTorch's benchmark mode and enables its deterministic mode.
            If False, leaves PyTorch's settings unchanged. Defaults to True.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if make_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False, warn_only=True)
