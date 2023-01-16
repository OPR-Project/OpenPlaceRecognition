import numpy as np
import torch

from opr.utils import set_seed


def test_set_seed():
    # Test that setting the seed works
    set_seed(42)
    assert np.random.get_state()[1][0] == 42  # Check numpy
    assert torch.initial_seed() == 42  # Check torch

    # Test that setting the seed again resets the state
    set_seed(42)
    assert np.random.get_state()[1][0] == 42  # Check numpy
    assert torch.initial_seed() == 42  # Check torch

    # Test that disabling determinism works
    set_seed(42, make_deterministic=False)
    assert torch.backends.cudnn.benchmark is True
    assert torch.backends.cudnn.deterministic is False

    # Test that enabling determinism works
    set_seed(42, make_deterministic=True)
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is True
