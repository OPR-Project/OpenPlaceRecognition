"""Package-level utility functions."""
import random
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from torch import Tensor


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


def parse_device(device: Union[str, int, torch.device]) -> torch.device:
    """Parse given device argument and return torch.device object.

    Args:
        device (Union[str, int, torch.device]): Device argument.

    Returns:
        torch.device: Device object.

    Raises:
        ValueError: If device is not a string, integer or torch.device object.
    """
    if isinstance(device, torch.device):
        return device
    elif isinstance(device, str):
        return torch.device(device)
    elif isinstance(device, int):
        return torch.device(type="cuda", index=device) if device >= 0 else torch.device(type="cpu")
    else:
        raise ValueError(f"Invalid device: {device}")


def in_sorted_array(e: int, array: Tensor) -> bool:
    """Checks whether the given value `e` is in sorted array.

    Code adopted from repository: https://github.com/jac99/MinkLocMultimodal, MIT License

    Args:
        e (int): Value to search for.
        array (Tensor): Sorted array to look from.

    Returns:
        bool: Whether the given value `e` is in sorted `array`.
    """
    pos = torch.searchsorted(array, e)
    if pos == len(array) or pos == -1:
        return False
    else:
        return array[pos] == e


def accumulate_dict(dst_dict: Dict[str, Any], src_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Updates dst_dict with values from src_dict.

    Recursively traverses the src_dict dictionary and appends values to lists in dst_dict.
    If a key does not exist in dst_dict, it is created with a new list containing the corresponding value.

    Args:
        dst_dict (Dict[str, Any]): A dictionary representing statistics for an epoch.
        src_dict (Dict[str, Any]): A dictionary representing statistics for a step within an epoch.

    Returns:
        Dict[str, Any]: A dictionary representing updated statistics for an epoch.

    Example usage:
        >>> dst_dict = {}
        >>> src_dict_1 = {"train": {"total": 0.1, "image": {"loss": 0.1}}}
        >>> src_dict_2 = {"train": {"total": 0.2, "image": {"loss": 0.2}}}
        >>> dst_dict = accumulate_dict(dst_dict, src_dict_1)
        >>> print(dst_dict)
        {'train': {'total': [0.1], 'image': {'loss': [0.1]}}}
        >>> dst_dict = accumulate_dict(dst_dict, src_dict_2)
        >>> print(dst_dict)
        {'train': {'total': [0.1, 0.2], 'image': {'loss': [0.1, 0.2]}}}
    """
    for key, value in src_dict.items():
        if isinstance(value, dict):
            if key not in dst_dict:
                dst_dict[key] = {}
            dst_dict[key] = accumulate_dict(dst_dict[key], value)
        elif value is not None:
            if key not in dst_dict:
                dst_dict[key] = [value]
            else:
                dst_dict[key].append(value)
    return dst_dict


def compute_epoch_stats_mean(epoch_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Computes the mean value of each list in epoch_stats.

    Recursively traverses the epoch_stats dictionary and computes the mean value of each list
    using np.mean(). If a key does not contain a list, its value is returned as is.

    Args:
        epoch_stats (Dict[str, Any]): A dictionary representing statistics for an epoch.

    Returns:
        Dict[str, Any]: A dictionary representing the mean value of each list in epoch_stats.

    Example usage:
        >>> epoch_stats = {'train': {'total': [0.1, 0.2], 'image': {'loss': [0.1, 0.2]}}}
        >>> epoch_stats_mean = compute_epoch_stats_mean(epoch_stats)
        >>> print(epoch_stats_mean)
        {'train': {'total': 0.15000000000000002, 'image': {'loss': 0.15000000000000002}}}
    """
    for key, value in epoch_stats.items():
        if isinstance(value, dict):
            epoch_stats[key] = compute_epoch_stats_mean(value)
        elif isinstance(value, list):
            epoch_stats[key] = np.mean(value)
    return epoch_stats


def merge_nested_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two nested dictionaries that have overlapping outer keys.

    Args:
        dict1 (Dict[str, Any]): First dictionary object.
        dict2 (Dict[str, Any]): Second dictionary object.

    Returns:
        Dict[str, Any]: Merged dictionary.
    """
    merged_dict = dict1.copy()
    for key, value in dict2.items():
        if key in merged_dict and isinstance(merged_dict[key], dict) and isinstance(value, dict):
            merged_dict[key] = merge_nested_dicts(merged_dict[key], value)
        else:
            merged_dict[key] = value
    return merged_dict


def flatten_dict(nested_dict: Dict[str, Any], parent_key: str = "", sep: str = "/") -> Dict[str, Any]:
    """Flatten a nested dictionary with keys separated by `sep`.

    Args:
        nested_dict (Dict[str, Any]): A nested dictionary to flatten.
        parent_key (str): The string of parent key (used for recursion).
        sep (str): The separator to use between keys in the flattened dictionary.

    Returns:
        Dict[str, Any]: A flattened dictionary with keys separated by `sep`.
    """
    items: List[Tuple[str, Any]] = []
    for key, value in nested_dict.items():
        new_key = parent_key + sep + key if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


# TODO: refactor - may it be re-written in vectorized form?
def cartesian_to_spherical(points: np.ndarray, dataset_name: str) -> np.ndarray:
    """Converts cartesian coordinates to spherical coordinates."""
    if (np.abs(points[:, :3]) < 1e-4).all(axis=1).any():
        points = points[(np.abs(points[:, :3]) >= 1e-4).any(axis=1)]

    r = np.linalg.norm(points[:, :3], axis=1)

    # Theta is calculated as an angle measured from the y-axis towards the x-axis
    # Shifted to range (0, 360)
    theta = np.arctan2(points[:, 1], points[:, 0]) * 180 / np.pi
    theta[theta < 0] += 360

    if dataset_name.lower() == "usyd":
        # VLP-16 has 2 deg VRes and (+15, -15 VFoV).
        # Phi calculated from the vertical axis, so (75, 105)
        # Shifted to (0, 30)
        phi = (np.arccos(points[:, 2] / r) * 180 / np.pi) - 75

    elif dataset_name in ["intensityoxford", "oxford"]:
        # Oxford scans are built from a 2D scanner.
        # Phi calculated from the vertical axis, so (0, 180)
        phi = np.arccos(points[:, 2] / r) * 180 / np.pi

    elif dataset_name == "kitti":
        # HDL-64 has 0.4 deg VRes and (+2, -24.8 VFoV).
        # Phi calculated from the vertical axis, so (88, 114.8)
        # Shifted to (0, 26.8)
        phi = (np.arccos(points[:, 2] / r) * 180 / np.pi) - 88

    elif dataset_name.lower() == "nclt":
        # from +10.67° to -30.67°
        phi = (np.arccos(points[:, 2] / r) * 180 / np.pi) - 79.33

    else:
        raise NotImplementedError(f"Converting cartesian to spherical for {dataset_name} no supported.")

    if points.shape[-1] == 4:
        return np.column_stack((r, theta, phi, points[:, 3]))
    else:
        return np.column_stack((r, theta, phi))
