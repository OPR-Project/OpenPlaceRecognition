"""Package-level utility functions."""
import random
from typing import Any, Dict, List, Tuple

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


def in_sorted_array(e: int, array: np.ndarray) -> bool:
    """Checks whether the given value `e` is in sorted array.

    Code adopted from repository: https://github.com/jac99/MinkLocMultimodal, MIT License

    Args:
        e (int): Value to search for.
        array (np.ndarray): Sorted array to look from.

    Returns:
        bool: Whether the given value `e` is in sorted `array`.
    """
    pos = np.searchsorted(array, e)
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


def flatten_dict(nested_dict: Dict[str, Any], parent_key="", sep="/") -> Dict[str, Any]:
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
