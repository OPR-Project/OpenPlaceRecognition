"""Utility functions for the dataset preprocessing."""
from typing import List, Tuple


def check_in_test_set(
    northing: float,
    easting: float,
    test_boundary_points: List[Tuple[float, float]],
    boundary_width: Tuple[float, float],
) -> bool:
    """Checks whether the given point is in the test set.

    Args:
        northing (float): x coordinate of the point.
        easting (float): y coordinate of the point.
        test_boundary_points (List[Tuple[float, float]]): List of boundary points of the test set.
        boundary_width (Tuple[float, float]): Boundary width.

    Returns:
        bool: Whether the given point is in the test set.
    """
    in_test_set = False
    x_width, y_width = boundary_width
    for boundary_point in test_boundary_points:
        if (
            boundary_point[0] - x_width < northing < boundary_point[0] + x_width
            and boundary_point[1] - y_width < easting < boundary_point[1] + y_width
        ):
            in_test_set = True
            break
    return in_test_set
