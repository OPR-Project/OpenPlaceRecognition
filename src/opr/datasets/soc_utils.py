"""Utility functions for Semantic-Object-Context modality."""
from typing import Optional

import cv2
import numpy as np
import seaborn as sns

# from loguru import logger


def semantic_mask_to_instances(
    mask: np.ndarray,
    area_threshold: Optional[int] = 10,
    labels_whitelist: Optional[list] = None,
) -> dict:
    """Get instance labels from semantic mask.

    Instances are defined as connected components of the same class.
    Connected components found using opencv connectedComponentsWithStats opencv algorithm
    in class-wise manner.

    Args:
        mask (ndarray): semantic mask in opencv  image format (ndarray)
        area_threshold (int, optional): minimum area of instance to be considered. Defaults to 10.
        labels_whitelist (list, optional): list of labels to consider. Defaults to None.

    Returns:
        instances (dict): dict of instances with keys as instance labels and values as instance masks.
    """
    instances = {}
    # logger.debug(f"Labels whitelist: {labels_whitelist}")
    for label in labels_whitelist:
        instances[label] = []
        binary_mask = (mask == label).astype(np.uint8)
        (
            totalLabels,
            label_ids,
            stats,
            centroid,
        ) = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

        components = []
        for label_id in range(1, totalLabels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area > area_threshold:
                components.append(label_ids == label_id)
        instances[label] = components

    return instances


def instance_masks_to_objects(
    instance_masks: dict,
    points_2d: np.ndarray,
    point_labels: np.ndarray,
    points_3d: np.ndarray,
) -> dict:
    """Get objects from instance masks.

    Args:
        instance_masks (dict): dict of instances with keys as instance labels and values as instance masks.
        points_2d (np.ndarray): 2d points of pointcloud projected to image plane
        point_labels (np.ndarray): labels of points
        points_3d (np.ndarray): 3d points of pointcloud

    Returns:
        objects (dict): dict of objects with keys as object labels and values as object properties.
    """
    objects = {}
    for label in instance_masks:
        for mask_id, mask in enumerate(instance_masks[label]):
            _, _, w, h = cv2.boundingRect(mask.astype(np.uint8))
            objects[(label, mask_id)] = {"points": [], "width": w, "height": h}

    for img_point, label, point_3d in zip(points_2d.T, point_labels, points_3d):  # points.T
        if label not in instance_masks:
            continue
        for mask_id, mask in enumerate(instance_masks[label]):
            if mask[img_point[1], img_point[0]]:
                objects[(label, mask_id)]["points"].append(point_3d)
                continue
        # if label in instance_masks:
        # logger.debug(f"Point {img_point} with label {label} not in any mask")

    for obj in objects:
        objects[obj]["points"] = np.array(objects[obj]["points"]).T
        if len(objects[obj]["points"]) == 0:
            continue
        objects[obj]["centroid"] = np.mean(objects[obj]["points"], axis=1)
        objects[obj]["num_points"] = objects[obj]["points"].shape[1]

    return objects


def generate_color_sequence(num_colors: int, palette: Optional[str] = "husl") -> list:
    """Generate color sequence.

    Args:
        num_colors (int): number of colors to generate
        palette (str, optional): palette to use. Defaults to "husl".

    Returns:
        colors (list): list of colors in RGB format.
    """
    # Using Seaborn's color_palette function to generate a sequence of high-contrast colors
    colors = sns.color_palette(
        palette,
        num_colors,
    )

    # Convert to RGB
    # colors = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors]
    return colors


def get_points_labels_by_mask(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Get point labels from semantic mask.

    Args:
        points (np.ndarray): array of 2D coordinates of projected points with shape (n, 2).
        Coordinates should match with cam_resolution.
        mask (np.ndarray): semantic mask in opencv  image format (ndarray)

    Returns:
        labels (np.ndarray): point labels taken from the mask.
    """
    labels = []
    for img_point in points.T:  # points.T
        labels.append(mask[img_point[1], img_point[0]])

    return np.asarray(labels)


def pack_objects(objects: dict, top_k: int, max_distance: float, special_classes: list) -> np.ndarray:
    """Pack objects into a single array.

    Args:
        objects (dict): dict of objects with keys as object labels and values as object properties.
        top_k (int): maximum number of each class objects to pack
        max_distance (float): maximum distance between objects
        special_classes (list): list of special classes to pack

    Returns:
        packed_objects (np.mdarray): array of packed objects with shape (N, K, 3), where N - number of classes,
        K - number of objects of each class, 3 - 3DoF coords.
    """
    classes_num = len(special_classes)
    packed_objects = [[] for _ in range(classes_num)]
    for key, obj in objects.items():
        if "centroid" not in obj:
            continue
        dist = np.linalg.norm(obj["centroid"])
        if dist > max_distance:
            continue
        idx = special_classes.index(key[0])
        packed_objects[idx].append(obj["centroid"])

    for i in range(classes_num):
        packed_objects[i] = np.array(sorted(packed_objects[i], key=lambda x: np.linalg.norm(x), reverse=True))
        if packed_objects[i].shape[0] > top_k:
            packed_objects[i] = packed_objects[i][:top_k]
        elif packed_objects[i].shape[0] < top_k:
            if packed_objects[i].shape[0] == 0:
                packed_objects[i] = np.zeros((top_k, 3))
            else:
                packed_objects[i] = np.vstack(
                    (
                        packed_objects[i],
                        np.zeros((top_k - packed_objects[i].shape[0], 3)),
                    )
                )

    packed_objects = np.array(packed_objects)

    return packed_objects


def euclidean_to_cylindrical(points: np.ndarray, to_2d: bool = False) -> np.ndarray:
    """Convert euclidean coordinates to cylindrical.

    Args:
        points (np.ndarray): array of 3D coordinates with shape (n, 3).
        to_2d (bool, optional): whether to return 2D cylindrical coordinates. Defaults to False.

    Returns:
        points (np.ndarray): array of cylindrical coordinates with shape (n, 3) or (n, 2) if to_2d is True.
    """
    points = np.atleast_2d(points)  # Ensure points are in a 2D array
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    if to_2d:
        return np.column_stack((r, theta))
    else:
        return np.column_stack((r, theta, z))


def cylindrical_to_euclidean(points: np.ndarray) -> np.ndarray:
    """Convert cylindrical coordinates to euclidean.

    Args:
        points (np.ndarray): array of cylindrical coordinates with shape (n, 3).

    Returns:
        points (np.ndarray): array of euclidean coordinates with shape (n, 3).
    """
    points = np.atleast_2d(points)  # Ensure points are in a 2D array
    r, theta, z = points[:, 0], points[:, 1], points[:, 2]
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y, z))


def euclidean_to_spherical(points: np.ndarray) -> np.ndarray:
    """Convert euclidean coordinates to spherical.

    Args:
        points (np.ndarray): array of 3D coordinates with shape (n, 3).

    Returns:
        points (np.ndarray): array of spherical coordinates with shape (n, 3).
    """
    points = np.atleast_2d(points)  # Ensure points are in a 2D array
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    rho = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / rho)  # polar angle
    phi = np.arctan2(y, x)  # azimuthal angle
    return np.column_stack((rho, theta, phi))


def spherical_to_euclidean(points: np.ndarray) -> np.ndarray:
    """Convert spherical coordinates to euclidean.

    Args:
        points (np.ndarray): array of spherical coordinates with shape (n, 3).

    Returns:
        points (np.ndarray): array of euclidean coordinates with shape (n, 3).
    """
    points = np.atleast_2d(points)  # Ensure points are in a 2D array
    rho, theta, phi = points[:, 0], points[:, 1], points[:, 2]
    x = rho * np.sin(theta) * np.cos(phi)
    y = rho * np.sin(theta) * np.sin(phi)
    z = rho * np.cos(theta)
    return np.column_stack((x, y, z))
