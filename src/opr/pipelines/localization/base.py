"""Hierarchical Localization Pipeline."""
from typing import Dict

import numpy as np
from geotransformer.utils.pointcloud import (
    get_rotation_translation_from_transform,
    get_transform_from_rotation_translation,
)
from scipy.spatial.transform import Rotation
from torch import Tensor

from opr.datasets.itlp import ITLPCampus
from opr.pipelines.place_recognition import PlaceRecognitionPipeline
from opr.pipelines.registration import PointcloudRegistrationPipeline


class LocalizationPipeline:
    """Hierarchical Localization Pipeline.

    The task of localiation is solved in two steps:
    1. Find the best match for the query in the database (Place Recognition).
    2. Refine the pose estimate using the query and the database match (Registration).
    """

    def __init__(
        self,
        place_recognition_pipeline: PlaceRecognitionPipeline,
        registration_pipeline: PointcloudRegistrationPipeline,
        db_dataset: ITLPCampus,  # TODO: replace with a generic "inference" dataset
    ) -> None:
        """Hierarchical Localization Pipeline.

        The task of localiation is solved in two steps:
        1. Find the best match for the query in the database (Place Recognition).
        2. Refine the pose estimate using the query and the database match (Registration).

        Args:
            place_recognition_pipeline (PlaceRecognitionPipeline): Place Recognition pipeline.
            registration_pipeline (PointcloudRegistrationPipeline): Registration pipeline.
            db_dataset (ITLPCampus): Database dataset.
        """
        self.pr_pipe = place_recognition_pipeline
        self.reg_pipeline = registration_pipeline
        self.db_dataset = db_dataset

    def infer(self, input_data: Dict[str, Tensor]) -> Dict[str, np.ndarray]:
        """Single sample inference.

        Args:
            input_data (Dict[str, Tensor]): Input data. Dictionary with keys in the following format:

                "image_{camera_name}" for images from cameras,

                "mask_{camera_name}" for semantic segmentation masks,

                "pointcloud_lidar_coords" for pointcloud coordinates from lidar,

                "pointcloud_lidar_feats" for pointcloud features from lidar.

        Returns:
            Dict[str, np.ndarray]: Inference results. Dictionary with keys:

                "db_match_pose" for database match pose in the format [tx, ty, tz, qx, qy, qz, qw],

                "estimated_pose" for estimated pose in the format [tx, ty, tz, qx, qy, qz, qw].
        """
        out_dict = {}

        pr_output = self.pr_pipe.infer(input_data)
        query_pc = input_data["pointcloud_lidar_coords"]
        db_pc = self.db_dataset[pr_output["idx"]]["pointcloud_lidar_coords"]
        db_pose = self.db_dataset[pr_output["idx"]]["pose"]
        out_dict["db_match_pose"] = db_pose

        db_pose = get_transform_from_rotation_translation(
            Rotation.from_quat(db_pose[3:]).as_matrix(), db_pose[:3]
        )
        estimated_transform = self.reg_pipeline.infer(query_pc, db_pc)
        estimated_pose = db_pose @ estimated_transform
        rot, trans = get_rotation_translation_from_transform(estimated_pose)
        rot = Rotation.from_matrix(rot).as_quat()
        pose = np.concatenate([trans, rot])
        out_dict["estimated_pose"] = pose

        return out_dict
