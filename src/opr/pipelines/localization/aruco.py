"""ArucoPlaceRecognitionPipeline pipeline."""
from typing import Dict
from os import PathLike

import cv2
import numpy as np
from torch import Tensor
from scipy.spatial.transform import Rotation as R
from geotransformer.utils.pointcloud import get_rotation_translation_from_transform
from opr.datasets.itlp import ITLPCampus
from opr.pipelines.localization import LocalizationPipeline
from opr.pipelines.place_recognition import PlaceRecognitionPipeline
from opr.pipelines.registration import PointcloudRegistrationPipeline
from opr.datasets.augmentations import DefaultImageTransform


def pose_to_matrix(pose):
    """From the 6D poses in the [tx ty tz qx qy qz qw] format to 4x4 pose matrices."""
    position = pose[:3]
    orientation_quat = pose[3:]
    rotation = R.from_quat(orientation_quat)
    pose_matrix = np.eye(4)
    pose_matrix[:3,:3] = rotation.as_matrix()
    pose_matrix[:3,3] = position
    return pose_matrix

class ArucoLocalizationPipeline(LocalizationPipeline):
    """ArucoLocalizationPipeline pipeline."""

    def __init__(
        self,
        place_recognition_pipeline: PlaceRecognitionPipeline,
        registration_pipeline: PointcloudRegistrationPipeline,
        aruco_metadata: Dict,
        camera_metadata: Dict,
        precomputed_reg_feats: bool = False,
        pointclouds_subdir: str | PathLike | None = None,
        fastest: bool = True,
        use_first_marker: bool = True
    ) -> None:
        """ArucoLocalization Pipeline.

        The task of localiation is solved in two branch:
        1. Find the best match for the query in the database (Place Recognition) and 
        Refine the pose estimate using the query and the database match (Registration).
        2. Detect Aruco Marker (Place Recognition) and find transformation from encoded in marker pose (Registration)

        Args:
            place_recognition_pipeline (PlaceRecognitionPipeline): Place Recognition pipeline.
            registration_pipeline (PointcloudRegistrationPipeline): Registration pipeline.
            aruco_metadata (Dict): Required information about aruco markers.
            camera_metadata (Dict): Required information about camera parameters.
            precomputed_reg_feats (bool): Whether to use precomputed registration features. Defaults to False.
            pointclouds_subdir (str | PathLike, optional): Sub-directory with pointclouds. Will be used
                for computing registration feats, if they are not exist; or for loading pointclouds
                if `precomputed_reg_feats=False`. Defaults to None.
            fastest (bool): cv2.useAruco3Detection option. If True - use faster algorithm.
            use_first_marker (bool): use first marker to calculate pose.
        """
        super().__init__(place_recognition_pipeline, registration_pipeline, precomputed_reg_feats, pointclouds_subdir)
        self.aruco_metadata = aruco_metadata
        self.camera_metadata = camera_metadata
        self.fastest = fastest
        self.use_first_marker = use_first_marker

    def aruco_part(self, input_data: Dict[str, Tensor]) -> np.ndarray:
        """Single aruco inference.

        Args:
            input_data (Dict[str, Tensor]): Input data. Dictionary with keys in the following format:

                "image_{camera_name}" for images from cameras,

                "mask_{camera_name}" for semantic segmentation masks,

                "pointcloud_lidar_coords" for pointcloud coordinates from lidar,

                "pointcloud_lidar_feats" for pointcloud features from lidar.
        Returns:
            np.ndarray: predicted pose in the format [tx, ty, tz, qx, qy, qz, qw],
        """
        pose_by_aruco = None
        min_dist = np.inf
        arucoDict = cv2.aruco.getPredefinedDictionary(self.aruco_metadata["aruco_type"])
        arucoParams = cv2.aruco.DetectorParameters()
        arucoParams.useAruco3Detection = self.fastest
        arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
        camera_names = [key for key in input_data.keys() if key.startswith("image_")]
        for camera_name_ in camera_names:
            frame = input_data[camera_name_]
            corners, ids, rejected = detector.detectMarkers(frame)
            sensor2baselink = pose_to_matrix(np.array(self.camera_metadata[f"{camera_name_[6:]}2baselink"]))

            if len(corners) > 0:
                for i in range(0, len(ids)):
                    aruco_gt = self.aruco_metadata["aruco_gt_pose_by_id"].get(ids[i][0])
                    if aruco_gt is not None:
                        aruco2world = pose_to_matrix(aruco_gt)
                        print(f"Detect Aruco with id {ids[i]} on {camera_name_}")
                    else:
                        continue

                    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners=corners[i],
                                                        markerLength=self.aruco_metadata["aruco_size"],
                                                        cameraMatrix=np.array(self.camera_metadata[f"{camera_name_[6:]}_intrinsics"]),
                                                        distCoeffs=np.array(self.camera_metadata[f"{camera_name_[6:]}_distortion"]))

                    rvec = cv2.Rodrigues(rvec)[0].reshape(3, 3)
                    tvec = np.squeeze(tvec)
                    aruco2sensor = np.eye(4)
                    aruco2sensor[:3, :3] = rvec
                    aruco2sensor[:3, 3] = tvec

                    dist = sum([t ** 2 for t in tvec]) ** 0.5
                    baselink2world = aruco2world @ np.linalg.inv(aruco2sensor) @ np.linalg.inv(sensor2baselink)

                    if dist < min_dist:
                        print(f"Utilize Aruco with id {ids[i]} on {camera_name_} for pose estimation due min distanse")
                        min_dist = dist
                        rot, trans = get_rotation_translation_from_transform(baselink2world)
                        rot = R.from_matrix(rot).as_quat()
                        pose_by_aruco = np.concatenate([trans, rot])
                        if self.use_first_marker:
                            return pose_by_aruco
        return pose_by_aruco

    def loc_part(self, input_data: Dict[str, Tensor]) -> np.ndarray:
        """Single localization inference.

        Args:
            input_data (Dict[str, Tensor]): Input data. Dictionary with keys in the following format:

                "image_{camera_name}" for images from cameras,

                "mask_{camera_name}" for semantic segmentation masks,

                "pointcloud_lidar_coords" for pointcloud coordinates from lidar,

                "pointcloud_lidar_feats" for pointcloud features from lidar.
        Returns:
            np.ndarray: predicted pose in the format [tx, ty, tz, qx, qy, qz, qw],
        """
        im_transform = DefaultImageTransform(resize=(320, 192), train=False)
        for key in input_data.keys():
            if key.startswith("image_"):
                input_data[key] = im_transform(input_data[key])
        return super().infer(input_data)["estimated_pose"]

    def infer(self, input_data: Dict[str, Tensor]) -> Dict[str, np.ndarray]:
        """Single sample inference.

        Args:
            input_data (Dict[str, Tensor]): Input data. Dictionary with keys in the following format:

                "image_{camera_name}" for images from cameras,

                "mask_{camera_name}" for semantic segmentation masks,

                "pointcloud_lidar_coords" for pointcloud coordinates from lidar,

                "pointcloud_lidar_feats" for pointcloud features from lidar.

        Returns:
            Dict[str, Dict[str, np.ndarray]]: Inference results. Dict with dictionaries:

                "pose_by_aruco": "pose" for predicted pose in the format [tx, ty, tz, qx, qy, qz, qw],

                "pose_by_place_recognition": "pose" for predicted pose in the format [tx, ty, tz, qx, qy, qz, qw].
        """
        poses = {"pose_by_aruco": None, "pose_by_place_recognition": None}
        poses["pose_by_aruco"] = self.aruco_part(input_data)
        if poses["pose_by_aruco"] is not None:
            return poses
        poses["pose_by_place_recognition"] = self.loc_part(input_data)
        return poses
