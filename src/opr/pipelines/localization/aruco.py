"""ArucoPlaceRecognitionPipeline pipeline."""
from typing import Dict

import cv2
import numpy as np
from torch import Tensor
from scipy.spatial.transform import Rotation as R
from geotransformer.utils.pointcloud import get_rotation_translation_from_transform
from opr.datasets.itlp import ITLPCampus
from opr.pipelines.localization import LocalizationPipeline
from opr.pipelines.place_recognition import PlaceRecognitionPipeline
from opr.pipelines.registration import PointcloudRegistrationPipeline


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
        db_dataset: ITLPCampus,  # TODO: replace with a generic "inference" dataset
        aruco_metadata: Dict,
        camera_metadata: Dict
    ) -> None:
        """ArucoLocalization Pipeline.

        The task of localiation is solved in two branch:
        1. Find the best match for the query in the database (Place Recognition) and 
        Refine the pose estimate using the query and the database match (Registration).
        2. Detect Aruco Marker (Place Recognition) and find transformation from encoded in marker pose (Registration)

        Args:
            place_recognition_pipeline (PlaceRecognitionPipeline): Place Recognition pipeline.
            registration_pipeline (PointcloudRegistrationPipeline): Registration pipeline.
            db_dataset (ITLPCampus): Database dataset.
            aruco_metadata (Dict): Required information about aruco markers.
            camera_metadata (Dict): Required information about camera parameters.
        """
        super().__init__(place_recognition_pipeline, registration_pipeline, db_dataset)
        self.aruco_metadata = aruco_metadata
        self.camera_metadata = camera_metadata

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
        min_dist = np.inf

        arucoDict = cv2.aruco.getPredefinedDictionary(self.aruco_metadata["aruco_type"])
        arucoParams = cv2.aruco.DetectorParameters()
        arucoParams.useAruco3Detection = True
        arucoParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
        camera_names = [key for key in input_data.keys() if key.startswith("image_")]
        for camera_name_ in camera_names:
            frame = input_data[camera_name_].permute(1, 2, 0).numpy()
            corners, ids, _ = detector.detectMarkers(frame)

            sensor2baselink = pose_to_matrix(np.array(self.camera_metadata[f"{camera_name_[6:]}2baselink"]))

            if len(corners) > 0:
                for i in range(0, len(ids)):
                    print(f"Detect Aruco with id {ids[i]} on {camera_name_}")
                    try:
                        aruco2world = pose_to_matrix(self.aruco_metadata["aruco_gt_pose_by_id"][ids[i][0]])
                    except:
                        print(f"Can't find Aruco with id {ids[i]} in aruco_metadata !")
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
                        poses["pose_by_aruco"] = np.concatenate([trans, rot])

        poses["pose_by_place_recognition"] = super().infer(input_data)["estimated_pose"]

        return poses
