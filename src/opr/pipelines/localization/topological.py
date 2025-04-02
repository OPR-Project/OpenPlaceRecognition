"""ArucoPlaceRecognitionPipeline pipeline."""
from typing import Dict, List
from os import PathLike
from tqdm import tqdm

import numpy as np
from torch import Tensor
from scipy.spatial.transform import Rotation
from opr.pipelines.localization import LocalizationPipeline
from opr.pipelines.place_recognition import PlaceRecognitionPipeline
from opr.pipelines.registration import PointcloudRegistrationPipeline, SequencePointcloudRegistrationPipeline
from opr.models.localization.topo_graph import TopologicalGraph
from opr.models.localization.local_grid import LocalGrid
from opr.models.localization.pose_utils import *
from opr.datasets.base import BasePlaceRecognitionDataset
from geotransformer.utils.pointcloud import (
    get_rotation_translation_from_transform,
    get_transform_from_rotation_translation,
)

class TopologicalLocalizationPipeline(LocalizationPipeline):
    """TopologicalLocalizationPipeline pipeline."""
    
    def __init__(
        self,
        database: BasePlaceRecognitionDataset,
        place_recognition_pipeline: PlaceRecognitionPipeline,
        registration_pipeline: PointcloudRegistrationPipeline | SequencePointcloudRegistrationPipeline,
        precomputed_reg_feats: bool = False,
        pointclouds_subdir: str | PathLike | None = None,
        camera_names: List[str] = [],
        multi_sensor_fusion: bool = False,
        edge_threshold: float = 5.0,
        top_k: int = 5
    ) -> None:
        """Topological Localization Pipeline.

        The task of localiation is solved in three steps:
        1. Find the top-k match candidates for the query in the graph (Place Recognition).
        2. For each candidate, refine the pose estimate using registration pipeline (Registration).
        3. Find the proper match from candidates depending from current state in the graph and its edges.

        Args:
            place_recognition_pipeline (PlaceRecognitionPipeline): Place Recognition pipeline.
            registration_pipeline (PointcloudRegistrationPipeline): Registration pipeline.
            precomputed_reg_feats (bool): Whether to use precomputed registration features. Defaults to False.
            pointclouds_subdir (str | PathLike, optional): Sub-directory with pointclouds. Will be used
                for computing registration feats, if they are not exist; or for loading pointclouds
                if `precomputed_reg_feats=False`. Defaults to None.
            edge_threshold (float): Threshold of distance between graph's nodes to link them by an edge (in meters). Defaults to 5.0.
            top_k (int): Number of matches that we consider to localize in the graph.

        Raises:
            ValueError: Pointclouds sub-directory must be provided if precomputed registration
                features are not used.
            ValueError: Precomputed registration features are only supported for HRegNet.
        """
        super().__init__(place_recognition_pipeline, registration_pipeline, precomputed_reg_feats, pointclouds_subdir)
        self.edge_threshold = edge_threshold
        self.top_k = top_k
        self.camera_names = camera_names
        self.multi_sensor_fusion = multi_sensor_fusion
        self.database = database
        self._build_graph_from_database()
        self.vcur = None
        self.lost_state = True

    def _build_graph_from_database(self) -> None:
        """Building topological graph at start.
        At start, builds the topological graph from all the samples from database of self.pr_pipe.
        """
        print('Start building topological graph from database')
        print('Preprocessing database...')
        self.graph = TopologicalGraph(self.pr_pipe.model, self.reg_pipe, fusion=self.multi_sensor_fusion)
        self.poses = []
        for i in range(len(self.database)):
            input_data = self.database[i]
            # for key in input_data:
            #     if key.startswith('image_'):
            #         input_data[key] = input_data[key].astype(float)
            db_pose = input_data['pose'].cpu().numpy()
            x, y = db_pose[:2]
            _, __, theta = Rotation.from_quat(db_pose[3:]).as_rotvec()
            self.graph.add_vertex(x, y, theta, input_data)
            self.poses.append(db_pose)
        graph_grids = []
        for x, y, theta, cloud in self.graph.vertices:
            grid = LocalGrid()
            grid.load_from_cloud(cloud)
            graph_grids.append(grid)
        print('Extracting topological connectivity...')
        for i in tqdm(range(len(self.graph.vertices))):
            for j in range(i + 1, len(self.graph.vertices)):
                xi, yi, theta_i, cloud_i = self.graph.get_vertex(i)
                xj, yj, theta_j, cloud_j = self.graph.get_vertex(j)
                rel_x, rel_y, rel_theta = get_rel_pose(xj, yj, theta_j, xi, yi, theta_i)
                dst = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                adjacent = False
                if dst < self.edge_threshold:
                    cloud_j_transformed = transform_pcd(cloud_j, rel_x, rel_y, rel_theta)
                    grid_i = graph_grids[i]
                    grid_j = LocalGrid()
                    grid_j.load_from_cloud(cloud_j_transformed)
                    adjacent = ((grid_i.grid * grid_j.grid == 1).sum() > 0)
                if adjacent:
                    self.graph.add_edge(j, i, rel_x, rel_y, rel_theta)
        print('Done!')

    def infer(self, input_data: Dict[str, Tensor] | list[Dict[str, Tensor]]) -> Dict[str, np.ndarray]:
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

        Raises:
            ValueError: Provided input data is a list, but the pipeline is not for sequences.
            ValueError: Provided input data is not a list, but the pipeline is for sequences.
        """
        if isinstance(input_data, list) and not self.sequences:
            raise ValueError("Provided input data is a list, but the pipeline is not for sequences.")
        if not isinstance(input_data, list) and self.sequences:
            raise ValueError("Provided input data is not a list, but the pipeline is for sequences.")
        if isinstance(input_data, list):
            query_pc = [x["pointcloud_lidar_coords"] for x in input_data]
        else:
            query_pc = input_data["pointcloud_lidar_coords"]
        images = []
        for key in input_data.keys():
            if key.startswith('image_'):
                images.append(input_data[key])
        out_dict = {}

        pred_i, pred_tf = self.graph.get_k_most_similar(self.top_k, query_pc, *images)
        if self.vcur is None or self.lost_state:
            self.vcur = pred_i[0]
            tf = pred_tf[0]
        else:
            found = False
            for idx, tf in zip(pred_i, pred_tf):
                if self.vcur == idx or self.graph.has_edge(self.vcur, idx):
                    self.vcur = idx
                    found = True
                    self.lost_state = False
                    break
            if not found:
                self.vcur = pred_i[0]
                self.lost_state = True

        x, y, theta, cloud = self.graph.get_vertex(self.vcur)
        db_pose = get_transform_from_rotation_translation(
            Rotation.from_quat(self.poses[self.vcur][3:]).as_matrix(), self.poses[self.vcur][:3]
        )
        out_dict["db_match_pose"] = db_pose
        out_dict["db_match_idx"] = self.vcur

        estimated_transform = get_transform_from_rotation_translation(
            Rotation.from_rotvec([0, 0, tf[2]]).as_matrix(), tf[3:]
        )
        estimated_pose = db_pose @ estimated_transform#self._invert_rigid_transformation_matrix(estimated_transform)
        rot, trans = get_rotation_translation_from_transform(estimated_pose)
        rot = Rotation.from_matrix(rot).as_quat()
        pose = np.concatenate([trans, rot])
        out_dict["estimated_pose"] = pose

        return out_dict