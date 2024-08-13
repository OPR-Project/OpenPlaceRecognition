"""Pointcloud registration pipeline."""
from os import PathLike
from typing import List, Optional, Tuple, Union

import numpy as np
import open3d as o3d
import torch
import cv2
import os
import faiss
from torch import Tensor, nn
from skimage.io import imsave
from scipy.spatial.transform import Rotation

from opr.utils import init_model, parse_device

import warnings
warnings.filterwarnings('ignore')

class PointcloudRegistrationPipeline:
    """Pointcloud registration pipeline."""

    def __init__(
        self,
        model: nn.Module,
        model_weights_path: Optional[Union[str, PathLike]] = None,
        device: Union[str, int, torch.device] = "cuda",
        voxel_downsample_size: Optional[float] = 0.3,
    ) -> None:
        """Pointcloud registration pipeline.

        Args:
            model (nn.Module): Model.
            model_weights_path (Union[str, PathLike], optional): Path to the model weights.
                If None, the weights are not loaded. Defaults to None.
            device (Union[str, int, torch.device]): Device to use. Defaults to "cuda".
            voxel_downsample_size (Optional[float]): Voxel downsample size. Defaults to 0.3.
        """
        self.device = parse_device(device)
        self.model = init_model(model, model_weights_path, self.device)
        self.voxel_downsample_size = voxel_downsample_size

    def _downsample_pointcloud(self, pc: Tensor) -> Tensor:
        """Downsample the pointcloud.

        Args:
            pc (Tensor): Pointcloud. Coordinates array of shape (N, 3).

        Returns:
            Tensor: Downsampled pointcloud. Coordinates array of shape (M, 3), where M <= N.
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc.cpu().numpy())
        pcd = pcd.voxel_down_sample(self.voxel_downsample_size)
        pc = torch.from_numpy(np.array(pcd.points).astype(np.float32)).float()
        return pc

    def infer(self, query_pc: Tensor, db_pc: Tensor) -> np.ndarray:
        """Infer the transformation between the query and the database pointclouds.

        Args:
            query_pc (Tensor): Query pointcloud. Coordinates array of shape (N, 3).
            db_pc (Tensor): Database pointcloud. Coordinates array of shape (M, 3).

        Returns:
            np.ndarray: Transformation matrix.
        """
        query_pc = self._downsample_pointcloud(query_pc)
        db_pc = self._downsample_pointcloud(db_pc)
        with torch.no_grad():
            transform = self.model(query_pc, db_pc)["estimated_transform"]
        return transform.cpu().numpy()


class SequencePointcloudRegistrationPipeline(PointcloudRegistrationPipeline):
    """Pointcloud registration pipeline that supports sequences."""

    def __init__(
        self,
        model: nn.Module,
        model_weights_path: Optional[Union[str, PathLike]] = None,
        device: Union[str, int, torch.device] = "cuda",
        voxel_downsample_size: Optional[float] = 0.3,
    ) -> None:
        """Pointcloud registration pipeline that supports sequences.

        Args:
            model (nn.Module): Model.
            model_weights_path (Union[str, PathLike], optional): Path to the model weights.
                If None, the weights are not loaded. Defaults to None.
            device (Union[str, int, torch.device]): Device to use. Defaults to "cuda".
            voxel_downsample_size (Optional[float]): Voxel downsample size. Defaults to 0.3.
        """
        super().__init__(model, model_weights_path, device, voxel_downsample_size)
        self.ransac_pipeline = RansacGlobalRegistrationPipeline(
            voxel_downsample_size=0.5  # handcrafted optimal value for fast inference
        )

    def _transform_points(self, points: Tensor, transform: Tensor) -> Tensor:
        points_hom = torch.cat((points, torch.ones((points.shape[0], 1), device=points.device)), dim=1)
        # print(type(points_hom), type(transform))
        # print(points_hom.dtype, transform.dtype)
        points_transformed_hom = points_hom @ transform
        points_transformed = points_transformed_hom[:, :3] / points_transformed_hom[:, 3].unsqueeze(-1)
        return points_transformed

    def infer(self, query_pc_list: List[Tensor], db_pc: Tensor) -> np.ndarray:
        """Infer the transformation between the query sequence and the database pointclouds.

        Args:
            query_pc_list (List[Tensor]): Sequence of query pointclouds. Coordinates arrays of shape (N, 3).
            db_pc (Tensor): Database pointcloud. Coordinates array of shape (M, 3).

        Returns:
            np.ndarray: Transformation matrix.
        """
        if len(query_pc_list) > 1:
            accumulated_query_pc = query_pc_list[-1]
            for pc in query_pc_list[-2::-1]:
                transform = torch.tensor(
                    self.ransac_pipeline.infer(accumulated_query_pc, pc), dtype=torch.float32
                )
                accumulated_query_pc = torch.cat(
                    [accumulated_query_pc, self._transform_points(pc, transform)], dim=0
                )
        else:
            accumulated_query_pc = query_pc_list[0]
        return super().infer(accumulated_query_pc, db_pc)


class RansacGlobalRegistrationPipeline:
    """Pointcloud registration pipeline using RANSAC."""

    def __init__(self, voxel_downsample_size: float = 0.5) -> None:
        """Pointcloud registration pipeline using RANSAC.

        Args:
            voxel_downsample_size (float): Voxel downsample size. Defaults to 0.5.
        """
        self.voxel_downsample_size = voxel_downsample_size

    def _preprocess_point_cloud(
        self, points: Tensor
    ) -> Tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd_down = pcd.voxel_down_sample(self.voxel_downsample_size)
        radius_normal = self.voxel_downsample_size * 2
        pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        radius_feature = self.voxel_downsample_size * 5
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
        )
        return pcd_down, pcd_fpfh

    def _execute_global_registration(
        self,
        source_down: o3d.geometry.PointCloud,
        target_down: o3d.geometry.PointCloud,
        source_fpfh: o3d.pipelines.registration.Feature,
        target_fpfh: o3d.pipelines.registration.Feature,
        max_iter: int = 100000,
    ) -> o3d.pipelines.registration.RegistrationResult:
        distance_threshold = self.voxel_downsample_size * 1.5
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down,
            target_down,
            source_fpfh,
            target_fpfh,
            True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3,
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(max_iter, 0.999),
        )
        return result

    def transform_icp(self, ref_cloud, cand_cloud, ransac_transform):
        threshold = 0.3
        ref_cloud_o3d = o3d.geometry.PointCloud()
        cand_cloud_o3d = o3d.geometry.PointCloud()
        ref_cloud_o3d.points = o3d.utility.Vector3dVector(ref_cloud[:, :3])
        cand_cloud_o3d.points = o3d.utility.Vector3dVector(cand_cloud[:, :3])
        ref_cloud_o3d = ref_cloud_o3d.voxel_down_sample(0.1)
        cand_cloud_o3d = cand_cloud_o3d.voxel_down_sample(0.1)
        reg_p2p = registration.registration_icp(ref_cloud_o3d, cand_cloud_o3d, threshold, ransac_transform,
                registration.TransformationEstimationPointToPoint(),
                registration.ICPConvergenceCriteria(max_iteration = 500))
        tf_matrix = reg_p2p.transformation.copy()
        tf_rotation = Rotation.from_matrix(tf_matrix[:3, :3]).as_rotvec()
        tf_rotation[:2] = 0
        r = Rotation.from_rotvec(tf_rotation).as_matrix()
        tf_matrix[:3, :3] = r
        tf_translation = tf_matrix[:3, 3]
        return tf_matrix, reg_p2p.fitness

    def infer(self, query_pc: Tensor, db_pc: Tensor, max_iter: int = 100000) -> np.ndarray:
        """Infer the transformation between the query and the database pointclouds.

        Args:
            query_pc (Tensor): Query pointcloud. Coordinates array of shape (N, 3).
            db_pc (Tensor): Database pointcloud. Coordinates array of shape (M, 3).

        Returns:
            np.ndarray: Transformation matrix.
        """
        query_pc = query_pc.cpu().numpy()
        db_pc = db_pc.cpu().numpy()
        source_down, source_fpfh = self._preprocess_point_cloud(query_pc)
        target_down, target_fpfh = self._preprocess_point_cloud(db_pc)
        ransac_result = self._execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, max_iter)
        return self.transform_icp(query_pcd, db_pcd, result.transformation)


class Feature2DGlobalRegistrationPipeline:
    """Pointcloud registration pipeline using features extracted from pointclouds` 2d projection"""
    def __init__(self, 
                 voxel_downsample_size: float = 0.1,
                 detector_type: str = 'ORB',
                 n_keypoints: int = 400,
                 outlier_thresholds: List[float] = [5.0],
                 min_matches: int = 5,
                 max_range: float = 8.0,
                 floor_height: Union[float, str] = 'auto',
                 ceil_height: Union[float, str] = 'auto',
                 save_dir: str = '~') -> None:
        self.grid_size = voxel_downsample_size
        self.max_range = max_range
        self.floor_height = floor_height
        self.ceil_height = ceil_height
        self.detector_type = detector_type
        if self.detector_type == 'ORB':
            self.detector = cv2.ORB_create()
        elif self.detector_type == 'SIFT':
            self.detector = cv2.SIFT_create(n_keypoints)
        elif self.detector_type == 'HarrisWithDistance':
            self.detector = cv2.ORB_create()
        else:
            print('Only "ORB", "SIFT" of "HarrisWithDistance" detector types are supported. Set correct detector type')
            self.detector = None
        self.outlier_thresholds = np.array(outlier_thresholds) / self.grid_size
        self.min_matches = min_matches
        self.cnt = 0
        self.save_dir = save_dir

    def _point_based_matching(self, point_pairs: np.ndarray) -> Tuple[float, float, float]:
        """
        This function is based on the paper "Robot Pose Estimation in Unknown Environments by Matching 2D Range Scans"
        by F. Lu and E. Milios.

        :param point_pairs: the matched point pairs [((x1, y1), (x1', y1')), ..., ((xi, yi), (xi', yi')), ...]
        :return: the rotation angle and the 2D translation (x, y) to be applied for matching the given pairs of points
        """
        x_mean = 0
        y_mean = 0
        xp_mean = 0
        yp_mean = 0
        n = len(point_pairs)
        if n == 0:
            return None, None, None
        for pair in point_pairs:
            (x, y), (xp, yp) = pair
            x_mean += x
            y_mean += y
            xp_mean += xp
            yp_mean += yp
        x_mean /= n
        y_mean /= n
        xp_mean /= n
        yp_mean /= n
        s_x_xp = 0
        s_y_yp = 0
        s_x_yp = 0
        s_y_xp = 0
        for pair in point_pairs:
            (x, y), (xp, yp) = pair
            s_x_xp += (x - x_mean)*(xp - xp_mean)
            s_y_yp += (y - y_mean)*(yp - yp_mean)
            s_x_yp += (x - x_mean)*(yp - yp_mean)
            s_y_xp += (y - y_mean)*(xp - xp_mean)
        rot_angle = np.arctan2(s_x_yp - s_y_xp, s_x_xp + s_y_yp)
        translation_x = xp_mean - (x_mean*np.cos(rot_angle) - y_mean*np.sin(rot_angle))
        translation_y = yp_mean - (x_mean*np.sin(rot_angle) + y_mean*np.cos(rot_angle))
        return rot_angle, translation_x, translation_y

    def _remove_floor_and_ceil(self, cloud: np.ndarray) -> np.ndarray:
        if self.floor_height == 'auto':
            heights = np.linspace(-4.0, 4.0, 41)
            bins = []
            for i, height in enumerate(heights[:-1]):
                bins.append(len(cloud[(cloud[:, 2] > height) * (cloud[:, 2] < heights[i + 1])]))
            #print('Bins:', bins)
            floor_index = np.argmax(bins[:20]) + 1
            floor_height = heights[floor_index]
            assert floor_index < len(heights) - 5
            ceil_index = floor_index + 5 + np.argmax(bins[floor_index + 5:])
            ceil_height = heights[ceil_index]
        else:
            floor_height = self.floor_height
            ceil_height = self.ceil_height
        #print('Floor height:', floor_height)
        #print('Ceil height:', ceil_height)
        return cloud[(cloud[:, 2] > floor_height) * (cloud[:, 2] < ceil_height)]

    def _extract_floor(self, cloud: np.ndarray) -> np.ndarray:
        if self.floor_height == 'auto':
            heights = np.linspace(-4.0, 4.0, 41)
            bins = []
            for i, height in enumerate(heights[:-1]):
                bins.append(len(cloud[(cloud[:, 2] > height) * (cloud[:, 2] < heights[i + 1])]))
            #print('Bins:', bins)
            floor_index = np.argmax(bins[:20]) + 1
            floor_height = heights[floor_index]
            assert floor_index < len(heights) - 5
        else:
            floor_height = self.floor_height
        return cloud[cloud[:, 2] <= floor_height + 0.1]

    def _pcd_to_img(self, cloud: np.ndarray) -> np.ndarray:
        points = cloud[(np.abs(cloud[:, 0]) < self.max_range) * (np.abs(cloud[:, 1]) < self.max_range)]
        grid = np.zeros((int((2 * self.max_range + 2) / self.grid_size) + 1, int((2 * self.max_range + 2) / self.grid_size) + 1))
        points_int = ((points + [self.max_range + 1, self.max_range + 1]) / self.grid_size).astype(np.int32)
        grid[points_int[:, 0], points_int[:, 1]] = 1
        return grid

    def transform_pcd(self, points: np.ndarray, tf_matrix: np.ndarray,) -> np.ndarray:
        points_expanded = np.concatenate([points[:, :3], np.ones((points.shape[0], 1))], axis=1)
        points_transformed = np.linalg.inv(tf_matrix) @ points_expanded.T
        points_transformed = points_transformed.T[:, :3] / points_transformed.T[:, 3:]
        return points_transformed

    def _raycast(self, grid: np.ndarray, n_rays: int = 1000, center_point: Union[Tuple[float, float], None] = None) -> np.ndarray:
        grid_raycasted = grid.copy()
        if center_point is None:
            center_point = (grid.shape[0] // 2, grid.shape[1] // 2)
        for sector in range(n_rays):
            angle = sector / n_rays * 2 * np.pi - np.pi
            ii = center_point[0] + np.sin(angle) * np.arange(0, grid.shape[0] // 2)
            jj = center_point[1] + np.cos(angle) * np.arange(0, grid.shape[0] // 2)
            ii = ii.astype(int)
            jj = jj.astype(int)
            good_ids = ((ii > 0) * (ii < grid.shape[0]) * (jj > 0) * (jj < grid.shape[1])).astype(bool)
            ii = ii[good_ids]
            jj = jj[good_ids]
            points_on_ray = grid[ii, jj]
            if len(points_on_ray.nonzero()[0]) > 0:
                last_obst = points_on_ray.nonzero()[0][-1]
                grid_raycasted[ii[:last_obst], jj[:last_obst]] = 1
            else:
                grid_raycasted[ii, jj] = 1
        return grid_raycasted

    def get_fitness(self, ref_cloud: np.ndarray, cand_cloud: np.ndarray, tf_matrix: np.ndarray, save_dir: Union[str, None] = None) -> float:
        ref_cloud_wo_floor_and_ceil = self._remove_floor_and_ceil(ref_cloud)[:, :2]
        ref_cloud_floor = self._extract_floor(ref_cloud)[:, :2]
        cand_cloud_transformed = self.transform_pcd(cand_cloud, tf_matrix)
        cand_cloud_wo_floor_and_ceil = self._remove_floor_and_ceil(cand_cloud_transformed)[:, :2]
        cand_cloud_floor = self._extract_floor(cand_cloud_transformed)[:, :2]
        kernel = np.ones((7, 7), dtype=np.uint8)
        ref_wall_mask = self._pcd_to_img(ref_cloud_wo_floor_and_ceil)
        ref_wall_mask_dilated = cv2.dilate(ref_wall_mask, kernel)
        cand_wall_mask = self._pcd_to_img(cand_cloud_wo_floor_and_ceil)
        cand_wall_mask_dilated = cv2.dilate(cand_wall_mask, kernel)
        ref_floor_mask = self._pcd_to_img(ref_cloud_floor)
        ref_floor_mask = self._raycast(ref_wall_mask) - ref_wall_mask
        cand_floor_mask = self._pcd_to_img(cand_cloud_floor)
        cand_floor_mask = self._raycast(cand_wall_mask, center_point=(cand_floor_mask.shape[0] // 2 + tf_matrix[0, 3] / self.grid_size, 
                                                                      cand_floor_mask.shape[0] // 2 + tf_matrix[1, 3] / self.grid_size)) \
                                                                      - cand_wall_mask
        intersection = np.sum(np.clip(ref_wall_mask + ref_floor_mask, 0, 1) * np.clip(cand_wall_mask + cand_floor_mask, 0, 1))
        union = np.sum(np.clip(ref_wall_mask + ref_floor_mask + cand_wall_mask + cand_floor_mask, 0, 1))
        #print('Intersection:', intersection)
        #print('Union:', union)
        iou = intersection / union
        rot_angle = Rotation.from_matrix(tf_matrix[:3, :3]).as_rotvec()[2]
        #print('Transform:', tf_matrix[0, 3], tf_matrix[1, 3], rot_angle)
        #print('Iou from get_fitness:', iou)
        good_match = np.sum(cand_wall_mask_dilated * ref_wall_mask)
        bad_match = np.sum(cand_floor_mask * (1 - cand_wall_mask_dilated) * ref_wall_mask) + \
                    np.sum(ref_floor_mask * (1 - ref_wall_mask_dilated) * cand_wall_mask)
        save_dir = os.path.join(self.save_dir, str(self.cnt))
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            np.savetxt(os.path.join(save_dir, 'tf_matrix.txt'), tf_matrix)
            imsave(os.path.join(save_dir, 'ref_wall_mask.png'), (ref_wall_mask * 255).astype(np.uint8))
            imsave(os.path.join(save_dir, 'cand_wall_mask.png'), (cand_wall_mask * 255).astype(np.uint8))
            imsave(os.path.join(save_dir, 'ref_floor_mask.png'), (ref_floor_mask * 255).astype(np.uint8))
            imsave(os.path.join(save_dir, 'cand_floor_mask.png'), (cand_floor_mask * 255).astype(np.uint8))
            grid_aligned = np.zeros((ref_wall_mask.shape[0], ref_wall_mask.shape[1], 3))
            grid_aligned[:, :, 0] = ref_floor_mask + ref_wall_mask
            grid_aligned[:, :, 1] = cand_floor_mask + cand_wall_mask
            imsave(os.path.join(save_dir, 'grid_aligned.png'), (grid_aligned * 255).astype(np.uint8))
            np.savez(os.path.join(save_dir, 'ref_cloud.npz'), ref_cloud)
            np.savez(os.path.join(save_dir, 'cand_cloud.npz'), cand_cloud)
            np.savez(os.path.join(save_dir, 'cand_cloud_transformed.npz'), cand_cloud_transformed)
            np.savetxt(os.path.join(save_dir, 'results.txt'), 
                    [good_match, bad_match, iou, good_match / (good_match + bad_match) * iou ** 0.25])
            #print('cnt:', self.cnt)
        self.cnt += 1
        print('Reg score from fitness:', good_match / (good_match + bad_match) * iou ** 0.25)
        return good_match / (good_match + bad_match) * iou ** 0.25

    def infer(self, ref_cloud: Tensor, cand_cloud: Tensor, save_dir: Union[str, None] = None) -> Tuple[Union[np.ndarray, None], Union[float, None]]:
        # Convert clouds from tensors to numpy arrays
        ref_cloud_numpy = ref_cloud.cpu().numpy()
        cand_cloud_numpy = cand_cloud.cpu().numpy()

        # Convert clouds to images
        ref_cloud_numpy_cropped = self._remove_floor_and_ceil(ref_cloud_numpy)[:, :2]
        #print('Ref cloud cropped shape:', ref_cloud_numpy_cropped.shape)
        img_ref = self._pcd_to_img(ref_cloud_numpy_cropped)
        img_ref = (img_ref * 255).astype(np.uint8)
        kernel = np.ones((2, 2))
        img_ref = cv2.dilate(img_ref, kernel)
        img_ref = cv2.GaussianBlur(img_ref, (3, 3), 0.5)
        #img_ref = cv2.resize(img_ref, None, fx=0.5, fy=0.5)
        #print('Img ref min and max:', img_ref.min(), img_ref.mean(), img_ref.max())
        cand_cloud_numpy_cropped = self._remove_floor_and_ceil(cand_cloud_numpy)[:, :2]
        #print('Cand cloud numpy cropped shape:', cand_cloud_numpy_cropped.shape)
        img_cand = self._pcd_to_img(cand_cloud_numpy_cropped)
        img_cand = (img_cand * 255).astype(np.uint8)
        img_cand = cv2.dilate(img_cand, kernel)
        img_cand = cv2.GaussianBlur(img_cand, (3, 3), 0.5)
        #img_cand = cv2.resize(img_cand, None, fx=0.5, fy=0.5)
        #print('Img cand min and max:', img_cand.min(), img_cand.mean(), img_cand.max())
        
        # Extract features
        if self.detector_type == 'SIFT' or self.detector_type == 'ORB':
            kp_ref, des_ref = self.detector.detectAndCompute(img_ref, None)
            kp_cand, des_cand = self.detector.detectAndCompute(img_cand, None)
        elif self.detector_type == 'HarrisWithDistance':
            dst = cv2.cornerHarris(img_ref, 5, 3, 0.04)
            kp_ref = np.argwhere(dst > 0.01 * dst.max())
            kp_ref = [cv2.KeyPoint(float(y), float(x), 1) for [x, y] in kp_ref]
            kp_ref, des_ref = self.detector.compute(img_ref, kp_ref)
            dst = cv2.cornerHarris(img_cand, 5, 3, 0.04)
            kp_cand = np.argwhere(dst > 0.01 * dst.max())
            kp_cand = [cv2.KeyPoint(float(y), float(x), 1) for [x, y] in kp_cand]
            kp_cand, des_cand = self.detector.compute(img_cand, kp_cand)
            # Add geometry constraints
            distance_coef = 1.0
            xy_ref = np.array([kp.pt for kp in kp_ref]) * distance_coef
            des_ref = np.concatenate([des_ref, xy_ref], axis=1)
            xy_cand = np.array([kp.pt for kp in kp_cand]) * distance_coef
            des_cand = np.concatenate([des_cand, xy_cand], axis=1)
        #print(kp_ref, des_ref)
        #print(kp_cand, des_cand)
        
        # Match features using KNN
        if self.detector_type == 'SIFT':
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            flann = cv2.FlannBasedMatcher(index_params)
            if len(des_ref) < 2 or len(des_cand) < 2:
                print('Too few keypoints! Unable to match')
                return None, 0
            matches = flann.knnMatch(des_ref, des_cand, k=2)
            matches = [x for x in matches if len(x) == 2]
        elif self.detector_type == 'ORB':
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm = FLANN_INDEX_LSH,
                            table_number = 6, # 12
                            key_size = 12,     # 20
                            multi_probe_level = 1) #2
            search_params = dict(checks=50)   # or pass empty dictionary
            flann = cv2.FlannBasedMatcher(index_params,search_params)
            if len(des_ref) < 2 or len(des_cand) < 2:
                print('Too few keypoints! Unable to match')
                return None, 0
            matches = flann.knnMatch(des_ref, des_cand, k=2)
            matches = [x for x in matches if len(x) == 2]
        elif self.detector_type == 'HarrisWithDistance':
            des_index = faiss.IndexFlatL2(34)
            for des in des_cand:
                des_index.add(des[np.newaxis, :])
            matches = []
            for i, des in enumerate(des_ref):
                dist, idx = des_index.search(des[np.newaxis, :], k=2)
                m = cv2.DMatch(_imgIdx=0, _queryIdx=i, _trainIdx=idx[0][0], _distance=np.sqrt(dist[0][0]))
                n = cv2.DMatch(_imgIdx=0, _queryIdx=i, _trainIdx=idx[0][1], _distance=np.sqrt(dist[0][1]))
                matches.append((m, n))
        else:
            print('Incorrect detector type')
            return None, None
        #if 'inline' in self.save_dir:
        #print('Found {} matches'.format(len(matches)))
        
        # Get 2d point sets from matched features
        good = []
        if self.detector_type == 'HarrisWithDistance':
            matching_threshold = 0.8
        else:
            matching_threshold = 0.8
        for i,(m,n) in enumerate(matches):
            if m.distance < matching_threshold * n.distance:
                good.append(m)
        #if 'inline' in self.save_dir:
        #print('{} of them are good'.format(len(good)))
        src_pts = np.float32([ kp_ref[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp_cand[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        
        # Remove outliers
        for i in range(len(self.outlier_thresholds)):
            if len(src_pts) < self.min_matches:
                #if 'inline' in self.save_dir:
                #print('Unable to find transform: too few matches!')
                return None, 0
            point_pairs = np.concatenate([src_pts, dst_pts], axis=1)
            rot_angle, trans_j, trans_i = self._point_based_matching(point_pairs)
            src_transformed = src_pts[:, 0, :].copy()
            src_transformed[:, 0] = src_pts[:, 0, 0] * np.cos(-rot_angle) + src_pts[:, 0, 1] * np.sin(-rot_angle) + trans_j
            src_transformed[:, 1] = -src_pts[:, 0, 0] * np.sin(-rot_angle) + src_pts[:, 0, 1] * np.cos(-rot_angle) + trans_i
            matching_error = np.sqrt((src_transformed[:, 0] - dst_pts[:, 0, 0]) ** 2 + (src_transformed[:, 1] - dst_pts[:, 0, 1]) ** 2)
            #if 'inline' in self.save_dir:
            #    print(matching_error)
            if matching_error.max() < self.outlier_thresholds[-1]:
                break
            #if 'inline' in self.save_dir:
            #print('Number of outliers:', (matching_error > self.outlier_thresholds[i]).sum())
            src_pts = src_pts[matching_error < self.outlier_thresholds[i]]
            dst_pts = dst_pts[matching_error < self.outlier_thresholds[i]]
        
        # Calculate transformation matrix for input point clouds
        rot_angle, trans_j, trans_i = self._point_based_matching(point_pairs)
        plus8 = np.eye(4)
        plus8[0, 3] = self.max_range + 1
        plus8[1, 3] = self.max_range + 1
        minus8 = np.eye(4)
        minus8[0, 3] = -self.max_range - 1
        minus8[1, 3] = -self.max_range - 1
        tf_matrix = np.array([
            [np.cos(rot_angle), np.sin(rot_angle), 0, trans_i * self.grid_size],
            [-np.sin(rot_angle), np.cos(rot_angle), 0, trans_j * self.grid_size],
            [0,                  0,                 1, 0],
            [0,                  0,                 0, 1]
        ])
        tf_matrix = minus8 @ tf_matrix @ plus8
        return tf_matrix, self.get_fitness(ref_cloud_numpy, cand_cloud_numpy, tf_matrix, save_dir)