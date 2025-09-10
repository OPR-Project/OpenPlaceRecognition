"""Occupancy grid registration pipeline."""
import os
from skimage.io import imsave
from os import PathLike
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
import time
from torch import Tensor, nn
import faiss

class Feature2DGlobalRegistrationPipeline:
    """Occupancy grid registration pipeline using extracted features` 2d projection"""
    def __init__(self, 
                 voxel_downsample_size: float = 0.1,
                 detector_type: str = 'ORB',
                 n_keypoints: int = 400,
                 outlier_thresholds: List[float] = [5.0],
                 min_matches: int = 5,
                 save_dir: Union[str, None] = None) -> None:
        self.grid_size = voxel_downsample_size
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

    def _raycast(self, grid: np.ndarray, n_rays: int = 1000, radius: int = 80, center_point: Union[Tuple[float, float], None] = None) -> np.ndarray:
        grid_cropped = grid[grid.shape[0] // 2 - radius:grid.shape[0] // 2 + radius,
                              grid.shape[0] // 2 - radius:grid.shape[0] // 2 + radius]
        grid_raycasted = grid_cropped.copy()
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

    def _transform_grid(self, grid: np.ndarray, transform: np.ndarray):
        i, j = (grid == 1).nonzero()
        trans_i, trans_j, rot_angle = transform
        i_transformed = i * np.cos(rot_angle) + j * np.sin(rot_angle) + trans_i
        j_transformed = -i * np.sin(rot_angle) + j * np.cos(rot_angle) + trans_j
        ij_transformed = np.concatenate([i_transformed[:, np.newaxis], j_transformed[:, np.newaxis]], axis=1)
        ij_transformed = ij_transformed[(ij_transformed[:, 0] >= 0) * (ij_transformed[:, 0] < grid.shape[0]) * (ij_transformed[:, 1] >= 0) * (ij_transformed[:, 1] < grid.shape[1])]
        ij_transformed = ij_transformed.astype(int)
        grid_transformed = np.zeros_like(grid)
        grid_transformed[ij_transformed[:, 0], ij_transformed[:, 1]] = 1
        i, j = (grid == 2).nonzero()
        trans_i, trans_j, rot_angle = transform
        i_transformed = i * np.cos(rot_angle) + j * np.sin(rot_angle) + trans_i
        j_transformed = -i * np.sin(rot_angle) + j * np.cos(rot_angle) + trans_j
        ij_transformed = np.concatenate([i_transformed[:, np.newaxis], j_transformed[:, np.newaxis]], axis=1)
        ij_transformed = ij_transformed[(ij_transformed[:, 0] >= 0) * (ij_transformed[:, 0] < grid.shape[0]) * (ij_transformed[:, 1] >= 0) * (ij_transformed[:, 1] < grid.shape[1])]
        ij_transformed = ij_transformed.astype(int)
        grid_transformed[ij_transformed[:, 0], ij_transformed[:, 1]] = 2
        return grid_transformed

    def get_fitness(self, ref_grid: np.ndarray, cand_grid: np.ndarray, transform: np.ndarray) -> float:
        kernel = np.ones((7, 7), dtype=np.uint8)
        #print(ref_grid.shape, ref_grid.dtype)
        #print(ref_wall_mask.shape, ref_wall_mask.dtype)
        ref_grid_transformed = self._transform_grid(ref_grid, transform)
        ref_wall_mask = (ref_grid_transformed == 2).astype(np.uint8)
        ref_wall_mask_dilated = cv2.dilate(ref_wall_mask, kernel)
        cand_wall_mask = (cand_grid == 2).astype(np.uint8)
        cand_wall_mask_dilated = cv2.dilate(cand_wall_mask, kernel)
        ref_floor_mask = (ref_grid_transformed == 1).astype(np.uint8)
        #ref_floor_mask = self._raycast(ref_wall_mask) - ref_wall_mask + ref_floor_mask
        #ref_floor_mask = np.clip(ref_floor_mask, 0, 1)
        cand_floor_mask = (cand_grid == 1).astype(np.uint8)
        trans_i, trans_j, rot_angle = transform
        #cand_floor_mask = self._raycast(cand_wall_mask) - cand_wall_mask + cand_floor_mask
        #cand_floor_mask = np.clip(cand_floor_mask, 0, 1)
        intersection = np.sum(np.clip(ref_wall_mask + ref_floor_mask, 0, 1) * np.clip(cand_wall_mask + cand_floor_mask, 0, 1))
        union = np.sum(np.clip(ref_wall_mask + ref_floor_mask + cand_wall_mask + cand_floor_mask, 0, 1))
        #print('Intersection:', intersection)
        #print('Union:', union)
        iou = intersection / union
        #print('Transform:', tf_matrix[0, 3], tf_matrix[1, 3], rot_angle)
        #print('Iou from get_fitness:', iou)
        good_match = np.sum(cand_wall_mask_dilated * ref_wall_mask)
        bad_match = np.sum(cand_floor_mask * (1 - cand_wall_mask_dilated) * ref_wall_mask) + \
                    np.sum(ref_floor_mask * (1 - ref_wall_mask_dilated) * cand_wall_mask)
        if self.save_dir is not None:
            save_dir = os.path.join(self.save_dir, str(self.cnt))
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            np.savetxt(os.path.join(save_dir, 'transform.txt'), transform)
            imsave(os.path.join(save_dir, 'ref_wall_mask.png'), (ref_wall_mask * 255).astype(np.uint8))
            imsave(os.path.join(save_dir, 'cand_wall_mask.png'), (cand_wall_mask * 255).astype(np.uint8))
            imsave(os.path.join(save_dir, 'ref_floor_mask.png'), (ref_floor_mask * 255).astype(np.uint8))
            imsave(os.path.join(save_dir, 'cand_floor_mask.png'), (cand_floor_mask * 255).astype(np.uint8))
            grid_aligned = np.zeros((ref_wall_mask.shape[0], ref_wall_mask.shape[1], 3))
            grid_aligned[:, :, 0] = ref_floor_mask + ref_wall_mask
            grid_aligned[:, :, 1] = cand_floor_mask + cand_wall_mask
            imsave(os.path.join(save_dir, 'grid_aligned.png'), (grid_aligned * 255).astype(np.uint8))
            np.savetxt(os.path.join(save_dir, 'results.txt'), 
                    [good_match, bad_match, iou, good_match / (good_match + bad_match) * iou ** 0.25])
            # print('cnt:', self.cnt)
        self.cnt += 1
        # print('Reg score from fitness:', good_match / (good_match + bad_match) * iou ** 0.25)
        return good_match / (good_match + bad_match) * iou ** 0.25#, ref_wall_mask, ref_floor_mask, cand_wall_mask, cand_floor_mask

    def infer(self, ref_grid: Tensor, cand_grid: Tensor, verbose: bool = False) -> Tuple[Union[np.ndarray, None], Union[float, None]]:
    # def infer(self, img_ref, img_cand, verbose=True):
        # Convert clouds from tensors to numpy arrays
        # t1 = time.time()
        ref_grid_numpy = ref_grid.cpu().numpy()
        cand_grid_numpy = cand_grid.cpu().numpy()
        img_ref = (ref_grid_numpy == 2)
        img_ref = (img_ref * 255).astype(np.uint8)
        kernel = np.ones((2, 2))
        img_ref = cv2.dilate(img_ref, kernel)
        img_ref = cv2.GaussianBlur(img_ref, (3, 3), 0.5)
        #img_ref = cv2.resize(img_ref, None, fx=0.5, fy=0.5)
        #print('Img ref min and max:', img_ref.min(), img_ref.mean(), img_ref.max())
        img_cand = (cand_grid_numpy == 2)
        img_cand = (img_cand * 255).astype(np.uint8)
        img_cand = cv2.dilate(img_cand, kernel)
        img_cand = cv2.GaussianBlur(img_cand, (3, 3), 0.5)
        # if self.detector_type == 'HarrisWithDistance':
        #     print('Ref grid:', img_ref.shape, img_ref.min(), img_ref.mean(), img_ref.max())
        #     print('Cand grid:', img_cand.shape, img_cand.min(), img_cand.mean(), img_cand.max())
        #img_cand = cv2.resize(img_cand, None, fx=0.5, fy=0.5)
        #print('Img cand min and max:', img_cand.min(), img_cand.mean(), img_cand.max())
        if self.save_dir is not None:
            save_dir = os.path.join(self.save_dir, str(self.cnt))
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            imsave(os.path.join(save_dir, 'ref_grid.png'), img_ref)
            imsave(os.path.join(save_dir, 'cand_grid.png'), img_cand)
        # t2 = time.time()
        # if self.detector_type == 'HarrisWithDistance':
        #     print('Preprocessing time:', t2 - t1)
        
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
            distance_coef = 0.5
            if len(kp_ref) == 0 or len(kp_cand) == 0:
                if verbose:
                    print('                No kp!')
                self.cnt += 1
                return None, 0
            xy_ref = np.array([kp.pt for kp in kp_ref]) * distance_coef
            des_ref = np.concatenate([des_ref, xy_ref], axis=1).astype(np.uint8)
            xy_cand = np.array([kp.pt for kp in kp_cand]) * distance_coef
            des_cand = np.concatenate([des_cand, xy_cand], axis=1).astype(np.uint8)
        #print(kp_ref, des_ref)
        #print(kp_cand, des_cand)
        # t3 = time.time()
        # if self.detector_type == 'HarrisWithDistance':
        #     print('Found {} keypoints for ref and {} for cand'.format(len(kp_ref), len(kp_cand)))
        #     print('Keypoints estimation time:', t3 - t2)
        
        # Match features using KNN
        if self.detector_type == 'SIFT':
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            flann = cv2.FlannBasedMatcher(index_params)
            if des_ref is None or des_cand is None or len(des_ref) < 2 or len(des_cand) < 2:
                if verbose:
                    print('Too few keypoints! Unable to match')
                self.cnt += 1
                return None, 0
            matches = flann.knnMatch(des_ref, des_cand, k=2)
            matches = [x for x in matches if len(x) == 2]
        elif self.detector_type == 'ORB' or self.detector_type == 'HarrisWithDistance':
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm = FLANN_INDEX_LSH,
                            table_number = 6, # 12
                            key_size = 12,     # 20
                            multi_probe_level = 1) #2
            search_params = dict(checks=50)   # or pass empty dictionary
            flann = cv2.FlannBasedMatcher(index_params,search_params)
            if des_ref is None or des_cand is None or len(des_ref) < 2 or len(des_cand) < 2:
                if verbose:
                    print('Too few keypoints! Unable to match')
                self.cnt += 1
                return None, 0
            matches = flann.knnMatch(des_ref, des_cand, k=2)
            matches = [x for x in matches if len(x) == 2]
        # elif self.detector_type == 'HarrisWithDistance':
            # nlist = 20
            # k = 4
            # d = 34
            # quantizer = faiss.IndexFlatL2(d)  # the other index
            # des_index = faiss.IndexIVFFlat(quantizer, d, nlist)
            # des_index.train(des_cand)
            # des_index = faiss.IndexFlatL2(34)
            # for des in des_cand:
            #     des_index.add(des[np.newaxis, :])
            # matches = []
            # for i, des in enumerate(des_ref):
            #     dist, idx = des_index.search(des[np.newaxis, :], k=2)
            #     m = cv2.DMatch(_imgIdx=0, _queryIdx=i, _trainIdx=idx[0][0], _distance=np.sqrt(dist[0][0]))
            #     n = cv2.DMatch(_imgIdx=0, _queryIdx=i, _trainIdx=idx[0][1], _distance=np.sqrt(dist[0][1]))
            #     matches.append((m, n))
        else:
            print('Incorrect detector type')
            return None, None
        if verbose:
            print('Found {} matches'.format(len(matches)))
        # t4 = time.time()
        # if self.detector_type == 'HarrisWithDistance':
        #     print('Matches estimation time:', t4 - t3)
        
        # Get 2d point sets from matched features
        good = []
        if self.detector_type == 'HarrisWithDistance':
            matching_threshold = 0.8
        else:
            matching_threshold = 0.8
        for i,(m,n) in enumerate(matches):
            if m.distance < matching_threshold * n.distance:
                good.append(m)
        if verbose:
            print('{} of them are good'.format(len(good)))
        src_pts = np.float32([ kp_ref[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp_cand[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        
        # Remove outliers
        for i in range(len(self.outlier_thresholds)):
            if len(src_pts) < self.min_matches:
                if verbose:
                    print('Unable to find transform: too few matches!')
                self.cnt += 1
                return None, 0
            point_pairs = np.concatenate([src_pts, dst_pts], axis=1)
            rot_angle, trans_j, trans_i = self._point_based_matching(point_pairs)
            src_transformed = src_pts[:, 0, :].copy()
            src_transformed[:, 0] = src_pts[:, 0, 0] * np.cos(-rot_angle) + src_pts[:, 0, 1] * np.sin(-rot_angle) + trans_j
            src_transformed[:, 1] = -src_pts[:, 0, 0] * np.sin(-rot_angle) + src_pts[:, 0, 1] * np.cos(-rot_angle) + trans_i
            matching_error = np.sqrt((src_transformed[:, 0] - dst_pts[:, 0, 0]) ** 2 + (src_transformed[:, 1] - dst_pts[:, 0, 1]) ** 2)
            if matching_error.max() < self.outlier_thresholds[-1]:
                break
            if verbose:
                print('Number of outliers:', (matching_error > self.outlier_thresholds[i]).sum())
            src_pts = src_pts[matching_error < self.outlier_thresholds[i]]
            dst_pts = dst_pts[matching_error < self.outlier_thresholds[i]]
        
        # Calculate transformation matrix for input point clouds
        rot_angle, trans_j, trans_i = self._point_based_matching(point_pairs)
        transform = [trans_i, trans_j, rot_angle]
        #print('Trans i, trans j, rot angle:', trans_i, trans_j, rot_angle)
        # t5 = time.time()
        # if self.detector_type == 'HarrisWithDistance':
        #     print('Outlier removing time:', t5 - t4)
        return transform, self.get_fitness(ref_grid_numpy, cand_grid_numpy, transform)