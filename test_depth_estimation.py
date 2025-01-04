import sys
sys.path.append('/home/docker_opr/OpenPlaceRecognition/OpenPlaceRecognition/third_party/AdelaiDepth/LeReS/Minist_Test')
from lib.multi_depth_model_woauxi import RelDepthModel
from lib.net_tools import load_ckpt

from opr.pipelines.depth_estimation import DepthEstimationPipeline
from skimage.io import imread
import numpy as np
from scipy.spatial.transform import Rotation
import time

def parse_args(a):
    parser = argparse.ArgumentParser(
        description='Configs for LeReS')
    parser.add_argument('--load_ckpt', default='./res50.pth', help='Checkpoint path to load')
    parser.add_argument('--backbone', default='resnext101', help='Checkpoint path to load')

    args = parser.parse_args(a)
    return args

import argparse
arguments = "--load_ckpt /home/kirill/AdelaiDepth/weights/res50.pth \
            --backbone resnet50".split()
args = parse_args(arguments)

rel_depth_model = RelDepthModel(backbone='resnet50').cuda()
load_ckpt(args, rel_depth_model, None, None)

test_image = imread('/home/docker_opr/Datasets/test_img.png')
test_cloud = np.fromfile('/home/docker_opr/Datasets/test_cloud.bin', sep=',').reshape((-1, 4))[:, :3]
pipeline = DepthEstimationPipeline(rel_depth_model)

camera_matrix = {'f': 683.6, 'cx': 615.1, 'cy': 345.3}
rotation = [-0.498, 0.498, -0.495, 0.510]
R = Rotation.from_quat(rotation).as_matrix()
translation = np.array([[0.061], [0.049], [-0.131]])
tf_matrix = np.concatenate([R, translation], axis=1)
tf_matrix = np.concatenate([tf_matrix, np.array([[0, 0, 0, 1]])], axis=0)
pipeline.set_camera_matrix(camera_matrix)
pipeline.set_lidar_to_camera_transform(tf_matrix)

pipeline.get_depth_with_lidar(test_image, test_cloud)
pipeline.get_depth_with_lidar(test_image, test_cloud)
