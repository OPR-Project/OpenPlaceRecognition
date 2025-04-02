import numpy as np

def rotate_pcd(points, rotation_matrix):
    points_xyz = points[:, :3]
    #rotation_matrix = Rotation.from_rotvec(rotation_vector).as_matrix()
    points_xyz_rotated = points_xyz @ rotation_matrix
    points_rotated = points.copy()
    points_rotated[:, :3] = points_xyz_rotated
    return points_rotated

def transform_pcd(points, x, y, theta):
    points_transformed = points.copy()
    points_transformed[:, 0] = points[:, 0] * np.cos(theta) + points[:, 1] * np.sin(theta)
    points_transformed[:, 1] = -points[:, 0] * np.sin(theta) + points[:, 1] * np.cos(theta)
    points_transformed[:, 0] += x
    points_transformed[:, 1] += y
    return points_transformed

def normalize(angle):
    while angle < -np.pi:
        angle += 2 * np.pi
    while angle > np.pi:
        angle -= 2 * np.pi
    return angle

def rotate(x, y, angle):
    x_new = x * np.cos(angle) + y * np.sin(angle)
    y_new = -x * np.sin(angle) + y * np.cos(angle)
    return x_new, y_new

def remove_floor_and_ceil(cloud, floor_height=-0.9, ceil_height=1.5):
    heights = np.linspace(-4.0, 4.0, 41)
    floor_index = None
    if floor_height == 'auto':
        bins = []
        for i, height in enumerate(heights[:-1]):
            bins.append(len(cloud[(cloud[:, 2] > height) * (cloud[:, 2] < heights[i + 1])]))
        #print('Bins:', bins)
        floor_index = np.argmax(bins[:20]) + 1
        floor_height = heights[floor_index]
        assert floor_index < len(heights) - 5
    if ceil_height == 'auto':
        if floor_index is None:
            floor_index = 0
            while floor_index < len(heights) - 6 and heights[floor_index] < floor_height:
                floor_index += 1
        ceil_index = floor_index + 5 + np.argmax(bins[floor_index + 5:])
        ceil_height = heights[ceil_index]
    #print('Floor height:', floor_height)
    #print('Ceil height:', ceil_height)
    return cloud[(cloud[:, 2] > floor_height) * (cloud[:, 2] < ceil_height)]

def get_rel_pose(x, y, theta, x2, y2, theta2):
    rel_x, rel_y = rotate(x2 - x, y2 - y, theta)
    return [rel_x, rel_y, normalize(theta2 - theta)]

def apply_pose_shift(pose, rel_x, rel_y, rel_theta):
    x, y, theta = pose
    new_x = x + rel_x * np.cos(-theta) + rel_y * np.sin(-theta)
    new_y = y - rel_x * np.sin(-theta) + rel_y * np.cos(-theta)
    new_theta = theta + rel_theta
    return [new_x, new_y, new_theta]

def rotate_vertical(cloud, angle):
    cloud_rotated = cloud.copy()
    cloud_rotated[:, 0] = cloud[:, 0] * np.cos(angle) + cloud[:, 2] * np.sin(angle)
    cloud_rotated[:, 2] = -cloud[:, 0] * np.sin(angle) + cloud[:, 2] * np.cos(angle)
    return cloud_rotated