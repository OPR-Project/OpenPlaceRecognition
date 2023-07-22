"""Script for preprocessing NCLT dataset."""
# flake8: noqa
# TODO: fix all linter problems and add type annotations
import argparse
import concurrent
import re
import struct
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from time import time
from typing import List, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

matplotlib.use("Agg")

DEFAULT_TRACKLIST = [
    "2012-01-08",
    "2012-01-22",
    "2012-02-12",
    "2012-02-18",
    "2012-03-31",
    "2012-05-26",
    "2012-08-04",
    "2012-10-28",
    "2012-11-04",
    "2012-12-01",
]


class Undistort:
    """An object for undistorting images.

    Note:
        Source: code is taken from the sample scripts for the original dataset.
        URL: http://robots.engin.umich.edu/nclt/index.html
    """

    def __init__(self, map_filepath: Path):
        self.fin = map_filepath
        # read distortion maps
        with open(map_filepath, "r") as f:
            header = f.readline().rstrip()
            chunks = re.sub(r"[^0-9,]", "", header).split(",")
            self.mapu = np.zeros((int(chunks[1]), int(chunks[0])), dtype=np.float32)
            self.mapv = np.zeros((int(chunks[1]), int(chunks[0])), dtype=np.float32)
            for line in f.readlines():
                chunks = line.rstrip().split(" ")
                self.mapu[int(chunks[0]), int(chunks[1])] = float(chunks[3])
                self.mapv[int(chunks[0]), int(chunks[1])] = float(chunks[2])
        # generate a mask
        self.mask = np.ones(self.mapu.shape, dtype=np.uint8)
        self.mask = cv2.remap(self.mask, self.mapu, self.mapv, cv2.INTER_LINEAR)
        kernel = np.ones((30, 30), np.uint8)
        self.mask = cv2.erode(self.mask, kernel, iterations=1)

    def __call__(self, img):
        return cv2.resize(
            cv2.remap(img, self.mapu, self.mapv, cv2.INTER_LINEAR),
            (self.mask.shape[1], self.mask.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )


def parse_args() -> Tuple[Path, Path, bool, int]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root", required=True, type=Path, help="The path to the NCLT dataset root directory."
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="The path to the directory where processed data will be stored.",
    )
    parser.add_argument(
        "--save_large",
        action="store_true",
        required=False,
        default=False,
        help="Whether to save large-resolution images.",
    )
    parser.add_argument(
        "--num_threads",
        required=False,
        type=int,
        default=1,
        help="Number of threads to use.",
    )
    args = parser.parse_args()

    dataset_root: Path = args.dataset_root
    if not dataset_root.exists():
        raise ValueError("Given dataset_root directory does not exist.")
    actual_subdirs = [d.name for d in dataset_root.iterdir() if d.is_dir()]
    required_subdirs = ["images", "velodyne_data", "ground_truth", "undistort_maps"]
    missing_subdirs = [subdir for subdir in required_subdirs if subdir not in actual_subdirs]
    if len(missing_subdirs) > 0:
        raise ValueError(f"Missing some of the required subdirectories in dataset_root: {missing_subdirs}")

    output_dir: Path = args.output_dir
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        print(f"Created output directory: {output_dir}")
    else:
        print(f"Given output directory {output_dir} already exists.")

    save_large: bool = args.save_large

    num_threads: int = args.num_threads

    return dataset_root, output_dir, save_large, num_threads


def get_gt_files_list(dataset_root: Path, track_list: List[str]) -> List[Path]:
    """Get list of paths to the csv files with ground-truth poses.

    Args:
        dataset_root (Path): The root directory of NCLT dataset.
        track_list (List[str]): List of tracks that should be included.

    Raises:
        ValueError: If some of the files for the given track list are missing.

    Returns:
        List[Path]: List of paths to the csv files with ground-truth poses sorted by track name.
    """
    gt_dir = dataset_root / "ground_truth"
    gt_files = [f for f in gt_dir.iterdir() if f.suffix == ".csv"]
    gt_files = [f for f in gt_files if f.stem.split("_")[-1] in track_list]
    found_tracks = [f.stem.split("_")[-1] for f in gt_files]
    missing_tracks = [track for track in track_list if track not in found_tracks]
    if len(missing_tracks) > 0:
        raise ValueError(
            f"Missing ground_truth files for some of the tracks from the track_list: {missing_tracks}"
        )
    return sorted(gt_files)


def get_images_dirs_list(dataset_root: Path, track_list: List[str]) -> List[Path]:
    """Get list of directories with images.

    Args:
        dataset_root (Path): The root directory of NCLT dataset.
        track_list (List[str]): List of tracks that should be included.

    Raises:
        ValueError: If some of the directories for the given track list are missing.
        ValueError: If some of the 'CamN' subdirectories are missing.

    Returns:
        List[Path]: List of directories with images sorted by track name.
        There are six sub-directories for six cameras in each of the returned directories.
    """
    images_dir = dataset_root / "images"
    track_dirs = [d for d in images_dir.iterdir() if d.is_dir()]
    track_dirs = [d for d in track_dirs if d.name in track_list]
    found_tracks = [d.name for d in track_dirs]
    missing_tracks = [track for track in track_list if track not in found_tracks]
    if len(missing_tracks) > 0:
        raise ValueError(f"Missing images dirs for some of the tracks from the track_list: {missing_tracks}")
    track_dirs = [d / "lb3" for d in track_dirs]
    for track_dir in track_dirs:
        for i in range(6):
            if f"Cam{i}" not in [d.name for d in track_dir.iterdir()]:
                raise ValueError(f"Missing Cam{i} subdirectory in {track_dir}")
    return sorted(track_dirs)


def get_lidar_dirs_list(dataset_root: Path, track_list: List[str]) -> List[Path]:
    """Get list of directories with lidar point clouds.

    Args:
        dataset_root (Path): The root directory of NCLT dataset.
        track_list (List[str]): List of tracks that should be included.

    Raises:
        ValueError: If some of the directories for the given track list are missing.

    Returns:
        List[Path]: List of directories with lidar point clouds sorted by track name.
    """
    lidar_dir = dataset_root / "velodyne_data"
    track_dirs = [d for d in lidar_dir.iterdir() if d.is_dir()]
    track_dirs = [d for d in track_dirs if d.name in track_list]
    found_tracks = [d.name for d in track_dirs]
    missing_tracks = [track for track in track_list if track not in found_tracks]
    if len(missing_tracks) > 0:
        raise ValueError(f"Missing lidar dirs for some of the tracks from the track_list: {missing_tracks}")
    track_dirs = [d / "velodyne_sync" for d in track_dirs]
    return sorted(track_dirs)


def read_poses_csv(filepath: Path) -> pd.DataFrame:
    """Read csv file with ground-truth poses.

    Args:
        filepath (Path): The path to the file.

    Returns:
        pd.DataFrame: Pandas DataFrame with following columns: `timestamp`, `x`, `y`, `z`,
        `r`, `p`, `h`.
    """
    colnames = ["timestamp", "x", "y", "z", "r", "p", "h"]
    dtypes_dict = {
        "timestamp": np.int64,
        "x": np.float64,
        "y": np.float64,
        "z": np.float64,
        "r": np.float64,
        "p": np.float64,
        "h": np.float64,
    }
    df = pd.read_csv(filepath, header=None, names=colnames, dtype=dtypes_dict, skiprows=3)
    return df


def closest_values_indices(in_array: np.ndarray, from_array: np.ndarray) -> np.ndarray:
    """For each element in the first array find the closest value from the second array.

    Args:
        in_array (np.ndarray): First array.
        from_array (np.ndarray): Second array.

    Returns:
        np.ndarray: Indices of elements from `from_array` that are closest to
        corresponding values in `in_array`.
    """
    closest_idxs = np.zeros(len(in_array), dtype=np.int64)
    for i, a_val in enumerate(in_array):  # memory-optimized version
        abs_diffs = np.abs(from_array - a_val)
        closest_idxs[i] = np.argmin(abs_diffs)
    return closest_idxs


def filter_by_timestamps(
    gt_poses_df: pd.DataFrame,
    images_ts: np.ndarray,
    lidar_ts: np.ndarray,
    time_threshold: int = 10,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Filter the given data by timestamps. Return only corresponding values with closest timestamps.

    Args:
        gt_poses_df (pd.DataFrame): DataFrame with timestamps and poses.
        images_ts (np.ndarray): Image data timestamps.
        lidar_ts (np.ndarray): Lidar data timestamps.
        time_threshold (int, optional): The maximum allowed difference between timestamps (milliseconds).
        Defaults to 10.

    Returns:
        Tuple[pd.DataFrame, np.ndarray, np.ndarray]: Filtered values: i-th timestamp in any array
        is not farther than `threshold` ms from i-th timestamp in any other array.
    """
    time_threshold *= 1000  # ms -> microseconds
    poses_ts = gt_poses_df["timestamp"].to_numpy(dtype=np.int64)
    poses_indices = closest_values_indices(in_array=images_ts, from_array=poses_ts)
    gt_poses_df = gt_poses_df.iloc[poses_indices]
    poses_ts = poses_ts[poses_indices]
    lidar_ts = lidar_ts[closest_values_indices(in_array=images_ts, from_array=lidar_ts)]
    timestamps = np.hstack([poses_ts[:, np.newaxis], images_ts[:, np.newaxis], lidar_ts[:, np.newaxis]])
    row_indices = []
    for i, row in enumerate(timestamps):
        row_sorted = np.sort(row)
        if np.max(np.diff(row_sorted)) <= time_threshold:
            row_indices.append(i)

    return gt_poses_df.iloc[row_indices], images_ts[row_indices], lidar_ts[row_indices]


def filter_by_distance_indices(utm_points: np.ndarray, distance: float = 5.0) -> np.ndarray:
    """Filter points so that each point is approximatly `distance` meters away from the previous.

    Args:
        utm_points (np.ndarray): The array of UTM coordinates.
        distance (float): The desirable distance between points. Defaults to 5.0.

    Returns:
        np.ndarray: The indices of the filtered points.
    """
    filtered_points = np.array([0], dtype=int)  # start with the index of the first point
    for i in range(1, utm_points.shape[0]):
        # calculate the Euclidean distance between the current and previous point
        right_dist = np.linalg.norm(utm_points[i] - utm_points[filtered_points[-1]])
        if right_dist >= distance:  # we found the point to the right of the 'ideal point'
            left_dist = np.linalg.norm(utm_points[i - 1] - utm_points[filtered_points[-1]])
            if np.abs(right_dist - distance) < np.abs(left_dist - distance):
                filtered_points = np.append(filtered_points, i)  # the point to the right is closer
            else:
                filtered_points = np.append(filtered_points, i - 1)  # the point to the left is closer
    return filtered_points


def plot_track_map(utms: np.ndarray) -> np.ndarray:
    x, y = utms[:, 0], utms[:, 1]
    fig, ax = plt.subplots(dpi=200)
    ax.scatter(y, x, s=0.5, c="blue")
    ax.set_xlabel("y")
    ax.set_xlim(-750, 150)
    ax.set_ylabel("x")
    ax.set_ylim(-380, 140)
    ax.set_aspect("equal", adjustable="box")
    fig.canvas.draw()
    # convert canvas to image
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    # convert from RGB to BGR for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def draw_bbox_center(img: np.ndarray, bbox_size: Tuple[int, int]) -> np.ndarray:
    h, w = img.shape[:2]
    if h < bbox_size[1] or w < bbox_size[0]:
        raise ValueError("Given image is smaller than crop_size")
    center_y, center_x = h // 2, w // 2
    # Calculate the coordinates of the top-left and bottom-right corners of the bounding box
    box_width, box_height = bbox_size
    left = center_x - box_width // 2
    top = center_y - box_height // 2
    right = center_x + box_width // 2
    bottom = center_y + box_height // 2
    # Draw the bounding box in yellow color
    img = cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 255), thickness=2)
    return img


def center_crop(img: np.ndarray, crop_size: Tuple[int, int]) -> np.ndarray:
    h, w = img.shape[:2]
    if h < crop_size[1] or w < crop_size[0]:
        raise ValueError("Given image is smaller than crop_size")
    center_y, center_x = h // 2, w // 2
    left = center_x - crop_size[0] // 2
    top = center_y - crop_size[1] // 2
    right = left + crop_size[0]
    bottom = top + crop_size[1]
    cropped_img = img[top:bottom, left:right]
    return cropped_img


def process_images(
    images_dir: Path,
    output_dir: Path,
    timestamps: np.ndarray,
    undistortion_maps: List[Undistort],
    save_large: bool = False,
    num_threads: int = 4,
) -> None:
    output_small_dir = output_dir / "lb3_small"
    output_small_dir.mkdir(exist_ok=True)
    if save_large:
        output_large_dir = output_dir / "lb3_large"
        output_large_dir.mkdir(exist_ok=True)
    camera_subdirs = sorted([d for d in images_dir.iterdir() if d.is_dir() and d.name.startswith("Cam")])

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for n, camera_subdir in enumerate(tqdm(camera_subdirs, desc="Cameras", position=1, leave=False)):
            output_small_cam_subdir = output_small_dir / camera_subdir.name
            output_small_cam_subdir.mkdir(exist_ok=True)
            if save_large:
                output_large_cam_subdir = output_large_dir / camera_subdir.name
                output_large_cam_subdir.mkdir(exist_ok=True)

            futures = []
            for timestamp in timestamps:
                img_filepath = camera_subdir / f"{timestamp}.tiff"
                out_small_filepath = output_small_cam_subdir / f"{timestamp}.png"
                future = executor.submit(
                    process_image,
                    img_filepath,
                    undistortion_maps[n],
                    out_small_filepath,
                    save_large,
                    output_large_cam_subdir,
                    timestamp,
                )
                futures.append(future)
            for _ in tqdm(
                concurrent.futures.as_completed(futures),
                desc=f"Cam {n}",
                total=len(futures),
                position=2,
                leave=False,
            ):
                pass


def process_image(
    img_filepath: Path,
    undistortion_map: Undistort,
    out_small_filepath: Path,
    save_large: bool,
    output_large_cam_subdir: Path,
    timestamp: int,
):
    img = cv2.imread(str(img_filepath))
    img = undistortion_map(img)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = center_crop(img, (768, 960))
    if save_large:
        out_large_filepath = output_large_cam_subdir / f"{timestamp}.png"
        cv2.imwrite(str(out_large_filepath), img)
    img = cv2.resize(img, (256, 320), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(out_small_filepath), img)


def load_src_bin(filepath: Path) -> np.ndarray:
    hits = []
    scaling = np.float32(0.005)  # 5 mm
    offset = np.float32(-100.0)
    with open(filepath, "rb") as f_bin:
        while True:
            x_str = f_bin.read(2)
            if x_str == b"":  # eof
                break
            x = struct.unpack("<H", x_str)[0]
            y = struct.unpack("<H", f_bin.read(2))[0]
            z = struct.unpack("<H", f_bin.read(2))[0]
            i = struct.unpack("B", f_bin.read(1))[0]  # intensity
            _ = struct.unpack("B", f_bin.read(1))[0]  # laser id

            hits += [[x, y, z, i]]
    pc = np.array(hits, dtype=np.float32)
    pc[:, :3] = pc[:, :3] * scaling + offset
    return pc


def process_lidar(lidar_dir: Path, output_dir: Path, timestamps: np.ndarray, num_threads: int = 1) -> None:
    output_dir = output_dir / "velodyne_data"
    output_dir.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for timestamp in timestamps:
            src_filepath = lidar_dir / f"{timestamp}.bin"
            out_filepath = output_dir / f"{timestamp}.bin"
            future = executor.submit(process_lidar_file, src_filepath, out_filepath)
            futures.append(future)
        for _ in tqdm(
            concurrent.futures.as_completed(futures),
            desc="Lidar data",
            total=len(futures),
            position=1,
            leave=False,
        ):
            pass


def process_lidar_file(src_filepath: Path, out_filepath: Path) -> None:
    pc = load_src_bin(src_filepath)
    pc.tofile(out_filepath)


def preprocess_nclt(
    dataset_root: Path,
    output_dir: Path,
    track_list: List[str] = DEFAULT_TRACKLIST,
    save_large: bool = False,
    num_threads: int = 1,
) -> None:
    gt_files = get_gt_files_list(dataset_root, track_list)
    images_dirs = get_images_dirs_list(dataset_root, track_list)
    lidar_dirs = get_lidar_dirs_list(dataset_root, track_list)

    undistortion_maps = [
        Undistort(map_filepath=(dataset_root / "undistort_maps" / f"U2D_Cam{i}_1616X1232.txt"))
        for i in range(6)
    ]

    for gt_filepath, images_dir, lidar_dir in tqdm(
        zip(gt_files, images_dirs, lidar_dirs),
        desc="Processing tracks",
        position=0,
        leave=False,
        total=len(gt_files),
    ):
        cur_track_name = gt_filepath.stem.split("_")[-1]
        gt_poses_df = read_poses_csv(gt_filepath)
        images_timestamps = np.sort(
            np.array([np.int64(f.stem) for f in (images_dir / "Cam0").iterdir()], dtype=np.int64)
        )
        images_timestamps = images_timestamps[30:]  # remove some static frames from the beginning
        lidar_timestamps = np.sort(np.array([np.int64(f.stem) for f in lidar_dir.iterdir()], dtype=np.int64))
        gt_poses_df, images_timestamps, lidar_timestamps = filter_by_timestamps(
            gt_poses_df, images_timestamps, lidar_timestamps
        )

        utms = gt_poses_df[["x", "y"]].to_numpy(dtype=np.float64)
        filtered_indices = filter_by_distance_indices(utms, distance=5.0)
        gt_poses_df = gt_poses_df.iloc[filtered_indices]
        images_timestamps = images_timestamps[filtered_indices]
        gt_poses_df["image"] = images_timestamps
        lidar_timestamps = lidar_timestamps[filtered_indices]
        gt_poses_df["pointcloud"] = lidar_timestamps
        new_colnames = ["timestamp", "image", "pointcloud", "x", "y", "z", "r", "p", "h"]
        gt_poses_df = gt_poses_df.reset_index()[new_colnames]

        track_output_dir = output_dir / cur_track_name
        track_output_dir.mkdir(exist_ok=True)
        gt_poses_df.to_csv(track_output_dir / "track.csv")

        track_map_img = plot_track_map(gt_poses_df[["x", "y"]].to_numpy())
        cv2.imwrite(str(track_output_dir / "map.png"), track_map_img)

        process_images(
            images_dir,
            track_output_dir,
            images_timestamps,
            undistortion_maps,
            save_large=save_large,
            num_threads=num_threads,
        )
        process_lidar(lidar_dir, track_output_dir, lidar_timestamps, num_threads=num_threads)


if __name__ == "__main__":
    dataset_root, output_dir, save_large, num_threads = parse_args()

    start_time = time()
    preprocess_nclt(
        dataset_root=dataset_root, output_dir=output_dir, save_large=save_large, num_threads=num_threads
    )
    end_time = time()
    print()
    print(f"Time taken: {timedelta(seconds=(end_time - start_time))}")
