"""Custom SberRobotics dataset implementations."""
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union, Set
import cv2
import json

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R

from opr.datasets.augmentations import (
    DefaultImageTransform,
)


class SberRobotics(Dataset):
    """SberRobotics dataset implementation."""


    dataset_folder: Path
    cam_type: Literal["regular", "fisheye"]
    cam_id: int
    image_transform: DefaultImageTransform
    load_depth_images: bool
    load_bboxes: bool
    depth_images_resize: Tuple[int, int]
    timestamps: List[str]
    dataset_df: pd.DataFrame


    def __init__(
        self,
        dataset_root: Union[Path, str],
        cam_type: Literal["regular", "fisheye"] = "regular",
        image_transform = DefaultImageTransform(resize=(322, 322), train=False),
        load_depth_images: bool = False,
        load_bboxes: bool = False,
        depth_images_resize: Tuple[int, int] = (322, 322),
        use_test_dataset: bool = False
    ) -> None:
        """Initialize the SberRobotics dataset.

        Args:
            dataset_root (Union[Path, str]): Path to the dataset root directory.
            cam_type (Literal["regular", "fisheye"]): What type of camera to use. Defaults to 'regular'.
            image_transform: Image transform to apply to images. Defaults to DefaultImageTransform.
            load_depth_images (bool): Whether to load depth images. Defaults to False.
            load_depth_bboxes (bool): Whether to load bounding boxes. Defaults to False.
            depth_images_resize (Tuple[int, int]): Resize dimensions for depth images. Defaults to (322, 322).
            use_test_dataset (bool): Whether to use the test folder. Defaults to False.

        Raises:
            FileNotFoundError: If dataset_root doesn't exist.
        """

        super().__init__()
        dataset_root = Path(dataset_root)
        if not dataset_root.exists():
            raise FileNotFoundError(f"Given dataset_root={dataset_root} doesn't exist")
        
        if use_test_dataset:
            self.dataset_folder = dataset_root / "mmpr_dataset_test/mav0"
        else:
            self.dataset_folder = dataset_root / "mmpr_dataset_1/mav0"

        self.load_depth_images = load_depth_images
        self.load_bboxes = load_bboxes
        self.depth_images_resize = depth_images_resize
        self.cam_type = cam_type
        if cam_type == "fisheye":
            self.cam_id = 2
        else:
            self.cam_id = 0
        self.image_transform = image_transform

        if cam_type == "regular":
            self.dataset_df = pd.read_csv(self.dataset_folder / "track.csv")
        else:
            self.dataset_df = pd.read_csv(self.dataset_folder / "database_fisheye_track.csv")
        self.dataset_df["in_query"] = True # To change when subsets added

        self.timestamps = list(self.dataset_df["timestamp"])
    

    def generate_dataframe(self) -> pd.DataFrame:
        """Generate a DataFrame with poses for dataset_df.
        
        Returns:
            pd.DataFrame: DataFrame containing timestamp, poses data in tx, ty, tz, qx, qy, qz, qw and track number in track.
        """

        timestamps_with_all_data: Set[int] = set()
        timestamp_pose: Dict[int, List[float]] = {}
        with open(self.dataset_folder / "traj.txt", "r") as f:
            for line in f.readlines():
                desc = line.strip().split()
                if len(desc) > 0:
                    timestamp = int(float(desc[0]))
                    timestamps_with_all_data.add(timestamp)
                    timestamp_pose[timestamp] = list(map(float, desc[1:]))

        cam_timestamps: Set[int] = set()

        data_csv = pd.read_csv(self.dataset_folder / f"cam{self.cam_id}/data.csv", header=None)
        for timestamp in data_csv[0]:
            cam_timestamps.add(int(timestamp))
        
        timestamps_with_all_data &= cam_timestamps

        timestamps = list(timestamps_with_all_data)

        SECOND_TRACK_START: float = 1846952165000

        data: List[Dict[str, int]] = []

        for timestamp in timestamps:
            if timestamp not in timestamp_pose or len(timestamp_pose[timestamp]) != 7:
                raise ValueError(f"Incorrect pose in traj.txt for timestamp {timestamp}")
            
            tx, ty, tz, qx, qy, qz, qw = timestamp_pose[timestamp]
            data.append({
                "timestamp": timestamp,
                "tx": tx,
                "ty": ty,
                "tz": tz,
                "qx": qx,
                "qy": qy,
                "qz": qz,
                "qw": qw,
                "track": 0 if timestamp < SECOND_TRACK_START else 1
            })
        
        df = pd.DataFrame(data)

        return df.sort_values("timestamp")
    
    def __getitem__(self, idx: int) -> Dict[str, Union[Tensor, int]]:
        """Get item by index.

        Args:
            idx (int): Index of the item to get.

        Returns:
            Dict[str, Union[Tensor, int]]: Dictionary containing the item data.
                Contains:
                    - "image_front_cam{id}": Image tensors
                    - "pose": 6DOF pose tensor
                    - "idx": Index of the item
                    - "depth_image_front_cam": Depth image tensor (if load_depth_images is True)
        """

        data: Dict[str, Union[int, Tensor]] = {"idx": torch.tensor(idx)}

        data["pose"] = self._pose_to_6dof(idx)

        data[f"image_front_cam{self.cam_id}"] = self._load_image(idx, self.cam_id, transform=(self.image_transform is not None))

        if self.load_depth_images:
            data["depth_image_front_cam"] = self._load_depth_image(idx)

        if self.load_bboxes:
            data[f"bounding_box_front_cam{self.cam_id}"] = self._load_bboxes(idx, self.cam_id)

        return data


    def __len__(self) -> int:
        """Return the number of elements in the dataset."""

        return len(self.timestamps)
    

    def _pose_to_6dof(self, idx: int) -> Tensor:
        """Convert a pose from dataset_df to a 6DOF vector.
        
            Args:
                idx (int): Index of the item to convert.

            Returns:
                Tensor: 6DOF vector containing translation and rotation.
        """

        # Retrieve pose from the dataset_df
        pose_row = self.dataset_df.iloc[idx]
        tx, ty, tz = pose_row["tx"], pose_row["ty"], pose_row["tz"]
        qx, qy, qz, qw = pose_row["qx"], pose_row["qy"], pose_row["qz"], pose_row["qw"]

        # Convert quaternion to Euler angles (roll, pitch, yaw)
        rotation = R.from_quat([qx, qy, qz, qw])
        roll, pitch, yaw = rotation.as_euler('xyz', degrees=False)  # Euler angles in radians

        # Convert pose to tensor
        pose = torch.tensor([tx, ty, tz, roll, pitch, yaw], dtype=torch.float32)
        return pose
    

    def _load_image(self, idx: int, cam_id: int, transform: bool = True) -> Tensor:
        """Load an image from the dataset.

        Args:
            idx (int): Index.
            cam_id (int): Camera ID.
            transform (bool): Whether to apply the image transform. Defaults to True.

        Returns:
            Tensor: Loaded image tensor.
        """

        image_name = str(self.timestamps[idx]).rjust(18, '0')
        image_path = self.dataset_folder / f"cam{cam_id}/data/{image_name}.png"
        im = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if transform:
            im = self.image_transform(im)
        else:
            im = torch.from_numpy(im).float()
        return im
    

    def _load_bboxes(self, idx: int, cam_id: int) -> Tensor:
        """Load bounding boxes of an image from the dataset.

        Args:
            idx (int): Index.
            cam_id (int): Camera ID.

        Returns:
            Tensor: Loaded tensor with bounding box.
        """

        bbox_name = str(self.timestamps[idx]).rjust(18, '0')
        bbox_path = self.dataset_folder / f"cam{cam_id}/bboxes/{bbox_name}.bboxes.txt"
        
        with open(bbox_path, "r") as f:
            bboxes = [Tensor([float(coord) for coord in bbox.strip().split()]) for bbox in f.readlines()]
            if len(bboxes) != 0:
                bboxes = torch.stack(bboxes)
            else:
                bboxes = torch.empty((0, 4), dtype=torch.float32)

        return bboxes
    

    def _load_depth_image(self, idx: int) -> Tensor:
        """Load a depth image from the dataset.

        Args:
            idx (int): Frame index.

        Returns:
            Tensor: Loaded depth image tensor.
        """

        image_name = str(self.timestamps[idx]).rjust(18, "0")
        image_path = self.dataset_folder / f"depth/data/{image_name}.npy"
        im = np.load(image_path)
        im = cv2.resize(im, self.depth_images_resize, interpolation=cv2.INTER_NEAREST)
        return im


    def collate_fn(self, data_list: List[Dict[str, Tensor]]) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """Pack input data list into batch.

        Args:
            data_list (List[Dict[str, Tensor]]): batch data list generated by DataLoader.

        Returns:
            Dict[str, Tensor]: dictionary of batched data.
        """

        batch = {}
        for key in data_list[0].keys():
            if key == "idx":
                batch["idxs"] = torch.stack([data[key] for data in data_list])
            elif key == "pose":
                batch["poses"] = torch.stack([data[key] for data in data_list], dim=0)
            elif key.startswith("image"):
                batch["images" + key[5:]] = torch.stack([data[key] for data in data_list], dim=0)
            elif key == "depth_image_front_cam":
                batch["depth_images_front_cam"] = torch.stack([data[key] for data in data_list], dim=0)
            elif key.startswith("bounding_box"):
                batch[f"bounding_boxes{key[12:]}"] = [data[key] for data in data_list]

        return batch
