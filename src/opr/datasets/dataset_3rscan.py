"""Custom 3RScan dataset implementations."""
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union, Set
import zipfile
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


class Dataset3RScan(Dataset):
    """3RScan dataset implementation.
    
    Requires downloaded 3RScan repository, executed setup.sh script and downloaded scenes in data/3RScan directory.
    
    There is no pose transformation between different scans for test subset."""


    scenes_folder: Path
    subset: Literal["train", "validation", "test"]
    frame_ids: List[Tuple[str, int]]
    image_transform: DefaultImageTransform
    load_depth_images: bool
    depth_images_resize: Tuple[int, int]
    dataset_df: pd.DataFrame


    def __init__(
        self,
        dataset_root: Union[Path, str],
        subset: Literal["train", "validation", "test"],
        image_transform = DefaultImageTransform(resize=(322, 322), train=False),
        min_scans_cnt: int = 2,
        picture_limit_per_scan: int = 50,
        load_depth_images: bool = False,
        depth_images_resize: Tuple[int, int] = (322, 322)
    ) -> None:
        """3RScan dataset implementation.
        
        Args:
            dataset_root (Union[Path, str]): Root directory of dataset.
            subset (Literal["train", "validation", "test"]): Dataset subset to load.
            image_transform: Image transform to apply to images. Defaults to DefaultImageTransform.
            min_scans_cnt (int): Minimum number of scans in a scene, that are loaded. Defaults to 2.
            picture_limit_per_scan (int): Number of pictures to load from each scan. Defaults to 50.
            load_depth_images (bool): Whether to load depth images. Depth in mm. Defaults to False.
            depth_images_resize (Tuple[int, int]): Resize dimensions for depth images. Defaults to (322, 322).

        Raises:
            FileNotFoundError: If dataset_root doesn't exist.
        """

        super().__init__()
        dataset_root = Path(dataset_root)
        if not dataset_root.exists():
            raise FileNotFoundError(f"Given dataset_root={dataset_root} doesn't exist")

        self.subset = subset

        self.load_depth_images = load_depth_images
        self.depth_images_resize = depth_images_resize

        scene_folder = dataset_root / "data/3RScan"

        # Loading from 3RScan.json all the scene ids and their repeated scans
        with open(scene_folder / f"3RScan.json", "r") as f:
            json_3rscan = f.read()
        json_3rscan = json.loads(json_3rscan)

        scene_ids: Set[str] = set()
        scene_references: Dict[str, str] = {}
        track_numbers: Dict[str, int] = {}
        room_numbers: Dict[str, int] = {}
        transformation_matrices: Dict[str, np.ndarray] = {}

        room_cnt = 0
        for room in json_3rscan:
            if room["type"] != subset:
                continue

            if len(room["scans"]) + 1 < min_scans_cnt:
                continue

            scene_id = room["reference"]
            scene_ids.add(scene_id)
            scene_references[scene_id] = scene_id
            track_numbers[scene_id] = 0
            room_numbers[scene_id] = room_cnt
            room_cnt += 1
            for i, scan in enumerate(room["scans"]):
                scan_id = scan["reference"]
                scene_ids.add(scan_id)
                scene_references[scan_id] = scene_id
                track_numbers[scan_id] = i + 1
                if "transform" in scan:
                    transformation_matrix = np.array(scan["transform"]).reshape((4, 4)).T 
                    transformation_matrix = np.linalg.inv(transformation_matrix) # Transformation in the file in the other direction
                    transformation_matrix[:3, 3] /= 1000 # Transformation in mm, but pose in m
                    transformation_matrices[scan_id] = transformation_matrix
        
        # Saving frames ids
        self.scenes_folder = scene_folder
        self.frame_ids = []
        for scene_path in scene_folder.iterdir():
            if not scene_path.is_dir():
                continue

            scene_id = scene_path.name
            if not scene_id in scene_ids:
                continue

            files_cnt = len(list(scene_path.glob("sequence/*.jpg")))
            for i in range(0, files_cnt, max((files_cnt + picture_limit_per_scan - 1) // picture_limit_per_scan, 1)):
                self.frame_ids.append((scene_id, i))

        self.image_transform = image_transform

        self.dataset_df = self._generate_dataframe(scene_references, track_numbers, room_numbers, transformation_matrices)


    def __getitem__(self, idx: int) -> Dict[str, Union[Tensor, int]]:
        """Get item by index.

        Args:
            idx (int): Index of the item to get.

        Returns:
            Dict[str, Union[Tensor, int]]: Dictionary containing the item data.
                Contains:
                    - "image": Image tensor
                    - "pose": 6DOF pose tensor
                    - "idx": Index of the item
                    - "depth_image_front_cam": Depth image tensor
        """

        data: Dict[str, Union[int, Tensor]] = {"idx": torch.tensor(idx)}

        data["pose"] = self._pose_to_6dof(idx)

        data["image_front_cam"] = self._load_image(idx, transform=True)

        if self.load_depth_images:
            data["depth_image_front_cam"] = self._load_depth_image(idx)

        return data


    def __len__(self) -> int:
        """Return the number of scenes in the dataset."""
        return len(self.frame_ids)
    

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


    def _get_frame_image_path(self, idx: int) -> Path:
        """ Gets the path to the frame image.

            Args:
                idx (int): Frame index.
            
            Returns:
                Path: Path to the frame image.
        """

        scene_id, frame_id = self.frame_ids[idx]
        scene_path = self.scenes_folder / scene_id
        return scene_path / f"sequence/frame-{frame_id:06d}.color.jpg"


    def _get_frame_pos_path(self, idx: int) -> Path:
        """ Gets the path to the frame pos.

            Args:
                idx (int): Frame index.
            
            Returns:
                Path: Path to the frame pos.
        """

        scene_id, frame_id = self.frame_ids[idx]
        scene_path = self.scenes_folder / scene_id
        return scene_path / f"sequence/frame-{frame_id:06d}.pose.txt"
    

    def _get_frame_depth_image_path(self, idx: int) -> Path:
        """ Gets the path to the frame depth image.

            Args:
                idx (int): Frame index.
            
            Returns:
                Path: Path to the frame depth image.
        """

        scene_id, frame_id = self.frame_ids[idx]
        scene_path = self.scenes_folder / scene_id
        return scene_path / f"sequence/frame-{frame_id:06d}.depth.pgm"


    def _load_image(self, idx: int, transform: bool = True) -> Tensor:
        """Load an image from the dataset.
        Args:
            idx (int): Frame index.
            transform (bool): Whether to apply the image transform. Defaults to True.

        Returns:
            Tensor: Loaded image tensor.
        """

        im = cv2.imread(str(self._get_frame_image_path(idx)), cv2.IMREAD_UNCHANGED)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if transform:
            im = self.image_transform(im)
        return im


    def _load_depth_image(self, idx: int) -> Tensor:
        """Load a depth image from the dataset.

        Args:
            idx (int): Frame index.

        Returns:
            Tensor: Loaded depth image tensor.
        """

        im = cv2.imread(str(self._get_frame_depth_image_path(idx)), cv2.IMREAD_UNCHANGED)
        im = cv2.resize(im, self.depth_images_resize, interpolation=cv2.INTER_NEAREST)
        return im
    

    def _generate_dataframe(self, scene_references: Dict[str, str], track_numbers: Dict[str, int],
                            room_numbers: Dict[str, int], transformation_matrices: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Generate a DataFrame with poses for dataset_df.

        Args:
            scene_references (Dict[str, str]): Dictionary mapping scene IDs to their references.
            track_numbers (Dict[str, int]): Dictionary mapping scene IDs to their track numbers.
            room_numbers (Dict[str, int]): Dictionary mapping scene IDs to their room numbers.

        Returns:
            pd.DataFrame: DataFrame containing  poses data in tx, ty, tz, qx, qy, qz, qw and track number.
        """

        data: List[Dict[str, float]] = []
        for idx in range(len(self.frame_ids)):
            scene_id, frame_id = self.frame_ids[idx]
            with open(self._get_frame_pos_path(idx), "r") as f:
                pose = np.array([float(x) for x in f.read().split()]).reshape((4, 4))
                
                if scene_id in transformation_matrices:
                    pose = transformation_matrices[scene_id] @ pose

                translation = pose[:3, 3]  # [Tx, Ty, Tz]
                rotation_matrix = pose[:3, :3]
                quaternion = R.from_matrix(rotation_matrix).as_quat()  # [Qx, Qy, Qz, Qw]

                tx, ty, tz = translation
                qx, qy, qz, qw = quaternion

            track = track_numbers[scene_id]

            data.append({
                # Adding coordinates to tx so that all rooms are in different coordinates
                "tx": tx + 100 * room_numbers[scene_references[self.frame_ids[idx][0]]],
                "ty": ty,
                "tz": tz,
                "qx": qx,
                "qy": qy,
                "qz": qz,
                "qw": qw,
                "track": track
            })

        df = pd.DataFrame(data)
        if self.subset == "test":
            df["in_query"] = True

        return df
    

    def collate_fn(self, data_list: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
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
            elif key == "image_front_cam":
                batch["images_front_cam"] = torch.stack([data[key] for data in data_list], dim=0)
            elif key == "depth_image_front_cam":
                batch["depth_images_front_cam"] = torch.stack([data[key] for data in data_list], dim=0)

        return batch


    @staticmethod
    def unzip_sequences(dataset_dir: Union[Path, str]) -> None:
        """Unzip all the sequence RGBD pictures in each downloaded scene.
        Is is assumed, that all the scenes are in the data/3RScan subdirectory.

        Args:
            dataset_dir (Union[Path, str]): Root directory of dataset.

        Raises:
            FileNotFoundError: If dataset_dir/data/3RScan doesn't exist or is not a directory.
        """

        # Check if the path to scenes exists and is a directory
        dataset_dir = Path(dataset_dir) / "data/3RScan"
        if not dataset_dir.exists() or not dataset_dir.is_dir():
            raise FileNotFoundError(f"{dataset_dir} does not exist or is not a directory.")

        # Unzip all the sequence RGBD pictures in each downloaded scene
        for scene_path in dataset_dir.iterdir():
            if scene_path.is_dir():
                if (scene_path / "sequence").exists():
                    print(f"{scene_path / 'sequence'} already exists, skipping extraction.")
                    continue

                if (scene_path / "sequence.zip").exists():
                    with zipfile.ZipFile(scene_path / "sequence.zip", "r") as zip_ref:
                        zip_ref.extractall(scene_path / "sequence")
                    (scene_path / "sequence.zip").unlink()
                else:
                    print(f"{scene_path / 'sequence.zip'} does not exist, skipping extraction.")
        
        print("Unzipping completed.")
