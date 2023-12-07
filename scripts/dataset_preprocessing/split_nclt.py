"""Script for splitting NCLT dataset."""
# flake8: noqa
# TODO: fix all linter problems and add type annotations
import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib
import pandas as pd
from pandas import DataFrame
from utils import check_in_buffer_set, check_in_test_set

matplotlib.use("Agg")


P_WIDTH = (100, 100)
BUFFER_WIDTH = 10
P1 = (-260.0, -680.0)  # (x, y)
P2 = (-280.0, -420.0)
P3 = (20.0, -550.0)
P4 = (-100.0, -300.0)
P = [P1, P2, P3, P4]


def parse_args() -> Tuple[Path, List[Path]]:
    """Parse input CLI arguments.

    Raises:
        ValueError: If the given '--dataset_root' directory does not exist.

    Returns:
        Path: Dataset root path.
        List[Path]: List of all 'track.csv' files in the subdirectories of the given dataset_root.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        type=Path,
        help="The path to the NCLT preprocessed dataset root directory.",
    )
    args = parser.parse_args()

    dataset_root: Path = args.dataset_root
    if not dataset_root.exists():
        raise ValueError("Given dataset_root directory does not exist.")

    track_files = sorted(list(dataset_root.glob("*/track.csv")))

    return dataset_root, track_files


def split_dataframe(
    track_df: DataFrame, track_name: str, split_distance: int = 10
) -> Tuple[DataFrame, DataFrame]:
    """Split the given dataframe into train and test parts.

    Args:
        track_df (DataFrame): Track DataFrame.
        track_name (str): Track name (track's directory name).
        split_distance (int): The distance between frames in the split. Should be divisible by 5.
            Defaults to 10.

    Raises:
        ValueError: If the given split_distance is not divisible by 5.

    Returns:
        Tuple[DataFrame, DataFrame]: Train and test DataFrames.
    """
    if split_distance % 5 != 0:
        raise ValueError(f"Given split_distance={split_distance} is not divisible by 5.")
    train_rows = []
    test_rows = []
    step = split_distance // 5
    for i in range(0, len(track_df), step):
        row = track_df.iloc[i]
        if check_in_test_set(row["northing"], row["easting"], test_boundary_points=P, boundary_width=P_WIDTH):
            test_rows.append(row)
        elif not check_in_buffer_set(
            row["northing"], row["easting"], test_boundary_points=P, boundary_width=P_WIDTH, buffer_width=BUFFER_WIDTH
        ):
            train_rows.append(row)
    train_df = pd.DataFrame(train_rows)
    test_df = pd.DataFrame(test_rows)
    train_df["track"] = track_name
    test_df["track"] = track_name
    return train_df, test_df


if __name__ == "__main__":
    dataset_root, track_files = parse_args()
    print(f"Found {len(track_files)} 'track.csv' files in the subdirectories of the given dataset_root")
    column_names = ["track", "image", "pointcloud", "northing", "easting"]
    train_df = DataFrame(columns=column_names)
    test_df = DataFrame(columns=column_names)
    for track_file in track_files:
        track_name = track_file.parent.name
        track_df = pd.read_csv(track_file, index_col=0)
        track_train_df, track_test_df = split_dataframe(track_df, track_name, split_distance=10)
        train_df = pd.concat([train_df, track_train_df], ignore_index=True)
        test_df = pd.concat([test_df, track_test_df], ignore_index=True)
    train_df = train_df[column_names]
    test_df = test_df[column_names]
    # train_df = train_df.rename(columns={"x": "northing", "y": "easting"})
    # test_df = test_df.rename(columns={"x": "northing", "y": "easting"})
    train_df[["image", "pointcloud"]] = train_df[["image", "pointcloud"]].astype("int64")
    test_df[["image", "pointcloud"]] = test_df[["image", "pointcloud"]].astype("int64")
    train_df.to_csv(dataset_root / "train.csv")
    test_df.to_csv(dataset_root / "val.csv")  # for compatibility with oxford robotcar
    test_df.to_csv(dataset_root / "test.csv")
    print(f"Saved 'train.csv', 'val.csv' and 'test.csv' in the dataset directory: {dataset_root}")
