# Open Place Recognition library

## Installation

### Pre-requisites

- The library requires `PyTorch`, `MinkowskiEngine` and (optionally) `faiss` libraries to be installed manually:
  - [PyTorch Get Started](https://pytorch.org/get-started/locally/)
  - [MinkowskiEngine repository](https://github.com/NVIDIA/MinkowskiEngine)
  - [faiss repository](https://github.com/facebookresearch/faiss)

- Another option is to use the suggested Dockerfile. The following commands should be used to build, start and enter the container:

  1. Build the image

      ```bash
      bash docker/build.sh
      ```

  2. Start the container with the datasets directory mounted:

      ```bash
      bash docker/start.sh [DATASETS_DIR]
      ```

  3. Enter the container (if needed):

      ```bash
      bash docker/into.sh
      ```

### Library installation

- After the pre-requisites are met, install the Open Place Recognition library with the following command:

    ```bash
    pip install .
    ```

## Package Structure

### opr.datasets

Subpackage containing dataset classes and functions.

Usage example:

```python
from opr.datasets import OxfordDataset

train_dataset = OxfordDataset(
    dataset_root="/home/docker_opr/Datasets/pnvlad_oxford_robotcar_full/",
    subset="train",
    data_to_load=["image_stereo_centre", "pointcloud_lidar"]
)
```

The iterator will return a dictionary with the following keys:
- `"idx"`: index of the sample in the dataset, single number Tensor
- `"utm"`: UTM coordinates of the sample, Tensor of shape `(2)`
- (optional) `"image_stereo_centre"`: image Tensor of shape `(C, H, W)`
- (optional) `"pointcloud_lidar_feats"`: point cloud features Tensor of shape `(N, 1)`
- (optional) `"pointcloud_lidar_coords"`: point cloud coordinates Tensor of shape `(N, 3)`

More details can be found in the [demo_datasets.ipynb](./notebooks/demo_datasets.ipynb) notebook.

### opr.models

The `opr.models` subpackage contains ready-to-use neural networks implemented in PyTorch, featuring a common interface.

Usage example:

```python
from opr.models.place_recognition import MinkLoc3D

model = MinkLoc3D()

# forward pass
output = model(batch)
```

The models introduce unified input and output formats:
- **Input:** a `batch` dictionary with the following keys
  (all keys are optional, depending on the model and dataset):
  - `"images_<camera_name>"`: images Tensor of shape `(B, 3, H, W)`
  - `"masks_<camera_name>"`: semantic segmentation masks Tensor of shape `(B, 1, H, W)`
  - `"pointclouds_lidar_coords"`: point cloud coordinates Tensor of shape `(B * N_points, 4)`
  - `"pointclouds_lidar_feats"`: point cloud features Tensor of shape `(B * N_points, C)`
- **Output:** a dictionary with the requiered key `"final_descriptor"`
  and optional keys for intermediate descriptors:
  - `"final_descriptor"`: final descriptor Tensor of shape `(B, D)`

More details can be found in the [demo_models.ipynb](./notebooks/demo_models.ipynb) notebook.

### opr.pipelines

The `opr.pipelines` subpackage contains ready-to-use pipelines for model inference.

Usage example:

```python
from opr.models.place_recognition import MinkLoc3Dv2
from opr.pipelines.place_recognition import PlaceRecognitionPipeline

pipe = PlaceRecognitionPipeline(
    database_dir="/home/docker_opr/Datasets/ITLP_Campus/ITLP_Campus_outdoor/databases/00",
    model=MinkLoc3Dv2(),
    model_weights_path=None,
    device="cuda",
)

out = pipe.infer(sample)
```

The pipeline introduces a unified interface for model inference:
- **Input:** a dictionary with the following keys
  (all keys are optional, depending on the model and dataset):
  - `"image_<camera_name>"`: image Tensor of shape `(3, H, W)`
  - `"mask_<camera_name>"`: semantic segmentation mask Tensor of shape `(1, H, W)`
  - `"pointcloud_lidar_coords"`: point cloud coordinates Tensor of shape `(N_points, 4)`
  - `"pointcloud_lidar_feats"`: point cloud features Tensor of shape `(N_points, C)`
- **Output:** a dictionary with keys:
  - `"pose"` for predicted pose in the format `[tx, ty, tz, qx, qy, qz, qw]`,
  - `"descriptor"` for predicted descriptor.

## License

[MIT License](./LICENSE) (**_the license is subject to change in future versions_**)
