# Featured modules

This document provides a brief description of featured library modules.

## 1. PlaceRecognitionPipeline

A module that implements a neural network algorithm for searching a database of places already visited by a vehicle for the most similar records using sequences of data from lidars and cameras.

### Sample usage

```python
from hydra.utils import instantiate
from omegaconf import OmegaConf
from opr.datasets.itlp import ITLPCampus
from opr.pipelines.place_recognition import PlaceRecognitionPipeline

MODEL_CONFIG_PATH = "/path/to/OpenPlaceRecognition/configs/model/place_recognition/multi-image_lidar_late-fusion.yaml"
DATABASE_TRACK_DIR = "/path/to/ITLP-Campus-data/indoor/00_2023-10-25-night"
QUERY_TRACK_DIR = "/path/to/ITLP-Campus-data/indoor/01_2023-11-09-twilight"
SENSOR_SUITE = ["front_cam", "back_cam", "lidar"]
WEIGHTS_PATH = "/path/to/OpenPlaceRecognition/weights/place_recognition/multi-image_lidar_late-fusion_nclt.pth"
DEVICE = "cuda"

model_config = OmegaConf.load(MODEL_CONFIG_PATH)
model = instantiate(model_config)

pipe = PlaceRecognitionPipeline(
    database_dir=DATABASE_TRACK_DIR,
    model=model,
    model_weights_path=WEIGHTS_PATH,
    device=DEVICE,
)

query_dataset = ITLPCampus(
    dataset_root=QUERY_TRACK_DIR,
    sensors=SENSOR_SUITE,
    mink_quantization_size=0.5,
    subset="test",
    test_split=[1,2,3,4,5],
)

sample_query = query_dataset[0]
output = pipe.infer(sample_query)
```

## 2. SequencePointcloudRegistrationPipeline

A module that implements an algorithm for optimizing the position and orientation of a vehicle in space based on a sequence of multimodal data using neural network methods.

### Sample usage

```python
from hydra.utils import instantiate
from omegaconf import OmegaConf
from opr.datasets.itlp import ITLPCampus
from opr.pipelines.registration.pointcloud import SequencePointcloudRegistrationPipeline

REGISTRATION_MODEL_CONFIG_PATH = "/path/to/OpenPlaceRecognition/configs/model/registration/geotransformer_kitti.yaml"
REGISTRATION_WEIGHTS_PATH = "/path/to/OpenPlaceRecognition/weights/registration/geotransformer_kitti.pth"
DATABASE_TRACK_DIR = "/path/to/ITLP-Campus-data/indoor/00_2023-10-25-night"
QUERY_TRACK_DIR = "/path/to/ITLP-Campus-data/indoor/01_2023-11-09-twilight"
SENSOR_SUITE = ["front_cam", "back_cam", "lidar"]

geotransformer = instantiate(OmegaConf.load(REGISTRATION_MODEL_CONFIG_PATH))

registration_pipe = SequencePointcloudRegistrationPipeline(
    model=geotransformer,
    model_weights_path=REGISTRATION_WEIGHTS_PATH,
    device="cuda",  # the GeoTransformer currently only supports CUDA
    voxel_downsample_size=0.5,
)

query_dataset = ITLPCampus(
    dataset_root=QUERY_TRACK_DIR,
    sensors=SENSOR_SUITE,
    mink_quantization_size=0.5,
    max_point_distance=20,
    subset="test",
    test_split=[1,2,3,4,5],
)
db_dataset = ITLPCampus(
    dataset_root=DATABASE_TRACK_DIR,
    sensors=SENSOR_SUITE,
    mink_quantization_size=0.5,
    max_point_distance=20,
    subset="test",
    test_split=[1,2,3,4,5],
)

i = 10  # example index of the query sequence
query_seq = [query_dataset[i-1]["pointcloud_lidar_coords"], query_dataset[i]["pointcloud_lidar_coords"]]  # example of accumulation of two consecutive point clouds

place_recognition_output_idx = ...  # we assume that the place recognition module has already been run and the index of the most similar place in the database has been obtained
db_match = db_dataset[place_recognition_output_idx]
db_pc = db_match["pointcloud_lidar_coords"]

estimated_transform = registration_pipe.infer(query_seq, db_pc)
```

## 3. PlaceRecognitionPipeline with semantics

A module that implements an algorithm for optimizing the position and orientation of a vehicle in space based on a sequence of multimodal data using neural network methods.

### Sample usage

This method is a modification of the already described [PlaceRecognitionPipeline](#1-placerecognitionpipeline). The main difference is the use of a different configuration of the neural network model and additional loading of semantics in the dataset:

```python
MODEL_CONFIG_PATH = "../../configs/model/place_recognition/multi-image_multi-semantic_lidar_late-fusion.yaml"  # THIS
WEIGHTS_PATH = "../../weights/place_recognition/multi-image_multi-semantic_lidar_late-fusion_nclt.pth"  # THIS

query_dataset = ITLPCampus(
    dataset_root=QUERY_TRACK_DIR,
    sensors=SENSOR_SUITE,
    mink_quantization_size=0.5,
    subset="test",
    test_split=[1,2,3,4,5],
    load_semantics=True,  # AND THIS
)
```

## 4. ArucoLocalizationPipeline

A module that implements an algorithm for optimizing the position and orientation of a vehicle in space based on an user predefined Aruco Markers and multimodal data using neural network methods.

Sample usage: see [```notebooks/aruco_pipeline.ipynb```](../notebooks/aruco_pipeline.ipynb)

## 5. LocalizationPipeline without dynamic objects

A module that implements an algorithm for optimizing the position and orientation of a vehicle in space based on multimodal data without user predefined dynamic objects using neural network methods.

Sample usage: see [```notebooks/localization_with_dynamic_objects.ipynb```](../notebooks/localization_with_dynamic_objects.ipynb)

## 8. MultimodalPlaceRecognitionTrainer

A module that implements a training algorithm for a multimodal neural network model of global localization based on the contrastive learning approach.

**Sample usage:** see [`scripts/training/place_recognition/train_multimodal.py`](../scripts/training/place_recognition/train_multimodal.py) for an example of training a model.
