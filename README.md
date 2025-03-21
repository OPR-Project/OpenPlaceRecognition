# Open Place Recognition library

![Place Recognition overview](./docs/images/PR_overview.png)

_An overview of a typical place recognition pipeline. At first, the input data is encoded into a query descriptor. Then, a K-nearest neighbors search is performed between the query and the database. Finally, the position of the closest database descriptor found is considered as the answer._

This library is suitable for:

- 🚗 **Navigation of autonomous cars, robots, and drones** using cameras and lidars, especially in areas with limited or unavailable GPS signals.
- 📦 **Localization of delivery robots** needing reliable positioning both indoors and outdoors.
- 🔬 **Research and development of computer vision algorithms**, related to multimodal place recognition and localization.
- 🎓 **Educational purposes and research projects**, involving robotics, autonomous systems, and computer vision.

### Featured modules

Our library comes packed with ready-to-use modules that you can use "as-is" or as inspiration for your custom creations. Dive into the details in our [documentation](https://openplacerecognition.readthedocs.io/en/latest/featured_modules/index.html).

## Installation

### Quick-start

At first, ensure that you cloned the repository https://github.com/OPR-Project/OpenPlaceRecognition
and changed the directory to the root of the repository:

```bash
git clone https://github.com/OPR-Project/OpenPlaceRecognition.git
cd OpenPlaceRecognition
```

The recommended and easiest way to use the library is through the provided Docker environment.
The [`Dockerfile.base`](./docker/Dockerfile.base) contains all prerequisites needed to run the library,
including optional [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) and [faiss](https://github.com/facebookresearch/faiss) libraries.
You can either pull this image from Docker Hub (`docker pull alexmelekhin/open-place-recognition:base`),
or build it manually (`bash docker/build_base.sh`).

The [`Dockerfile.devel`](./docker/Dockerfile.devel) installs additional requirements from
[`requirements.txt`](./requirements.txt),
[`requirements-dev.txt`](./requirements-dev.txt),
and [`requirements-notebook.txt`](./requirements-notebook.txt) files,
and creates a non-root user inside the image to avoid permission issues when using the container with mounted volumes.

The `devel` version of the image should be built manually by the user:

```bash
bash docker/build_devel.sh
```

When starting the container, you must provide a data directory that will be mounted to `~/Datasets` inside the container:

```bash
bash docker/start.sh [DATASETS_DIR]
```

To enter the container's `/bin/bash` terminal, use the [`docker/into.sh`](./docker/into.sh) script:

```bash
bash docker/into.sh
```

After you enter the container, install the library (we recommend installing in editable mode with the `-e` flag to be able to make modifications to the code):

```bash
pip install -e ~/OpenPlaceRecognition
```

### Advanced

For more detailed instructions on installing third-party dependencies, configuring your environment manually, and additional setup options, please refer to the [Installation section of our documentation](https://openplacerecognition.readthedocs.io/en/latest/#installation).

### How to load the weights

You can download the weights from the public [Google Drive folder](https://drive.google.com/drive/folders/1uRiMe2-I9b5Tgv8mIJLkdGaHXccp_UFJ?usp=sharing).

<details>
  <summary>Developers only</summary>

  We use [DVC](https://dvc.org/) to manage the weights storage. To download the weights, run the following command (assuming that dvc is already installed):

  ```bash
  dvc pull
  ```

  You will be be asked to authorize the Google Drive access. After that, the weights will be downloaded to the `weights` directory. For more details, see the [DVC documentation](https://dvc.org/doc).
</details>

## ITLP-Campus dataset

Explore multimodal Place Recognition with ITLP Campus — a diverse dataset of indoor and outdoor university environments featuring synchronized RGB images,
LiDAR point clouds, semantic masks, and rich scene descriptions.
Built for real-world challenges, day or night, floor or field.

You can find more details in the [OPR-Project/ITLP-Campus](https://github.com/OPR-Project/ITLP-Campus) repository.

## Package Structure

You can learn more about the package structure in the
[Introduction - Package Structure](https://openplacerecognition.readthedocs.io/en/latest/index.html#package-structure)
section of the documentation.

## Model Zoo

### Place Recognition

| Model      | Modality | Train Dataset | Config | Weights |
| ---------- | -------- | ------------- | ------ | ------- |
| MinkLoc3D ([paper](https://openaccess.thecvf.com/content/WACV2021/html/Komorowski_MinkLoc3D_Point_Cloud_Based_Large-Scale_Place_Recognition_WACV_2021_paper.html)) | LiDAR | NCLT | [minkloc3d.yaml](./configs/model/place_recognition/minkloc3d.yaml) | `minkloc3d_nclt.pth` |
| Custom | Multi-Image, Multi-Semantic, LiDAR | NCLT | [multi-image_multi-semantic_lidar_late-fusion.yaml](./configs/model/place_recognition/multi-image_multi-semantic_lidar_late-fusion.yaml) | `multi-image_multi-semantic_lidar_late-fusion_nclt.pth` |
| Custom | Multi-Image, LiDAR | NCLT | [multi-image_lidar_late-fusion.yaml](./configs/model/place_recognition/multi-image_lidar_late-fusion.yaml) | `multi-image_lidar_late-fusion_nclt.pth` |

## Сonnected Projects

- [OPR-Project/OpenPlaceRecognition-ROS2](https://github.com/OPR-Project/OpenPlaceRecognition-ROS2) - ROS-2 implementation of OpenPlaceRecognition modules
- [OPR-Project/ITLP-Campus](https://github.com/OPR-Project/ITLP-Campus) - ITLP-Campus dataset tools.
- [KirillMouraviev/simple_toposlam_model](https://github.com/KirillMouraviev/simple_toposlam_model) - An implementation of the Topological SLAM method that uses the OPR library.


## License

[Apache 2.0 license](./LICENSE)
