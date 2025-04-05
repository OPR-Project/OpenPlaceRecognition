# Docker images

The Docker images built for this branch are adapted for use with ROS Noetic and [PRISM-TopoMap](https://github.com/KirillMouraviev/PRISM-TopoMap)

## Step-by-step instruction for building PRISM-TopoMap setup

1. Clone this repo with branch `feat/toposlam`:
`git clone --branch feat/toposlam https://github.com/OPR-Project/OpenPlaceRecognition`

2. Build the docker image from source (takes from 3 to 6 hours, depending from CPU and the Internet speed):
```bash
cd OpenPlaceRecognition/docker
bash build_base.sh
bash build_toposlam.sh
```
Or unpack the pre-built docker image:
```bash
docker load -i prism_topomap_docker
```

3. Run the docker container with the datasets directory:
```bash
bash start_toposlam.sh DATASETS_DIR
```
the `DATASETS_DIR` will be mounted into `~/Datasets` directory inside the container.

4. Enter into the container:
```bash
bash into.sh
```

5. Inside the container, setup the OpenPlaceRecognition library (mounted in the home directory):
```bash
cd OpenPlaceRecognition
pip install -e .
pip install rosnumpy memory_profiler
```

6. Inside the container, build the ROS Catkin workspace:
```bash
source /opt/ros/noetic/setup.bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

7. After that, you can launch PRISM-TopoMap:
For indoor demo (on AgileX Scout robot):
```bash
roslaunch prism_topomap build_map_by_iou_scout_rosbag.launch
```

For outdoor demo (on a self-driving truck, with `truck_localization` launch file created by you):
```bash
cd catkin_ws/src/PRISM-TopoMap
git checkout localization_mode
roslaunch prism_topomap truck_localization.launch
```

## Base image

The `alexmelekhin/open-place-recognition:base` image built by th contains following:

- Ubuntu 20.04
- Python 3.8
- CUDA 12.1.1
- cuDNN 8
- PyTorch 2.1.2
- torhvision 0.16.2
- MinkowskiEngine
- faiss

Or build it manually:

```bash
# from repo root dir
bash docker/build_base.sh
```

## TopoSLAM image (for PRISM-TopoMap)

The `prism_topomap_docker` image is build upon base image described above and contain pip requirements pre-installed as well as ROS Noetic and non-root user created inside. 

It should be build only manually (to correctly propagate your UID and GID):

```bash
# from repo root dir
bash docker/build_toposlam.sh
```

### Starting container for TopoSLAM

The container should be started using the `open-place-recognition:devel` image by using the following script:

```bash
# from repo root dir
bash docker/start_toposlam.sh [DATASETS_DIR]
```

The `[DATASETS_DIR]` will be mounted to the directory `/home/docker_opr/Datasets` with read and write access.

### Entering container

You can enter the container by using the following script:

```bash
# from repo root dir
bash docker/into.sh
```

## Jetson Xavier image

We provide an example of image to run on Jetson Xavier with arm64 architecture.

### jetson-containers

At first, you should build a starting image with [jetson-containers](https://github.com/dusty-nv/jetson-containers) tool.
We tested the code with the following environment configuration:

- JetPack 5
- L4T 35.4.1
- Python 3.10
- CUDA 11.4
- PyTorch 2.1.0
- Torchvision 0.16.2
- Faiss

To build the image run the command:

```bash
CUDA_VERSION=11.4 PYTHON_VERSION=3.10 TENSORRT_VERSION=8.5 PYTORCH_VERSION=2.1 TORCHVISION_VERSION=0.16.2 jetson-containers build --name=open-place-recognition-jetson tensorrt pytorch torchvision faiss
```

If your configuration is the same as described above (jetpack 5 and l4t 35.4.1) the above command will build an image named `open-place-recognition-jetson:r35.4.1-cu114-cp310`.

### Base image

Next, you should build a base image with main dependencies (MinkowskiEngine).
Use the provided [`Dockerfile.jetson-base`](Dockerfile.jetson-base) by running the [`build_jetson_base.sh`](build_jetson_base.sh) script:

```bash
# from repo root dir
bash docker/build_jetson_base.sh
```

### Devel image

To build a ready-to-use development image, use the provided [`Dockerfile.jetson-devel`](Dockerfile.jetson-devel) by running the [`build_jetson_devel.sh`](build_jetson_devel.sh) script:

```bash
# from repo root dir
bash docker/build_jetson_devel.sh
```

#### Starting container

The container should be started using the `open-place-recognition-jetson:devel-r35.4.1-cu114-cp310` image by using the following script:

```bash
# from repo root dir
bash docker/start_jetson.sh [DATASETS_DIR]
```

The `[DATASETS_DIR]` will be mounted to the directory `/home/docker_opr/Datasets` with read and write access.

#### Entering container

You can enter the container by using the following script:

```bash
# from repo root dir
bash docker/into.sh
```
