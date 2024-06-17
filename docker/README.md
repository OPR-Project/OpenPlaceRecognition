# Docker images

## Base image

The `alexmelekhin/open-place-recognition:base` image contains following:

- Ubuntu 22.04
- Python 3.10
- CUDA 12.1.1
- cuDNN 8
- PyTorch 2.1.2
- torhvision 0.16.2
- MinkowskiEngine
- faiss

You can either pull it from dockerhub:

```bash
docker pull alexmelekhin/open-place-recognition:base
```

Or build it manually:

```bash
# from repo root dir
bash docker/build_base.sh
```

## Devel image

The `open-place-recognition:devel` image are build upon base image described above and contain pip requirements pre-installed and non-root user created inside.

It should be build only manually (to correctly propagate your UID and GID):

```bash
# from repo root dir
bash docker/build_devel.sh
```

### Starting container

The container should be started using the `open-place-recognition:devel` image by using the following script:

```bash
# from repo root dir
bash docker/start.sh [DATASETS_DIR]
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

### Base image

The most convenient way to build a base image is the [jetson-containers](https://github.com/dusty-nv/jetson-containers) tool.
We tested the code with the following environment configuration:

- JetPack 5
- L4T 35.4.1
- Python 3.10
- CUDA 11.4
- PyTorch 2.1.0
- Torchvision 0.16.2
- Faiss

To build the base image run the command:

```bash
CUDA_VERSION=11.4 PYTHON_VERSION=3.10 PYTORCH_VERSION=2.1 jetson-containers build --name=open-place-recognition-jetson pytorch torchvision:0.16.2  faiss
```

If your configuration is the same as described above (jetpack 5 and l4t 35.4.1) the above command will build an image named `open-place-recognition-jetson:r35.4.1-cu114-cp310`.
This is a basic environment image.

### Devel image

To build a ready-to-use development image, use the provided [`Dockerfile.jetson-devel`](Dockerfile.jetson-devel) by running the [`build_jetson_devel.sh`](build_jetson_devel.sh) script:

```bash
# from repo root dir
bash docker/build_jetson_devel.sh
```

#### Starting container

The container should be started using the `open-place-recognition-jetson:devel` image by using the following script:

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
