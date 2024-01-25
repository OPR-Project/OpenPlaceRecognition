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
