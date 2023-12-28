#!/bin/bash

orange=`tput setaf 3`
reset_color=`tput sgr0`

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        realpath -m "$PWD"/"$1"
    fi
}

ARCH=`uname -m`
if [ $ARCH == "x86_64" ]; then
    if command -v nvtop-smi &> /dev/null; then
        DEVICE=cuda
        ARGS="--ipc host --gpus all"
    else
        echo "${orange}CPU-only${reset_color} build is not supported yet"
        exit 1
    fi
else
    echo "${orange}${ARCH}${reset_color} architecture is not supported"
    exit 1
fi

if [ $# != 1 ]; then
    echo "Usage:
          bash start.sh [DATASETS_DIR]
        "
    exit 1
fi

DATASETS_DIR=$(get_real_path "$1")

if [ ! -d $DATASETS_DIR ]; then
    echo "Error: DATASETS_DIR=$DATASETS_DIR is not an existing directory."
    exit 1
fi

PROJECT_ROOT_DIR=$(cd ./"`dirname $0`"/.. || exit; pwd)

echo "Running on ${orange}${ARCH}${reset_color} with ${orange}${DEVICE}${reset_color}"

LOCAL_DOCKER_IMAGE="open-place-recognition:"$DEVICE-$USER
DOCKER_HUB_IMAGE_NAME="creatorofuniverses/opr:dev"

# Check if the local image exists
image_exists=$(docker image ls | grep "$LOCAL_DOCKER_IMAGE" | wc -l)

# If the image does not exist locally, pull from Docker Hub
if [ $image_exists -eq 0 ]; then
    echo "Image $LOCAL_DOCKER_IMAGE not found locally. Pulling $DOCKER_HUB_IMAGE_NAME from Docker Hub..."
    docker pull $DOCKER_HUB_IMAGE_NAME

    # Optionally tag it as the local image name if you need to maintain that naming convention
    docker tag $DOCKER_HUB_IMAGE_NAME $LOCAL_DOCKER_IMAGE
    echo "Image $DOCKER_HUB_IMAGE_NAME has been re-tagged as $LOCAL_DOCKER_IMAGE"
else
    echo "Image $LOCAL_DOCKER_IMAGE found locally. No need to pull."
fi

echo "Using ${green}${LOCAL_DOCKER_IMAGE}${reset_color}"

docker run -it -d --rm \
    $ARGS \
    --privileged \
    --name ${USER}_opr \
    --net host \
    -v $PROJECT_ROOT_DIR:/home/docker_opr/OpenPlaceRecognition:rw \
    -v $DATASETS_DIR:/home/docker_opr/Datasets:rw \
    $LOCAL_DOCKER_IMAGE

docker exec --user root \
    ${USER}_opr bash -c "/etc/init.d/ssh start"