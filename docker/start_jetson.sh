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
if [ $ARCH != "aarch64" ]; then
    echo "${orange}${ARCH}${reset_color} architecture is not supported"
    exit 1
fi

ARGS="--runtime nvidia"

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

echo "Running on ${orange}${ARCH}${reset_color}"

docker run -it -d --rm \
    $ARGS \
    --privileged \
    --name ${USER}_opr \
    --net host \
    --shm-size=2g \
    -v $PROJECT_ROOT_DIR:/home/docker_opr/OpenPlaceRecognition:rw \
    -v $DATASETS_DIR:/home/docker_opr/Datasets:rw \
    open-place-recognition-jetson:devel-r35.4.1-cu114-cp310

docker exec --user root \
    ${USER}_opr bash -c "/etc/init.d/ssh start"
