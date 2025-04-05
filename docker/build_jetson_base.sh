#!/bin/bash

orange=`tput setaf 3`
reset_color=`tput sgr0`

ARCH=`uname -m`
if [ $ARCH != "aarch64" ]; then
    echo "${orange}${ARCH}${reset_color} architecture is not supported"
    exit 1
fi

echo "Building for ${orange}${ARCH}${reset_color}"

PROJECT_ROOT_DIR=$(cd ./"`dirname $0`"/.. || exit; pwd)
DOCKERFILE=Dockerfile.jetson-base

docker build $PROJECT_ROOT_DIR \
    --build-arg MAX_JOBS=4 \
    --build-arg UID=$(id -u) \
    --build-arg GID=$(id -g) \
    -f $PROJECT_ROOT_DIR/docker/$DOCKERFILE \
    -t open-place-recognition-jetson:base-r35.4.1-cu114-cp310 \
    --network=host
