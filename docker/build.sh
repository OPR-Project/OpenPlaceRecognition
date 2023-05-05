#!/bin/bash

orange=`tput setaf 3`
reset_color=`tput sgr0`

ARCH=`uname -m`
if [ $ARCH != "x86_64" ]; then
    echo "${orange}${ARCH}${reset_color} architecture is not supported"
    exit 1
fi

if command -v nvidia-smi &> /dev/null; then
    echo "Detected ${orange}CUDA${reset_color} hardware"
    DOCKERFILE=Dockerfile.cuda
    DEVICE=cuda
else
    echo "${orange}CPU-only${reset_color} build is not supported yet"
    exit 1
fi

echo "Building for ${orange}${ARCH}${reset_color} with ${orange}${DEVICE}${reset_color}"

PROJECT_ROOT_DIR=$(cd ./"`dirname $0`"/.. || exit; pwd)

docker build $PROJECT_ROOT_DIR \
    -f $PROJECT_ROOT_DIR/docker/$DOCKERFILE \
    --build-arg UID=$(id -u) \
    --build-arg GID=$(id -g) \
    -t open-place-recognition:$DEVICE-$USER
