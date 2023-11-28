#!/bin/bash

docker exec --user docker_opr -it ${USER}_opr \
    /bin/bash -c "cd /home/docker_opr; echo ${USER}_opr container; echo ; /bin/bash"