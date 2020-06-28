#!/bin/bash

# Creating docker image
docker build -t manolo_deepbrain .

# Runing the docker container
docker run --gpus all -v `pwd`:/code --rm manolo_deepbrain /bin/bash run_horovod_tensorboard.sh
