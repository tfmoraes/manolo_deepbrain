#!/bin/bash

# Creating docker image
docker build -t manolo_deepbrain .

# Runing the docker container
docker run --gpus all -v `pwd`:/code --rm manolo_deepbrain python3 train_same_patches.py --gpu -b plaidml
