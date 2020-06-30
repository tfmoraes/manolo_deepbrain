#!/bin/bash

# Creating docker image
docker build -t manolo_deepbrain .

# Runing the docker container
docker run --runtime=nvidia -v `pwd`:/code --rm --name paulo_nodes_lung manolo_deepbrain python3 -u gen_train_array.py
