#!/bin/bash

# Creating docker image
docker build -t manolo_deepbrain .

# Runing the docker container
#docker run --runtime=nvidia -it --rm -v `pwd`:/code -p 6006:6006 --gpus all --name paulo_nodulos_pulmao manolo_deepbrain_horovod python3 -u train_same_patches.py -b plaidml --gpu
docker run --runtime=nvidia -it --rm -v `pwd`:/code -p 6006:6006 --gpus all --name paulo_nodulos_pulmao manolo_deepbrain_horovod /bin/bash run_horovod_tensorboard.sh
