#! /usr/bin/env bash

tensorboard --logdir=logs > /dev/null &
horovodrun -np 4 -H localhost:4 python3 train_same_patche_horovod.py
