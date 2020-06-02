Run localy:

Install the packages:

$ sudo apt install python3-keras python3-keras-applications python3-nibabel python3-matplotlib python3-skimage python3-numpy python3-scipy

Or via pip and virtualenv:

$ mkvirtualenv -p /usr/bin/python3 deepbrain
$ pip3 install keras theano matplotlib scikit-image nibabel

Run:

$ ./prepare.sh
$ mkdir -p weights
$ export KERAS_BACKEND=theano
$ export THEANO_FLAGS=device=cuda0
$ python3 train.py

Run Via docker:

First add needed nvidia repos:

$ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
$ curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
$ curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu20.04/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
$ echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" | sudo tee /etc/apt/sources.list.d/cuda.list

Install needed packages:

$ apt update
$ sudo apt install nvidia-docker2 nvidia-container-toolkit cuda-toolkit-10-2 nvidia-driver-440

Use prepare.sh to download the datasets:

$ ./prepare.sh

Generate the array files to train the network:

$ ./run_gen_docker.sh

Then use run_train_docker.sh to train using docker:

$ ./run_train_docker.sh
