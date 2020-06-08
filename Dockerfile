FROM ubuntu:20.10

WORKDIR /code
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        nvidia-cuda-toolkit \
        nvidia-cuda-dev \
        git \
        curl \
        wget \
        build-essential \
        python3-all-dev \
        python3-setuptools \
        python3-distutils \
        python3-wheel \
        python3-pip \
        libgpuarray-dev \
        libopenblas-dev \
        graphviz \
        libnvrtc10.1 \
        libcublas10 \
        && \
        rm -rf /var/lib/apt/lists/* \
        && \
        pip3 install Theano \
        plaidml-keras \
        plaidbench \
        numpy \
        scipy \
        scikit-image \
        nibabel \
        matplotlib \
        pycuda \
        pyopencl

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.1 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411 brand=tesla,driver>=418,driver<419"
