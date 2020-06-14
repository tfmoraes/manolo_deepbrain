FROM nvidia/cuda:10.2-cudnn7-devel

WORKDIR /code
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        wget \
        build-essential \
        python3-all-dev \
        python3-setuptools \
        python3-distutils \
        python3-wheel \
        python3-pip \
        graphviz \
        openssh-client \
        openssh-server \
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
        pyopencl \
        torch \
        torchvision \
        tensorflow-gpu \
        mxnet-cu102 \
        pandas \
        scikit-learn

# Install Open MPI
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.0.tar.gz && \
    tar zxf openmpi-4.0.0.tar.gz && \
    cd openmpi-4.0.0 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

# Install Horovod, temporarily using CUDA stubs
RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_MXNET=1 \
    pip install --no-cache-dir horovod && \
    ldconfig
