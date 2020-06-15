FROM nvidia/cuda:10.2-devel-ubuntu18.04

WORKDIR /code
ENV DEBIAN_FRONTEND=noninteractive

ENV TENSORFLOW_VERSION=1.14.0
ENV PYTORCH_VERSION=1.5.0
ENV TORCHVISION_VERSION=0.6.0
ENV MXNET_VERSION=1.6.0
ENV BAZEL_VERSION=3.2.0

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
        pkg-config \
        unzip \
        zlib1g-dev \
        libnvidia-compute-440 \
        clinfo \
        cmake \
        && \
        rm -rf /var/lib/apt/lists/* \
        && \
        pip3 install --no-cache-dir \
        Theano \
        plaidml-keras \
        plaidbench \
        numpy \
        scipy \
        scikit-image \
        nibabel \
        matplotlib \
        tensorflow-gpu==${TENSORFLOW_VERSION} \
        torch==${PYTORCH_VERSION} \
        torchvision==${TORCHVISION_VERSION} \
        mxnet-cu102==${MXNET_VERSION} \
        pandas \
        scikit-learn


# Comente essas linhas se for instalar o tensorflow do pypi
# RUN # apt-get update && apt-get install -y --no-install-recommends openjdk-8-jdk && \
# RUN curl -Lo /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/download/v1.2.1/bazelisk-linux-amd64 && \
    # chmod +x /usr/local/bin/bazel && \
    # wget https://github.com/tensorflow/tensorflow/archive/v${TENSORFLOW_VERSION}.tar.gz && \
    # ln -sf /usr/bin/python3 /usr/bin/python && \
    # tar xf v${TENSORFLOW_VERSION}.tar.gz && \
    # cd tensorflow-${TENSORFLOW_VERSION} && \
    # bazel version && \
    # ./configure && \
    # bazel build --config=opt --config=cuda --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow/tools/pip_package:build_pip_package && \
    # ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg && \
    # pip3 install --upgrade --force-reinstall /tmp/tensorflow_pkg/tensorflow*.whl && \
    # cd .. && \
    # rm -rf tensorflow-${TENSORFLOW_VERSION} && \
    # rm -rf /tmp/* && \
    # # apt-get remove openjdk-8-jdk && \
    # rm -rf /var/lib/apt/lists/* && \
    # rm /usr/local/bin/bazel


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
    echo "deb http://archive.ubuntu.com/ubuntu/ xenial main" >> /etc/apt/sources.list && \
    echo "deb http://dk.archive.ubuntu.com/ubuntu/ xenial universe" >> /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends gcc-4.9 g++-4.9 && \
    rm -rf /var/lib/apt/lists/* && \
    CC=/usr/bin/gcc-4.9 CXX=/usr/bin/g++-4.9 pip3 install --no-cache-dir horovod && \
    ldconfig
