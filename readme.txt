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
