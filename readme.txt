Install the packages:

$ sudo apt install python3-keras python3-keras-applications python3-nibabel python3-matplotlib python3-skimage python3-numpy python3-scipy

Run:

$ ./prepare.sh
$ mkdir -p weights
$ export KERAS_BACKEND=theano
$ python3 train.py
