import itertools
import pathlib
import random
import sys
import os
import json

import socket
hostname = socket.gethostname()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

#  os.environ["KERAS_BACKEND"] = "theano"

#  if len(sys.argv) == 2 and sys.argv[1] == "--gpu":
#  os.environ["THEANO_FLAGS"] = "device=cuda0"

import tensorflow as tf

#  tf.disable_eager_execution()
#  tf.disable_v2_behavior()

import file_utils
import model
import keras
import nibabel as nb
import numpy as np
from constants import EPOCHS, SIZE, BATCH_SIZE, OVERLAP
from keras.callbacks import ModelCheckpoint, EarlyStopping
from skimage.transform import resize
from scipy.ndimage import rotate


import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint

import horovod.keras as hvd

hvd.init()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))


def apply_transform(image, rot1, rot2):
    if rot1 > 0:
        image = rotate(
            image, angle=rot1, axes=(1, 0), output=np.float32, order=0, prefilter=False
        )
    if rot2 > 0:
        image = rotate(
            image, angle=rot2, axes=(2, 1), output=np.float32, order=0, prefilter=False
        )
    return image


def get_epoch_size(files, patch_size=SIZE):
    size = 0
    for image_filename, mask_filename in files:
        sz, sy, sx = nb.load(str(image_filename)).shape
        size += int(
            np.ceil(sz / OVERLAP)
            * np.ceil(sy / OVERLAP)
            * np.ceil(sx / OVERLAP)
        )
    return size


def gen_patches(image, mask, patch_size=SIZE):
    sz, sy, sx = image.shape
    i_cuts = itertools.product(
        range(0, sz, OVERLAP), range(0, sy, OVERLAP), range(0, sx, OVERLAP)
    )
    sub_image = np.empty(
        shape=(patch_size, patch_size, patch_size), dtype="float32"
    )
    sub_mask = np.empty_like(sub_image)

    for iz, iy, ix in i_cuts:
        sub_image[:] = 0
        sub_mask[:] = 0

        _sub_image = image[
            iz : iz + patch_size, iy : iy + patch_size, ix : ix + patch_size
        ]
        _sub_mask = mask[
            iz : iz + patch_size, iy : iy + patch_size, ix : ix + patch_size
        ]

        sz, sy, sx = _sub_image.shape

        sub_image[0:sz, 0:sy, 0:sx] = _sub_image
        sub_mask[0:sz, 0:sy, 0:sx] = _sub_mask

        yield sub_image, sub_mask


def get_patch(image, mask, patch, patch_size):
    sub_image = np.zeros(
        shape=(patch_size, patch_size, patch_size), dtype="float32"
    )
    sub_mask = np.zeros_like(sub_image)
    iz, iy, ix = patch

    _sub_image = image[
        iz : iz + patch_size, iy : iy + patch_size, ix : ix + patch_size
    ]
    _sub_mask = mask[
        iz : iz + patch_size, iy : iy + patch_size, ix : ix + patch_size
    ]

    sz, sy, sx = _sub_image.shape

    sub_image[0:sz, 0:sy, 0:sx] = _sub_image
    sub_mask[0:sz, 0:sy, 0:sx] = _sub_mask

    return sub_image, sub_mask


def gen_image_arrays_patches(dataset, patch_size):
    last_filename = ""
    for image_filename, mask_filename, patch in dataset:
        if image_filename != last_filename:
            image = nb.load(image_filename).get_fdata()
            image = model.image_normalize(image)
            mask = nb.load(mask_filename).get_fdata()
            mask = model.image_normalize(mask)
            print(image_filename, image.min(), image.max(), mask.min(), mask.max())

        last_filename = image_filename
        sub_image, sub_mask = get_patch(image, mask, patch, patch_size)
        yield (sub_image, sub_mask)



def load_models_patches(files, transformations, patch_size=SIZE, batch_size=BATCH_SIZE):
    for image_filename, mask_filename in files:
        image = nb.load(str(image_filename)).get_fdata()
        mask = nb.load(str(mask_filename)).get_fdata()
        image = model.image_normalize(image)
        mask = model.image_normalize(mask)
        rot1, rot2 = random.choice(transformations)
        t_image = apply_transform(image, rot1, rot2)
        t_mask = apply_transform(mask, rot1, rot2)

        for sub_image, sub_mask in gen_patches(t_image, t_mask, patch_size):
            yield (sub_image, sub_mask)


def gen_train_arrays(files, patch_size=SIZE, batch_size=BATCH_SIZE):
    transformations = list(itertools.product(range(0, 360, 30), range(0, 360, 30)))
    #  size = get_epoch_size(files, patch_size) #* len(transformations)
    size = len(files)
    yield int(np.ceil(size / batch_size))
    images = np.zeros((batch_size, SIZE, SIZE, SIZE, 1), dtype="float32")
    masks = np.zeros((batch_size, SIZE, SIZE, SIZE, 1), dtype="float32")
    while True:
        ip = 0
        for image, mask in gen_image_arrays_patches(files, patch_size):
            images[ip] = image.reshape(SIZE, SIZE, SIZE, 1)
            masks[ip] = mask.reshape(SIZE, SIZE, SIZE, 1)
            ip += 1

            if ip == batch_size:
                yield (images, masks)
                ip = 0


def load_models(files, batch_size=1):
    transformations = list(itertools.product(range(0, 360, 90), range(0, 360, 90)))

    size = len(transformations) * len(files)
    yield int(np.ceil(size / batch_size))

    images = np.zeros((batch_size, SIZE, SIZE, SIZE, 1), dtype="float32")
    masks = np.zeros((batch_size, SIZE, SIZE, SIZE, 1), dtype="float32")
    ip = 0
    while True:
        for image_filename, mask_filename in files:
            image = nb.load(str(image_filename)).get_fdata()
            mask = nb.load(str(mask_filename)).get_fdata()
            image = resize(
                image, (SIZE, SIZE, SIZE), mode="constant", anti_aliasing=True
            )
            mask = resize(mask, (SIZE, SIZE, SIZE), mode="constant", anti_aliasing=True)
            image = model.image_normalize(image)
            mask = model.image_normalize(mask)
            for rot1, rot2 in transformations:
                t_image = apply_transform(image, rot1, rot2)
                t_mask = apply_transform(mask, rot1, rot2)

                print(hostname, image_filename, rot1, rot2)

                images[ip] = t_image.reshape(SIZE, SIZE, SIZE, 1)
                masks[ip] = t_mask.reshape(SIZE, SIZE, SIZE, 1)
                ip += 1

                if ip == batch_size:
                    yield (images, masks)
                    ip = 0


def train(kmodel, deepbrain_folder):
    with open("train_files.json", "r") as f:
        training_files = json.load(f)
        node_training_size = int(np.ceil(len(training_files) / hvd.size()))
        training_files = training_files[hvd.rank() * node_training_size: (hvd.rank() + 1) * node_training_size]

    with open("testing_files.json", "r") as f:
        testing_files = json.load(f)
        node_testing_size = int(np.ceil(len(testing_files) / hvd.size()))
        testing_files = testing_files[hvd.rank() * node_testing_size: (hvd.rank() + 1) * node_testing_size]

    training_files_gen = gen_train_arrays(training_files, SIZE, BATCH_SIZE)
    testing_files_gen = gen_train_arrays(testing_files, SIZE, BATCH_SIZE)
    len_training_files = next(training_files_gen)
    len_testing_files = next(testing_files_gen)

    print(len_testing_files, len_training_files)

    best_model_file = pathlib.Path("weights/weights.h5").resolve()
    best_model = ModelCheckpoint(
        str(best_model_file), monitor="val_loss", verbose=1, save_best_only=True
    )

    opt = keras.optimizers.Adadelta(learning_rate=1.0 * hvd.size())
    # Horovod: add Horovod Distributed Optimizer.
    opt = hvd.DistributedOptimizer(opt)
    kmodel.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

    callbacks = [
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        # Horovod: average metrics among workers at the end of every epoch.
        #
        # Note: This callback must be in the list before the ReduceLROnPlateau,
        # TensorBoard or other metrics-based callbacks.
        hvd.callbacks.MetricAverageCallback(),
        # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
        # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
        # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
        hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1),
        # Reduce the learning rate if training plateaues.
        keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=True),
    ]

    if hvd.rank() == 0:
        callbacks.append(best_model)
        callbacks.append(model.PlotLosses())
        callbacks.append(
            keras.callbacks.tensorboard_v1.TensorBoard(log_dir='./logs',
                                                       histogram_freq=0,
                                                       batch_size=BATCH_SIZE,
                                                       write_graph=True,
                                                       write_grads=False,
                                                       write_images=False,
                                                       embeddings_freq=0,
                                                       embeddings_layer_names=None,
                                                       embeddings_metadata=None,
                                                       embeddings_data=None,
                                                       update_freq='batch')
        )

    kmodel.fit_generator(
        training_files_gen,
        steps_per_epoch=len_training_files,
        epochs=EPOCHS,
        validation_data=testing_files_gen,
        validation_steps= len_testing_files,
        callbacks=callbacks,
    )


def main():
    kmodel = model.generate_model()
    train(kmodel, pathlib.Path("datasets").resolve())
    model.save_model(kmodel)


if __name__ == "__main__":
    main()
