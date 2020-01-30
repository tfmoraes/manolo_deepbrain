import itertools
import pathlib
import random
import sys
import os

#  os.environ["KERAS_BACKEND"] = "theano"

#  if len(sys.argv) == 2 and sys.argv[1] == "--gpu":
#  os.environ["THEANO_FLAGS"] = "device=cuda0"

import tensorflow as tf

#  tf.disable_eager_execution()
#  tf.disable_v2_behavior()

import file_utils
import model
import nibabel as nb
import numpy as np
from constants import EPOCHS, SIZE
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
        image = rotate(image, angle=rot1, axes=(1, 0))
    if rot2 > 0:
        image = rotate(image, angle=rot2, axes=(2, 1))
    return image


def load_models(files, batch_size=1):
    transformations = list(itertools.product(range(0, 360, 90), range(0, 360, 90)))

    size = len(transformations) * len(files)
    yield int(np.ceil(size / batch_size))

    images = np.zeros((batch_size, SIZE, SIZE, SIZE, 1), dtype="float32")
    masks = np.zeros((batch_size, SIZE, SIZE, SIZE, 1), dtype="float32")
    while True:
        ip = 0
        for image_filename, mask_filename in files:
            image = nb.load(str(image_filename)).get_fdata()
            mask = nb.load(str(mask_filename)).get_fdata()
            image = resize(image, (SIZE, SIZE, SIZE), mode="constant", anti_aliasing=True)
            mask = resize(mask, (SIZE, SIZE, SIZE), mode="constant", anti_aliasing=True)
            image = model.image_normalize(image)
            mask = model.image_normalize(mask)
            for rot1, rot2 in transformations:
                t_image = apply_transform(image, rot1, rot2)
                t_mask = apply_transform(mask, rot1, rot2)

                print(image_filename, rot1, rot2)

                images[ip] = t_image.reshape(SIZE, SIZE, SIZE, 1)
                masks[ip] = t_mask.reshape(SIZE, SIZE, SIZE, 1)
                ip += 1

                if ip == batch_size:
                    yield (images, masks)
                    images[:] = 0
                    masks[:] = 0
                    ip = 0
        if ip:
            yield (images, masks)
            images[:] = 0
            masks[:] = 0


def train(kmodel, deepbrain_folder):
    cc359_files = file_utils.get_cc359_filenames(deepbrain_folder)
    nfbs_files = file_utils.get_nfbs_filenames(deepbrain_folder)
    files = cc359_files + nfbs_files
    random.shuffle(files)

    training_files = files[: int(len(files) * 0.75)]
    testing_files = files[int(len(files) * 0.75) :]
    training_files_gen = load_models(training_files, 8)
    testing_files_gen = load_models(testing_files, 8)
    len_training_files = next(training_files_gen)
    len_testing_files = next(testing_files_gen)

    best_model_file = pathlib.Path("weights/weights.h5").resolve()
    best_model = ModelCheckpoint(
        str(best_model_file), monitor="val_loss", verbose=1, save_best_only=True
    )

    opt = keras.optimizers.Adadelta(learning_rate=1.0 * hvd.size())
    kmodel.compile("adam", loss="binary_crossentropy", metrics=["accuracy"])

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
    ]

    if hvd.rank() == 0:
        callbacks.append(best_model)
        callbacks.append(model.PlotLosses())

    kmodel.fit_generator(
        training_files_gen,
        steps_per_epoch=len_training_files // hvd.size(),
        epochs=EPOCHS,
        validation_data=testing_files_gen,
        validation_steps= 3 * len_testing_files // hvd.size(),
        callbacks=callbacks,
    )


def main():
    kmodel = model.generate_model()
    train(kmodel, pathlib.Path("datasets").resolve())
    model.save_model(kmodel)


if __name__ == "__main__":
    main()
