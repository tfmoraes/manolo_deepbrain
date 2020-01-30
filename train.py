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
from constants import EPOCHS, SIZE
from skimage.transform import resize


import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint

import horovod.keras as hvd

hvd.init()

print(">>>>", str(hvd.local_rank()))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))



def load_models(files):
    swaps = itertools.cycle(
        ((None, None),) + tuple(itertools.combinations(range(3), 2))
    )
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

            i, j = next(swaps)

            with open("output_stdout.txt", "a") as f:
                f.write(str(image_filename) + "\n")

            if i is None:
                yield image.astype("float32").reshape(
                    1, SIZE, SIZE, SIZE, 1
                ), mask.astype("float32").reshape(1, SIZE, SIZE, SIZE, 1)
            else:
                yield image.astype("float32").swapaxes(i, j).reshape(
                    1, SIZE, SIZE, SIZE, 1
                ), mask.astype("float32").swapaxes(i, j).reshape(1, SIZE, SIZE, SIZE, 1)


def train(kmodel, deepbrain_folder):
    cc359_files = file_utils.get_cc359_filenames(deepbrain_folder)
    nfbs_files = file_utils.get_nfbs_filenames(deepbrain_folder)
    files = cc359_files + nfbs_files
    random.shuffle(files)

    training_files = files[: int(len(files) * 0.75)]
    testing_files = files[int(len(files) * 0.75) :]
    training_files_gen = load_models(training_files)
    testing_files_gen = load_models(testing_files)

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
        steps_per_epoch=len(training_files) // hvd.size(),
        epochs=EPOCHS,
        validation_data=testing_files_gen,
        validation_steps=len(testing_files),
        callbacks=callbacks,
    )


def main():
    kmodel = model.generate_model()
    train(kmodel, pathlib.Path("datasets").resolve())
    model.save_model(kmodel)


main()
