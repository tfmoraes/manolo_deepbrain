import argparse
import itertools
import os
import pathlib
import random
import sys
import json

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
#os.environ["PLAIDML_DEVICE_IDS"] = "llvm_cpu.0"

#  import ngraph_bridge
#  ngraph_bridge.set_backend('PLAIDML')


import h5py
import nibabel as nb
import numpy as np
import pylab as plt
import file_utils
from constants import BATCH_SIZE, EPOCHS, OVERLAP, SIZE, NUM_PATCHES
from utils import apply_transform, image_normalize, get_plaidml_devices
from skimage.transform import resize

import keras
import model
from keras import backend as K
from keras.callbacks import ModelCheckpoint

import tensorflow as tf
import horovod.keras as hvd

hvd.init()

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.compat.v1.Session(config=config))



class PlotLosses(keras.callbacks.Callback):
    def __init__(self, continue_train=False):
        self.val_dice_coef = []
        self.dice_coef = []
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.accuracies = []
        self.val_accuracies = []
        self.logs = []
        if continue_train:
            try:
                with open("results.json", "r") as f:
                    logs = json.load(f)
                    self.logs = logs
                    self.losses = logs["losses"]
                    self.val_losses = logs["val_losses"]
                    self.accuracies = logs["accuracy"]
                    self.val_accuracies = logs["val_accuracy"]
                    self.i = len(self.losses)
                    self.x = list(range(self.i))
            except Exception as e:
                print("Not possible to continue. Starting from 0", e)

    def get_number_trains(self):
        return self.i - 1

    def on_epoch_end(self, epoch, logs={}):
        print(logs)
        #  self.logs.append(logs)

        self.x.append(self.i)

        self.losses.append(float(logs.get("loss")))
        self.val_losses.append(float(logs.get("val_loss")))
        self.accuracies.append(float(logs.get("acc")))
        self.val_accuracies.append(float(logs.get("val_acc")))
        self.i += 1

        plt.title("model loss")
        plt.plot(self.x, self.losses, color="steelblue", label="train")
        plt.plot(self.x, self.val_losses, color="orange", label="test")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")
        plt.tight_layout()
        plt.gcf().savefig("./model_loss.png")
        plt.clf()

        plt.title("model accuracy")
        plt.plot(self.x, self.accuracies, color="steelblue", label="train")
        plt.plot(self.x, self.val_accuracies, color="orange", label="test")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")
        plt.tight_layout()
        plt.gcf().savefig("./model_accuracy.png")
        plt.clf()

        with open("results.json", "w") as f:
            json.dump(
                {
                    "losses": self.losses,
                    "val_losses": self.val_losses,
                    "accuracy": self.accuracies,
                    "val_accuracy": self.val_accuracies,
                },
                f,
            )


class HDF5Sequence(keras.utils.Sequence):
    def __init__(self, filename, batch_size):
        self.f_array = h5py.File(filename, "r")
        x = self.f_array["images"]
        y = self.f_array["masks"]
        self.batch_size = batch_size
        node_array_size = int(np.ceil(len(x) / hvd.size()))
        self.init_array = hvd.rank() * node_array_size
        self.end_array = self.init_array + node_array_size
        self.x = x
        self.y = y
        print("calculating size")
        print("size", len(self))

    def calc_proportions(self):
        sum_bg = 0.0
        sum_fg = 0.0
        for m in range(self.init_array, self.end_array):
            sum_bg += (self.y[m] < 0.5).sum()
            sum_fg += (self.y[m] >= 0.5).sum()
        return 1.0 - (sum_bg / self.y.size), 1.0 - (sum_fg / self.y.size)

    def __len__(self):
        return int(np.ceil((self.end_array - self.init_array) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.x[self.init_array + idx * self.batch_size :self.init_array +  (idx + 1) * self.batch_size]
        batch_y = self.y[self.init_array + idx * self.batch_size :self.init_array +  (idx + 1) * self.batch_size]

        return np.array([batch_x, batch_y])


def train(kmodel, deepbrain_folder):
    training_files_gen = HDF5Sequence("train_arrays.h5", BATCH_SIZE)
    testing_files_gen = HDF5Sequence("test_arrays.h5", BATCH_SIZE)
    prop_bg, prop_fg = 0.2829173877105532, 0.7170826122894467 

    print("proportion", prop_fg, prop_bg)
    best_model_file = pathlib.Path(
        "weights/weights-improvement-{epoch:03d}.hdf5"
    ).resolve()
    best_model = ModelCheckpoint(
        str(best_model_file), monitor="val_loss", verbose=1, save_best_only=True
    )

    scaled_lr = 1.0 * hvd.size()
    opt = keras.optimizers.Adadelta(lr=scaled_lr)
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
        hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1, initial_lr=scaled_lr),
        # Reduce the learning rate if training plateaues.
        keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1),
        #keras.callbacks.EarlyStopping(
        #    monitor="val_loss", mode="min", patience=20, verbose=True
        #),
    ]

    verbose = 0

    if hvd.rank() == 0:
        verbose = 1
        callbacks.append(best_model)
        callbacks.append(
            keras.callbacks.TensorBoard(
                log_dir="./logs",
                histogram_freq=0,
                batch_size=1,
                write_graph=True,
                write_grads=False,
                write_images=False,
                embeddings_freq=0,
                embeddings_layer_names=None,
                embeddings_metadata=None,
                embeddings_data=None,
                update_freq="batch",
            )
        )

    kmodel.fit_generator(
        training_files_gen,
        epochs=EPOCHS,
        validation_data=testing_files_gen,
        callbacks=callbacks,
        class_weight=np.array((prop_bg, prop_fg)),
        verbose=verbose,
    )


def main():
    kmodel = model.generate_model()
    train(kmodel, pathlib.Path("datasets").resolve())
    model.save_model(kmodel)


if __name__ == "__main__":
    main()
