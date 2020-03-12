import argparse
import itertools
import os
import pathlib
import random
import sys
import json

import h5py
import nibabel as nb
import numpy as np
import pylab as plt
import file_utils
from constants import BATCH_SIZE, EPOCHS, OVERLAP, SIZE, NUM_PATCHES
from utils import apply_transform, image_normalize, get_plaidml_devices
from skimage.transform import resize

parser = argparse.ArgumentParser()

parser.add_argument("--gpu", action="store_true", help="use gpu", dest="use_gpu")
parser.add_argument("-c", "--continue", action="store_true", dest="continue_train")
parser.add_argument("-b", "--backend", help="Backend", dest="backend")
args, _ = parser.parse_known_args()

import keras
import model as brain_model
from keras.callbacks import ModelCheckpoint

import ray
from ray.tune import grid_search, run, sample_from
from ray.tune import Trainable
from ray.tune.schedulers import PopulationBasedTraining,  AsyncHyperBandScheduler


#  from ray.tune import track

class TuneReporterCallback(keras.callbacks.Callback):
    """Tune Callback for Keras."""

    def __init__(self, reporter=None, freq="batch", logs={}):
        """Initializer.
        Args:
            reporter (StatusReporter|tune.track.log|None): Tune object for
                returning results.
            freq (str): Sets the frequency of reporting intermediate results.
                One of ["batch", "epoch"].
        """
        self.reporter = reporter or track.log
        self.iteration = 0
        if freq not in ["batch", "epoch"]:
            raise ValueError("{} not supported as a frequency.".format(freq))
        self.freq = freq
        super(TuneReporterCallback, self).__init__()

    def on_batch_end(self, batch, logs={}):
        if not self.freq == "batch":
            return
        self.iteration += 1
        for metric in list(logs):
            if "loss" in metric and "neg_" not in metric:
                logs["neg_" + metric] = -logs[metric]
        if "acc" in logs:
            self.reporter(keras_info=logs, mean_accuracy=logs["acc"])
        else:
            self.reporter(keras_info=logs, mean_accuracy=logs.get("accuracy"))

    def on_epoch_end(self, batch, logs={}):
        if not self.freq == "epoch":
            return
        self.iteration += 1
        for metric in list(logs):
            if "loss" in metric and "neg_" not in metric:
                logs["neg_" + metric] = -logs[metric]
        if "acc" in logs:
            self.reporter(keras_info=logs, mean_accuracy=logs["acc"])
        else:
            self.reporter(keras_info=logs, mean_accuracy=logs.get("accuracy"))


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
        plt.gcf().savefig("/apps/model_loss.png")
        plt.clf()

        plt.title("model accuracy")
        plt.plot(self.x, self.accuracies, color="steelblue", label="train")
        plt.plot(self.x, self.val_accuracies, color="orange", label="test")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")
        plt.tight_layout()
        plt.gcf().savefig("/apps/model_accuracy.png")
        plt.clf()

        with open("/apps/results.json", "w") as f:
            json.dump({
                'losses': self.losses,
                'val_losses': self.val_losses,
                'accuracy': self.accuracies,
                'val_accuracy': self.val_accuracies
            }, f)


def calc_proportions(masks):
    sum_bg = 0.0
    sum_fg = 0.0
    for m in masks:
        sum_bg += (m < 0.5).sum()
        sum_fg += (m >= 0.5).sum()

    return 1.0 - (sum_bg/masks.size), 1.0 - (sum_fg/masks.size)


class HDF5Sequence(keras.utils.Sequence):
    def __init__(self, filename, batch_size):
        self.f_array = h5py.File(filename, "r")
        self.x = self.f_array["images"]
        self.y = self.f_array["masks"]
        self.batch_size = batch_size

    def calc_proportions(self):
        sum_bg = 0.0
        sum_fg = 0.0
        for m in self.y:
            sum_bg += (m < 0.5).sum()
            sum_fg += (m >= 0.5).sum()
        return 1.0 - (sum_bg/self.y.size), 1.0 - (sum_fg/self.y.size)

    def __len__(self):
        return int(np.ceil(self.x.shape[0]/self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([batch_x, batch_y])


class BrainModel(Trainable):
    def _build_model(self):
        return brain_model.generate_model()

    def _setup(self, config):
        batch_size = self.config.get("batch_size", BATCH_SIZE)
        self.train_data = HDF5Sequence("/apps/train_arrays.h5", batch_size)
        self.test_data = HDF5Sequence("/apps/test_arrays.h5", batch_size)
        self.prop_bg, self.prop_fg = self.train_data.calc_proportions()
        print("proportion", self.prop_bg, self.prop_fg)
        optimizer=keras.optimizers.Adam(
            lr=self.config["lr"],
        )
        model = self._build_model()
        model.compile(
            loss="binary_crossentropy",
            optimizer=optimizer,
            metrics=["accuracy"]
        )
        self.model = model

    def _train(self):
        batch_size = self.config.get("batch_size", 32)

        self.model.fit_generator(
            self.train_data,
            epochs=self.config.get("epochs", 1),
            validation_data=None,
            class_weight=np.array((self.prop_bg, self.prop_fg)),
        )

        _, accuracy = self.model.evaluate(self.test_data, verbose=0)

        return {"mean_accuracy": accuracy}

    def _save(self, checkpoint_dir):
        file_path = checkpoint_dir + "/model"
        self.model.save(file_path)
        return file_path

    def _restore(self, path):
        del self.model
        self.model = keras.models.load_model(path)

    def _stop(self):
        pass

if __name__ == "__main__":
    train_spec = {
        "resources_per_trial": {
            "cpu": 1,
            "gpu": 0
        },
        "stop": {
            "mean_accuracy": 0.80,
            "training_iteration": 30,
        },
        "config": {
            "epochs": 1,
            "batch_size": 32,
            "lr": grid_search([10**-4, 10**-5]),
            "decay": sample_from(lambda spec: spec.config.lr / 100.0),
            "dropout": grid_search([0.25, 0.5]),
        },
        "num_samples": 4,
    }

    ray.init(address="200.144.114.144:6379")

    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        metric="mean_accuracy",
        mode="max",
        max_t=400,
        grace_period=20)

    run(
        BrainModel,
        name="brain_model",
        scheduler=sched,
        **train_spec
    )
