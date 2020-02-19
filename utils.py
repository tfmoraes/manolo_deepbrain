import json

import keras
import numpy as np
import pylab as plt
from scipy.ndimage import rotate
from skimage.transform import resize


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
            json.dump({
                'losses': self.losses,
                'val_losses': self.val_losses,
                'accuracy': self.accuracies,
                'val_accuracy': self.val_accuracies
            }, f)


def image_normalize(image, min_=0.0, max_=1.0):
    imin, imax = image.min(), image.max()
    return (image - imin) * ((max_ - min_) / (imax - imin)) + min_
