import datetime
import itertools
import pathlib
import sys

import keras
import nibabel as nb
import numpy as np
import pylab as plt
from constants import SIZE
from keras import backend as K
from keras import layers
from keras.models import Sequential
from skimage.transform import resize

#  from keras.backend.control_flow_ops.array_ops import placeholder_with_default


class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []

        self.val_dice_coef = []
        self.dice_coef = []

        self.fig = plt.figure()
        self.logs = []

        plt.plot(self.x, self.losses, color="steelblue", label="train")
        plt.plot(self.x, self.val_losses, color="orange", label="test")

        plt.ylabel("loss")
        plt.xlabel("epoch")

        plt.title("model loss")
        plt.legend(["train", "test"], loc="upper left")

    def on_epoch_end(self, epoch, logs={}):

        self.logs.append(logs)

        self.x.append(self.i)

        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))
        self.i += 1

        plt.plot(self.x, self.losses, color="steelblue")
        plt.plot(self.x, self.val_losses, color="orange")

        plt.tight_layout()
        plt.gcf().savefig("./model_loss.png")


def image_normalize(image, min_=0.0, max_=1.0):
    imin, imax = image.min(), image.max()
    return (image - imin) * ((max_ - min_) / (imax - imin)) + min_


def custom_loss():
    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true, y_pred):
        return keras.losses.binary_crossentropy(y_true, y_pred)

    # Return a function
    return loss


def load_models(folder):
    original_folder = folder.joinpath("Original/Original")
    mask_folder = folder.joinpath("Silver-standard-machine-learning/Silver-standard")

    for n in range(200):
        for original_filename in list(original_folder.glob("*.nii.gz"))[:10]:
            mask_filename = mask_folder.joinpath(
                "{}_ss.nii.gz".format(original_filename.stem.split(".")[0])
            )
            print(original_filename, mask_filename)

            image = nb.load(str(original_filename)).get_fdata()
            mask = nb.load(str(mask_filename)).get_fdata()

            image = resize(
                image, (SIZE, SIZE, SIZE), mode="constant", anti_aliasing=True
            )
            mask = resize(mask, (SIZE, SIZE, SIZE), mode="constant", anti_aliasing=True)

            image = image_normalize(image)
            mask = image_normalize(mask)

            print(image.min(), image.max(), mask.min(), mask.max())

            yield image.astype("float32").reshape(1, SIZE, SIZE, SIZE, 1), mask.astype(
                "float32"
            ).reshape(1, SIZE, SIZE, SIZE, 1)

            for i, j in itertools.combinations(range(3), 2):
                print(i, j)
                yield image.astype("float32").swapaxes(i, j).reshape(
                    1, SIZE, SIZE, SIZE, 1
                ), mask.astype("float32").swapaxes(i, j).reshape(1, SIZE, SIZE, SIZE, 1)


def load_models2(folder):
    original_folder = folder.joinpath("Original/Original")
    mask_folder = folder.joinpath("Silver-standard-machine-learning/Silver-standard")

    for n in range(200):
        for original_filename in list(original_folder.glob("*.nii.gz"))[10:20]:
            mask_filename = mask_folder.joinpath(
                "{}_ss.nii.gz".format(original_filename.stem.split(".")[0])
            )
            print(original_filename, mask_filename)

            image = nb.load(str(original_filename)).get_fdata()
            mask = nb.load(str(mask_filename)).get_fdata()

            image = resize(
                image, (SIZE, SIZE, SIZE), mode="constant", anti_aliasing=True
            )
            mask = resize(mask, (SIZE, SIZE, SIZE), mode="constant", anti_aliasing=True)

            image = image_normalize(image)
            mask = image_normalize(mask)

            print(image.min(), image.max(), mask.min(), mask.max())

            yield image.astype("float32").reshape(1, SIZE, SIZE, SIZE, 1), mask.astype(
                "float32"
            ).reshape(1, SIZE, SIZE, SIZE, 1)

            for i, j in itertools.combinations(range(3), 2):
                print(i, j)
                yield image.astype("float32").swapaxes(i, j).reshape(
                    1, SIZE, SIZE, SIZE, 1
                ), mask.astype("float32").swapaxes(i, j).reshape(1, SIZE, SIZE, SIZE, 1)


def generate_model():
    init = keras.initializers.glorot_uniform()

    training = K.variable(True, name="Training")

    input_ = layers.Input(shape=(SIZE, SIZE, SIZE, 1), dtype="float32", name="img")
    #  input_ = layers.InputLayer(input_tensor=input_)

    out = layers.Conv3D(
        filters=8,
        kernel_size=5,
        activation="relu",
        kernel_initializer=init,
        padding="same",
    )(input_)
    out = layers.Conv3D(
        filters=8,
        kernel_size=5,
        activation="relu",
        kernel_initializer=init,
        padding="same",
    )(out)

    conv1 = out

    out = layers.MaxPooling3D(pool_size=2, strides=2)(out)
    out = layers.Dropout(rate=0.3)(out)
    out = layers.Conv3D(
        filters=SIZE//8,
        kernel_size=5,
        activation="relu",
        kernel_initializer=init,
        padding="same",
    )(out)
    out = layers.Conv3D(
        filters=SIZE//8,
        kernel_size=5,
        activation="relu",
        kernel_initializer=init,
        padding="same",
    )(out)

    conv2 = out

    out = layers.MaxPooling3D(pool_size=2, strides=2)(out)
    out = layers.Dropout(rate=0.3)(out)
    out = layers.Conv3D(
        filters=SIZE//4,
        kernel_size=5,
        activation="relu",
        kernel_initializer=init,
        padding="same",
    )(out)
    out = layers.Conv3D(
        filters=SIZE//4,
        kernel_size=5,
        activation="relu",
        kernel_initializer=init,
        padding="same",
    )(out)

    conv3 = out

    out = layers.MaxPooling3D(pool_size=2, strides=2)(out)
    out = layers.Dropout(rate=0.3)(out)

    out = layers.Conv3DTranspose(
        filters=SIZE//4,
        kernel_size=5,
        strides=2,
        kernel_initializer=init,
        padding="same",
        use_bias=False,
    )(out)
    out = layers.concatenate([out, conv3], axis=-1)
    out = layers.Conv3D(
        filters=SIZE//4,
        kernel_size=5,
        activation="relu",
        kernel_initializer=init,
        padding="same",
    )(out)

    out = layers.Dropout(rate=0.3)(out)

    out = layers.Conv3DTranspose(
        filters=SIZE//8,
        kernel_size=5,
        strides=2,
        kernel_initializer=init,
        padding="same",
        use_bias=False,
    )(out)
    out = layers.concatenate([out, conv2], axis=-1)
    out = layers.Conv3D(
        filters= SIZE//8,
        kernel_size=5,
        activation="relu",
        kernel_initializer=init,
        padding="same",
    )(out)

    out = layers.Dropout(rate=0.3)(out)

    out = layers.Conv3DTranspose(
        filters=8,
        kernel_size=5,
        strides=2,
        kernel_initializer=init,
        padding="same",
        use_bias=False,
    )(out)
    out = layers.concatenate([out, conv1], axis=-1)
    out = layers.Conv3D(
        filters=8,
        kernel_size=5,
        activation="relu",
        kernel_initializer=init,
        padding="same",
    )(out)

    out = layers.Dropout(rate=0.3)(out)
    out = layers.Conv3D(
        filters=1, kernel_size=1, kernel_initializer=init, padding="same"
    )(out)

    #  out = layers.Activation("sigmoid")(out)
    out = layers.Dense(1, activation="sigmoid")(out)

    model = keras.models.Model(input_, out)

    return model


def train(model, deepbrain_folder):
    gen_model = load_models(deepbrain_folder)
    val_model = load_models2(deepbrain_folder)
    model.fit_generator(
        gen_model,
        steps_per_epoch=10 * 4,
        epochs=200,
        validation_data=val_model,
        validation_steps=40 * 4,
        callbacks=[PlotLosses()],
    )


def save_model(model):
    """
    Save model and its weights
    """
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


def load_model():
    with open("model.json", "r") as json_file:
        model = keras.models.model_from_json(json_file.read())
    model.load_weights("model.h5")
    model.compile("Adam", "mean_squared_error")
    return model


def main():
    #  img = np.random.random((SIZE, SIZE, SIZE))
    #  mask = np.random.random((SIZE, SIZE, SIZE))
    deepbrain_folder = pathlib.Path(sys.argv[1]).resolve()

    model = generate_model()
    train(model, deepbrain_folder)
    save_model(model)


if __name__ == "__main__":
    main()
