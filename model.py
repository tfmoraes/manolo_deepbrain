import pathlib
import sys

from constants import SIZE

import keras
import nibabel as nb
import numpy as np
from keras import backend as K
from keras import layers
from keras.models import Sequential
from skimage.transform import resize

#  from keras.backend.control_flow_ops.array_ops import placeholder_with_default

def load_models(folder):
    original_folder = folder.joinpath("Original/Original")
    mask_folder = folder.joinpath("Silver-standard-machine-learning/Silver-standard")

    for original_filename in original_folder.glob("*.nii.gz"):
        mask_filename = mask_folder.joinpath(
            "{}_ss.nii.gz".format(original_filename.stem.split(".")[0])
        )
        print(original_filename, mask_filename)

        image = nb.load(str(original_filename)).get_fdata()
        mask = nb.load(str(mask_filename)).get_fdata()

        image = resize(image, (SIZE, SIZE, SIZE), mode="constant", anti_aliasing=True)
        mask = resize(mask, (SIZE, SIZE, SIZE), mode="constant", anti_aliasing=True)

        image = image / image.max()
        mask = mask / mask.max()

        yield image.astype("float32").reshape(1, SIZE, SIZE, SIZE, 1), mask.astype(
            "float32"
        ).reshape(1, SIZE, SIZE, SIZE, 1)


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
        filters=16,
        kernel_size=5,
        activation="relu",
        kernel_initializer=init,
        padding="same",
    )(out)
    out = layers.Conv3D(
        filters=16,
        kernel_size=5,
        activation="relu",
        kernel_initializer=init,
        padding="same",
    )(out)

    conv2 = out

    out = layers.MaxPooling3D(pool_size=2, strides=2)(out)
    out = layers.Dropout(rate=0.3)(out)
    out = layers.Conv3D(
        filters=32,
        kernel_size=5,
        activation="relu",
        kernel_initializer=init,
        padding="same",
    )(out)
    out = layers.Conv3D(
        filters=32,
        kernel_size=5,
        activation="relu",
        kernel_initializer=init,
        padding="same",
    )(out)

    conv3 = out

    out = layers.MaxPooling3D(pool_size=2, strides=2)(out)
    out = layers.Dropout(rate=0.3)(out)

    out = layers.Conv3DTranspose(
        filters=32,
        kernel_size=5,
        strides=2,
        kernel_initializer=init,
        padding="same",
        use_bias=False,
    )(out)
    out = layers.concatenate([out, conv3], axis=-1)
    out = layers.Conv3D(
        filters=32,
        kernel_size=5,
        activation="relu",
        kernel_initializer=init,
        padding="same",
    )(out)

    out = layers.Dropout(rate=0.3)(out)

    out = layers.Conv3DTranspose(
        filters=16,
        kernel_size=5,
        strides=2,
        kernel_initializer=init,
        padding="same",
        use_bias=False,
    )(out)
    out = layers.concatenate([out, conv2], axis=-1)
    out = layers.Conv3D(
        filters=16,
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

    out = layers.Activation("sigmoid")(out)

    model = keras.models.Model(input_, out)
    model.compile("Adam", "mean_squared_error")

    return model


def train(model, deepbrain_folder):
    gen_model = load_models(deepbrain_folder)
    model.fit_generator(gen_model, steps_per_epoch=359)


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


def main():
    #  img = np.random.random((SIZE, SIZE, SIZE))
    #  mask = np.random.random((SIZE, SIZE, SIZE))
    deepbrain_folder = pathlib.Path(sys.argv[1]).resolve()

    model = generate_model()
    train(model, deepbrain_folder)
    save_model(model)


if __name__ == "__main__":
    main()
