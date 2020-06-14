import keras
from keras import backend as K
from keras import layers

from constants import SIZE


def custom_loss():
    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true, y_pred):
        return keras.losses.binary_crossentropy(y_true, y_pred)

    # Return a function
    return loss


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
        name='conv1'
    )(input_)
    out = layers.Conv3D(
        filters=8,
        kernel_size=5,
        activation="relu",
        kernel_initializer=init,
        padding="same",
        name='conv2'
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
        name='conv3'
    )(out)
    out = layers.Conv3D(
        filters=16,
        kernel_size=5,
        activation="relu",
        kernel_initializer=init,
        padding="same",
        name='conv4'
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
        name='conv5'
    )(out)
    out = layers.Conv3D(
        filters=32,
        kernel_size=5,
        activation="relu",
        kernel_initializer=init,
        padding="same",
        name='conv6'
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
        name='conv7'
    )(out)
    out = layers.concatenate([out, conv3], axis=-1)
    out = layers.Conv3D(
        filters=32,
        kernel_size=5,
        activation="relu",
        kernel_initializer=init,
        padding="same",
        name='conv8'
    )(out)

    out = layers.Dropout(rate=0.3)(out)

    out = layers.Conv3DTranspose(
        filters=16,
        kernel_size=5,
        strides=2,
        kernel_initializer=init,
        padding="same",
        use_bias=False,
        name='conv9'
    )(out)
    out = layers.concatenate([out, conv2], axis=-1)
    out = layers.Conv3D(
        filters=16,
        kernel_size=5,
        activation="relu",
        kernel_initializer=init,
        padding="same",
        name='conv10'
    )(out)

    out = layers.Dropout(rate=0.3)(out)

    out = layers.Conv3DTranspose(
        filters=8,
        kernel_size=5,
        strides=2,
        kernel_initializer=init,
        padding="same",
        use_bias=False,
        name='conv11'
    )(out)
    out = layers.concatenate([out, conv1], axis=-1)
    out = layers.Conv3D(
        filters=8,
        kernel_size=5,
        activation="relu",
        kernel_initializer=init,
        padding="same",
        name='conv12'
    )(out)

    out = layers.Dropout(rate=0.3)(out)
    out = layers.Conv3D(
        filters=1, kernel_size=1, kernel_initializer=init, padding="same",
        name='conv13'
    )(out)

    #  out = layers.Activation("sigmoid")(out)
    out = layers.Dense(1, activation="sigmoid")(out)

    model = keras.models.Model(input_, out)
    model.compile("adam", loss="binary_crossentropy", metrics=["accuracy"])

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
    #  deepbrain_folder = pathlib.Path(sys.argv[1]).resolve()

    model = generate_model()
    #  train(model, deepbrain_folder)
    save_model(model)


if __name__ == "__main__":
    main()
