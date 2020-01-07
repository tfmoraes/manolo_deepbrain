import numpy as np
from keras.models import Sequential
from keras import layers
from keras import backend as K
import keras

#  from keras.backend.control_flow_ops.array_ops import placeholder_with_default

SIZE=128

def generate_model():
    init = keras.initializers.glorot_uniform()

    training = K.variable(True, name="Training")

    input_ = layers.Input(shape=(SIZE, SIZE, SIZE, 1), dtype="float32", name="img")
    #  input_ = layers.InputLayer(input_tensor=input_)

    out = layers.Conv3D(filters=8, kernel_size=5, activation='relu', kernel_initializer=init, padding="same")(input_)
    out = layers.Conv3D(filters=8, kernel_size=5, activation='relu', kernel_initializer=init, padding="same")(out)

    conv1 = out

    out = layers.MaxPooling3D(pool_size=2, strides=2)(out)
    out = layers.Dropout(rate=0.3)(out)
    out = layers.Conv3D(filters=16, kernel_size=5, activation='relu', kernel_initializer=init, padding="same")(out)
    out = layers.Conv3D(filters=16, kernel_size=5, activation='relu', kernel_initializer=init, padding="same")(out)

    conv2 = out

    out = layers.MaxPooling3D(pool_size=2, strides=2)(out)
    out = layers.Dropout(rate=0.3)(out)
    out = layers.Conv3D(filters=32, kernel_size=5, activation='relu', kernel_initializer=init, padding="same")(out)
    out = layers.Conv3D(filters=32, kernel_size=5, activation='relu', kernel_initializer=init, padding="same")(out)

    conv3 = out

    out = layers.MaxPooling3D(pool_size=2, strides=2)(out)
    out = layers.Dropout(rate=0.3)(out)

    out = layers.Conv3DTranspose(filters=32, kernel_size=5, strides=2, kernel_initializer=init, padding="same", use_bias=False)(out)
    out = layers.concatenate([out, conv3], axis=-1)
    out = layers.Conv3D(filters=32, kernel_size=5, activation="relu", kernel_initializer=init, padding="same")(out)

    out = layers.Dropout(rate=0.3)(out)

    out = layers.Conv3DTranspose(filters=16, kernel_size=5, strides=2, kernel_initializer=init, padding="same", use_bias=False)(out)
    out = layers.concatenate([out, conv2], axis=-1)
    out = layers.Conv3D(filters=16, kernel_size=5, activation="relu", kernel_initializer=init, padding="same")(out)

    out = layers.Dropout(rate=0.3)(out)

    out = layers.Conv3DTranspose(filters=8, kernel_size=5, strides=2, kernel_initializer=init, padding="same", use_bias=False)(out)
    out = layers.concatenate([out, conv1], axis=-1)
    out = layers.Conv3D(filters=8, kernel_size=5, activation="relu", kernel_initializer=init, padding="same")(out)

    out = layers.Dropout(rate=0.3)(out)
    out = layers.Conv3D(filters=1, kernel_size=1, kernel_initializer=init, padding="same")(out)

    out = layers.Activation("sigmoid")(out)

    model = keras.models.Model(input_, out)
    model.compile("Adam", "mean_squared_error")

    return model


def train(model, img, mask):
    z, y, x = img.shape
    model.fit(img.reshape(1, z, y, x, 1), mask.reshape(1, z, y, x, 1))



def main():
    img = np.random.random((SIZE, SIZE, SIZE))
    mask = np.random.random((SIZE, SIZE, SIZE))
    model = generate_model()
    train(model, img, mask)

if __name__ == "__main__":
    main()
