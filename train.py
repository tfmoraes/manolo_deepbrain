import itertools
import pathlib
import random
import sys
import os

os.environ["KERAS_BACKEND"] = "theano"

if len(sys.argv) == 2 and sys.argv[1] == "--gpu":
    os.environ["THEANO_FLAGS"] = "device=cuda0"

import file_utils
import model
import nibabel as nb
import numpy as np
from constants import EPOCHS, SIZE
from keras.callbacks import ModelCheckpoint
from skimage.transform import resize
from scipy.ndimage import rotate


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

    kmodel.fit_generator(
        training_files_gen,
        steps_per_epoch=len_training_files,
        epochs=EPOCHS,
        validation_data=testing_files_gen,
        validation_steps=len_testing_files,
        callbacks=[model.PlotLosses(), best_model],
    )


def main():
    kmodel = model.generate_model()
    train(kmodel, pathlib.Path("datasets").resolve())
    model.save_model(kmodel)


if __name__ == "__main__":
    main()
