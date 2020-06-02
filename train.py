import argparse
import itertools
import os
import pathlib
import random
import sys

os.environ["KERAS_BACKEND"] = "theano"

parser = argparse.ArgumentParser()

parser.add_argument("--gpu", action="store_true", help="use gpu", dest="use_gpu")
parser.add_argument("-c", "--continue", type=int, dest="initial_epoch")
args, _ = parser.parse_known_args()

if args.use_gpu:
    os.environ["THEANO_FLAGS"] = "device=cuda0"

import file_utils
import model
import nibabel as nb
import numpy as np
from constants import BATCH_SIZE, EPOCHS, OVERLAP, SIZE, NUM_PATCHES
from keras.callbacks import ModelCheckpoint
from scipy.ndimage import rotate
from skimage.transform import resize




def apply_transform(image, rot1, rot2):
    if rot1 > 0:
        image = rotate(
            image, angle=rot1, axes=(1, 0), output=np.float32, prefilter=False
        )
    if rot2 > 0:
        image = rotate(
            image, angle=rot2, axes=(2, 1), output=np.float32, prefilter=False
        )
    return image


def get_epoch_size(files, patch_size=SIZE):
    size = 0
    for image_filename, mask_filename in files:
        sz, sy, sx = nb.load(str(image_filename)).shape
        size += int(
            np.ceil(sz / (patch_size - OVERLAP))
            * np.ceil(sy / (patch_size - OVERLAP))
            * np.ceil(sx / (patch_size - OVERLAP))
        )
    return size


def gen_patches(image, mask, patch_size=SIZE):
    sz, sy, sx = image.shape
    i_cuts = itertools.product(
        range(0, sz, patch_size - OVERLAP),
        range(0, sy, patch_size - OVERLAP),
        range(0, sx, patch_size - OVERLAP),
    )


    sub_image = np.empty(
        shape=(patch_size, patch_size, patch_size), dtype="float32"
    )
    sub_mask = np.empty_like(sub_image)
    for iz, iy, ix in random.choices(list(i_cuts), k=NUM_PATCHES):
        sub_image[:] = 0
        sub_mask[:] = 0
        _sub_image = image[
            iz : iz + patch_size, iy : iy + patch_size, ix : ix + patch_size
        ]
        _sub_mask = mask[
            iz : iz + patch_size, iy : iy + patch_size, ix : ix + patch_size
        ]

        sz, sy, sx = _sub_image.shape

        sub_image[0:sz, 0:sy, 0:sx] = _sub_image
        sub_mask[0:sz, 0:sy, 0:sx] = _sub_mask

        yield sub_image, sub_mask


def load_models_patches(files, transformations, patch_size=SIZE, batch_size=BATCH_SIZE):
    for image_filename, mask_filename in files:
        image = nb.load(str(image_filename)).get_fdata()
        mask = nb.load(str(mask_filename)).get_fdata()
        image = model.image_normalize(image)
        mask = model.image_normalize(mask)
        rot1, rot2 = random.choice(transformations)
        t_image = apply_transform(image, rot1, rot2)
        t_mask = apply_transform(mask, rot1, rot2)

        print(image_filename, mask_filename, rot1, rot2, t_image.min(), t_image.max(), t_mask.min(), t_mask.max())

        for sub_image, sub_mask in gen_patches(t_image, t_mask, patch_size):
            yield (sub_image, sub_mask)


def gen_train_arrays(files, patch_size=SIZE, batch_size=BATCH_SIZE):
    transformations = list(itertools.product(range(0, 360, 15), range(0, 360, 15)))
    size = get_epoch_size(files, patch_size)
    yield int(np.ceil(NUM_PATCHES * len(files) / batch_size))
    images = np.zeros(shape=(batch_size, patch_size, patch_size, patch_size, 1), dtype=np.float32)
    masks = np.zeros_like(images)
    ip = 0
    while True:
        for image, mask in load_models_patches(files, transformations, patch_size, batch_size):
            images[ip] = image.reshape(1, patch_size, patch_size, patch_size, 1)
            masks[ip] = mask.reshape(1, patch_size, patch_size, patch_size, 1)
            ip += 1

            if ip == batch_size:
                yield images, masks
                ip = 0


def load_models(files, batch_size=1):
    transformations = list(itertools.product(range(0, 360, 90), range(0, 360, 90)))

    size = len(transformations) * len(files)
    yield int(np.ceil(size / batch_size))

    images = np.zeros((batch_size, SIZE, SIZE, SIZE, 1), dtype="float32")
    masks = np.zeros((batch_size, SIZE, SIZE, SIZE, 1), dtype="float32")
    ip = 0
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
            for rot1, rot2 in transformations:
                t_image = apply_transform(image, rot1, rot2)
                t_mask = apply_transform(mask, rot1, rot2)

                print(image_filename, rot1, rot2)

                images[ip] = t_image.reshape(SIZE, SIZE, SIZE, 1)
                masks[ip] = t_mask.reshape(SIZE, SIZE, SIZE, 1)
                ip += 1

                if ip == batch_size:
                    yield (images, masks)
                    ip = 0


def train(kmodel, deepbrain_folder):
    cc359_files = file_utils.get_cc359_filenames(deepbrain_folder)
    nfbs_files = file_utils.get_nfbs_filenames(deepbrain_folder)
    files = cc359_files  + nfbs_files
    random.shuffle(files)

    training_files = files[: int(len(files) * 0.80)]
    testing_files = files[int(len(files) * 0.80) :]
    training_files_gen = gen_train_arrays(training_files, SIZE, BATCH_SIZE)
    testing_files_gen = gen_train_arrays(testing_files, SIZE, BATCH_SIZE)
    len_training_files = next(training_files_gen)
    len_testing_files = next(testing_files_gen)

    best_model_file = pathlib.Path("weights/weights.h5").resolve()
    best_model = ModelCheckpoint(
        str(best_model_file), monitor="val_loss", verbose=1, save_best_only=True
    )

    if args.initial_epoch:
        kmodel.fit_generator(
            training_files_gen,
            steps_per_epoch=len_training_files,
            epochs=EPOCHS,
            validation_data=testing_files_gen,
            validation_steps=len_testing_files,
            callbacks=[model.PlotLosses(), best_model],
            initial_epoch=args.initial_epoch
        )
    else:
        kmodel.fit_generator(
            training_files_gen,
            steps_per_epoch=len_training_files,
            epochs=EPOCHS,
            validation_data=testing_files_gen,
            validation_steps=len_testing_files,
            callbacks=[model.PlotLosses(), best_model]
        )


def main():
    kmodel = model.generate_model()
    if args.initial_epoch:
        kmodel.load_weights("weights/weights.h5")
    train(kmodel, pathlib.Path("datasets").resolve())
    model.save_model(kmodel)


if __name__ == "__main__":
    main()
