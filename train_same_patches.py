import argparse
import itertools
import os
import pathlib
import random
import sys

os.environ["KERAS_BACKEND"] = "theano"

parser = argparse.ArgumentParser()

parser.add_argument("--gpu", action="store_true", help="use gpu", dest="use_gpu")
parser.add_argument("-c", "--continue", action="store_true", dest="continue_train")
args, _ = parser.parse_known_args()

if args.use_gpu:
    os.environ["THEANO_FLAGS"] = "device=cuda0"

import keras
import file_utils

import h5py
import model
import nibabel as nb
import numpy as np
from constants import BATCH_SIZE, EPOCHS, OVERLAP, SIZE, NUM_PATCHES
from keras.callbacks import ModelCheckpoint
from skimage.transform import resize
from utils import apply_transform, image_normalize, PlotLosses



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

def get_image_patch(image, patch, patch_size):
    sub_image = np.zeros(
        shape=(patch_size, patch_size, patch_size), dtype="float32"
    )

    iz, iy, ix = patch

    _sub_image = image[
        iz : iz + patch_size, iy : iy + patch_size, ix : ix + patch_size
    ]

    sz, sy, sx = _sub_image.shape

    sub_image[0:sz, 0:sy, 0:sx] = _sub_image

    return sub_image





def load_models_patches(files, transformations, patch_size=SIZE, batch_size=BATCH_SIZE):
    for image_filename, mask_filename in files:
        image = nb.load(str(image_filename)).get_fdata()
        mask = nb.load(str(mask_filename)).get_fdata()
        image = image_normalize(image)
        mask = image_normalize(mask)
        rot1, rot2 = random.choice(transformations)
        t_image = apply_transform(image, rot1, rot2)
        t_mask = apply_transform(mask, rot1, rot2)

        print(image_filename, mask_filename, rot1, rot2, t_image.min(), t_image.max(), t_mask.min(), t_mask.max())

        for sub_image, sub_mask in gen_patches(t_image, t_mask, patch_size):
            yield (sub_image, sub_mask)


def gen_all_patches(files, transformations, patch_size=SIZE, batch_size=BATCH_SIZE, num_patches=NUM_PATCHES):
    files_transforms_patches = []
    for image_filename, mask_filename in files:
        rot1, rot2 = random.choice(transformations)
        sz, sy, sx = nb.load(str(image_filename)).shape
        i_cuts = itertools.product(
            range(0, sz, patch_size - OVERLAP),
            range(0, sy, patch_size - OVERLAP),
            range(0, sx, patch_size - OVERLAP),
        )
        patches = random.choices(list(i_cuts), k=num_patches)
        for patch in patches:
            files_transforms_patches.append((image_filename, mask_filename, rot1, rot2, patch))

    return files_transforms_patches


def get_proportion(files_transforms_patches, patch_size=SIZE):
    sum_bg = 0.0
    sum_fg = 0.0
    last_filename = ""
    for image_filename, mask_filename, rot1, rot2, patch in files_transforms_patches:
        if image_filename != last_filename:
            mask = nb.load(str(mask_filename)).get_fdata()
            mask = image_normalize(mask)
            mask = apply_transform(mask, rot1, rot2)
            last_filename = image_filename

        _mask = get_image_patch(mask, patch, patch_size)
        sum_bg += (_mask < 0.5).sum()
        sum_fg += (_mask >= 0.5).sum()

    return sum_bg/(sum_fg + sum_bg), sum_fg/(sum_bg + sum_fg)




def gen_train_arrays(files, patch_size=SIZE, batch_size=BATCH_SIZE):
    transformations = list(itertools.product(range(0, 360, 15), range(0, 360, 15)))
    files_transforms_patches = gen_all_patches(files, transformations, patch_size, batch_size, NUM_PATCHES)
    size = len(files_transforms_patches)
    print(size)
    yield int(np.ceil(size / batch_size))
    images = np.zeros(shape=(batch_size, patch_size, patch_size, patch_size, 1), dtype=np.float32)
    masks = np.zeros_like(images)
    yield get_proportion(files_transforms_patches, patch_size)
    ip = 0
    last_filename = ""
    for image_filename, mask_filename, rot1, rot2, patch in itertools.cycle(files_transforms_patches):
        if image_filename != last_filename:
            image = nb.load(str(image_filename)).get_fdata()
            mask = nb.load(str(mask_filename)).get_fdata()
            image = image_normalize(image)
            mask = image_normalize(mask)
            image = apply_transform(image, rot1, rot2)
            mask = apply_transform(mask, rot1, rot2)
            last_filename = image_filename
            print(last_filename)

        images[ip] = get_image_patch(image, patch, patch_size).reshape(1, patch_size, patch_size, patch_size, 1)
        masks[ip] = get_image_patch(mask, patch, patch_size).reshape(1, patch_size, patch_size, patch_size, 1)
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
            image = image_normalize(image)
            mask = image_normalize(mask)
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


def load_train_arrays(images, masks, batch_size=BATCH_SIZE):
    while True:
        for i in range(0, images.shape[0], batch_size):
            yield images[i:i+batch_size], masks[i:i+batch_size]


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



def train(kmodel, deepbrain_folder):
    training_files_gen = HDF5Sequence("train_arrays.h5", BATCH_SIZE)
    testing_files_gen = HDF5Sequence("test_arrays.h5", BATCH_SIZE)
    prop_bg, prop_fg = training_files_gen.calc_proportions()

    print("proportion", prop_fg, prop_bg)

    best_model_file = pathlib.Path("weights/weights.h5").resolve()
    best_model = ModelCheckpoint(
        str(best_model_file), monitor="val_loss", verbose=1, save_best_only=True
    )

    plot_callback = PlotLosses(args.continue_train)
    callbacks = [plot_callback, best_model]
    if args.continue_train:
        kmodel.load_weights("weights/weights.h5")
        initial_epoch = plot_callback.get_number_trains()
        print("starting from", initial_epoch)
        kmodel.fit_generator(
            training_files_gen,
            epochs=EPOCHS,
            validation_data=testing_files_gen,
            callbacks=callbacks,
            class_weight=np.array((prop_bg, prop_fg)),
            initial_epoch=initial_epoch
        )
    else:
        kmodel.fit_generator(
            training_files_gen,
            epochs=EPOCHS,
            validation_data=testing_files_gen,
            callbacks=callbacks,
            class_weight=np.array((prop_bg, prop_fg)),
        )


def main():
    kmodel = model.generate_model()
    train(kmodel, pathlib.Path("datasets").resolve())
    model.save_model(kmodel)


if __name__ == "__main__":
    main()
