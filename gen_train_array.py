import itertools
import os
import pathlib
import random
import sys

import h5py

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
            image, angle=rot1, axes=(1, 0), output=np.float32, order=0, prefilter=False
        )
    if rot2 > 0:
        image = rotate(
            image, angle=rot2, axes=(2, 1), output=np.float32, order=0, prefilter=False
        )
    return image


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

def gen_all_patches(files, transformations, patch_size=SIZE, num_patches=NUM_PATCHES):
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


def gen_train_array(files, filename, patch_size=SIZE, overlap=OVERLAP, num_patches=NUM_PATCHES):
    transformations = list(itertools.product(range(0, 360, 15), range(0, 360, 15)))
    files_transforms_patches = gen_all_patches(files, transformations, patch_size, NUM_PATCHES)
    size = len(files_transforms_patches)

    with h5py.File(filename, "w") as f_array:
        images = f_array.create_dataset("images", (size, patch_size, patch_size, patch_size, 1), dtype="float32")
        masks = f_array.create_dataset("masks", (size, patch_size, patch_size, patch_size, 1), dtype="float32")

        last_filename = ""
        ip = 0
        for image_filename, mask_filename, rot1, rot2, patch in files_transforms_patches:
            if image_filename != last_filename:
                image = nb.load(str(image_filename)).get_fdata()
                mask = nb.load(str(mask_filename)).get_fdata()
                image = model.image_normalize(image)
                mask = model.image_normalize(mask)
                image = apply_transform(image, rot1, rot2)
                mask = apply_transform(mask, rot1, rot2)
                last_filename = image_filename
                print(last_filename)

            images[ip] = get_image_patch(image, patch, patch_size).reshape(patch_size, patch_size, patch_size, 1)
            masks[ip] = get_image_patch(mask, patch, patch_size).reshape(patch_size, patch_size, patch_size, 1)
            ip+= 1



def main():
    deepbrain_folder = pathlib.Path("datasets").resolve()
    cc359_files = file_utils.get_cc359_filenames(deepbrain_folder)
    nfbs_files = file_utils.get_nfbs_filenames(deepbrain_folder)
    files = cc359_files  + nfbs_files
    random.shuffle(files)

    training_files = files[: int(len(files) * 0.80)]
    testing_files = files[int(len(files) * 0.80) :]

    gen_train_array(training_files, "train_arrays.h5", SIZE, OVERLAP, NUM_PATCHES)
    gen_train_array(testing_files, "test_arrays.h5", SIZE, OVERLAP, NUM_PATCHES)


if __name__ == "__main__":
    main()
