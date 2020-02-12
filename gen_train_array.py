import itertools
import os
import pathlib
import random
import sys
from multiprocessing import Pool

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

def gen_all_patches(files, patch_size=SIZE, num_patches=NUM_PATCHES):
    image_filename, mask_filename = files
    image = nb.load(str(image_filename)).get_fdata()
    mask = nb.load(str(mask_filename)).get_fdata()
    transformations = list(itertools.product(range(0, 360, 15), range(0, 360, 15)))
    rot1, rot2 = random.choice(transformations)
    sz, sy, sx = image.shape
    patches = list(itertools.product(
        range(0, sz, patch_size - OVERLAP),
        range(0, sy, patch_size - OVERLAP),
        range(0, sx, patch_size - OVERLAP),
    ))
    random.shuffle(patches)

    image = apply_transform(image, rot1, rot2)
    image = model.image_normalize(image)

    mask = apply_transform(mask, rot1, rot2)
    mask = model.image_normalize(mask)

    images = []
    masks = []
    for patch in patches:
        sub_image = get_image_patch(image, patch, patch_size)
        sub_mask = get_image_patch(mask, patch, patch_size)
        if sub_mask.any():
            images.append(sub_image)
            masks.append(sub_mask)
        if len(masks) == num_patches:
            break

    for patch in patches:
        sub_image = get_image_patch(image, patch, patch_size)
        sub_mask = get_image_patch(mask, patch, patch_size)
        if len(masks) == num_patches:
            break

    return images, masks


def gen_train_array(files, filename, patch_size=SIZE, overlap=OVERLAP, num_patches=NUM_PATCHES):
    size = len(files) * num_patches
    with h5py.File(filename, "w") as f_array:
        images = f_array.create_dataset("images", (size, patch_size, patch_size, patch_size, 1), dtype="float32")
        masks = f_array.create_dataset("masks", (size, patch_size, patch_size, patch_size, 1), dtype="float32")
        with Pool() as pool:
            ip = 0
            for sub_images in pool.imap(gen_all_patches, files):
                for sub_image, sub_mask in zip(*sub_images):
                    images[ip] = sub_image.reshape(patch_size, patch_size, patch_size, 1)
                    masks[ip] = sub_mask.reshape(patch_size, patch_size, patch_size, 1)
                    ip+= 1
                    print(ip, size)



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
