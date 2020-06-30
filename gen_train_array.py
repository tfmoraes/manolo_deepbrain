import itertools
import os
import pathlib
import random
import sys
from multiprocessing import Pool
from tempfile import mktemp

import h5py
import nibabel as nb
import numpy as np
from scipy.ndimage import rotate
from skimage.transform import resize

import file_utils
from constants import BATCH_SIZE, EPOCHS, NUM_PATCHES, OVERLAP, SIZE, STEP_ROT
from utils import apply_transform, image_normalize


def get_image_patch(image, patch, patch_size):
    sub_image = np.zeros(shape=(patch_size, patch_size, patch_size), dtype="float32")
    iz, iy, ix = patch

    _sub_image = image[iz : iz + patch_size, iy : iy + patch_size, ix : ix + patch_size]

    sz, sy, sx = _sub_image.shape
    sub_image[0:sz, 0:sy, 0:sx] = _sub_image
    return sub_image


def gen_image_patches(files, patch_size=SIZE, num_patches=NUM_PATCHES):
    image_filename, mask_filename = files
    original_image = nb.load(str(image_filename)).get_fdata()
    original_mask = nb.load(str(mask_filename)).get_fdata()

    transformations = list(
        itertools.product(range(0, 360, STEP_ROT), range(0, 360, STEP_ROT))
    )

    patches_files = []
    # Mirroring
    for m in range(4):
        print("m", m)
        if m == 0:
            image = original_image[:].copy()
            mask = original_mask[:].copy()
        if m == 1:
            image = original_image[::-1].copy()
            mask = original_mask[::-1].copy()
        elif m == 2:
            image = original_image[:, ::-1, :].copy()
            mask = original_mask[:, ::-1, :].copy()
        elif m == 3:
            image = original_image[:, :, ::-1].copy()
            mask = original_mask[:, :, ::-1].copy()

        for n in range(6):
            if n == 0:
                rot1, rot2, rot3 = 0, 0
            else:
                rot1 = random.randint(1, 359)
                rot2 = random.randint(1, 359)
                rot3 = random.randint(1, 359)

            patches_added = 0

            _image = apply_transform(image, rot1, rot2, rot3)
            _image = image_normalize(_image)

            _mask = apply_transform(mask, rot1, rot2, rot3)
            _mask = image_normalize(_mask)

            if _image is None or _mask is None:
                continue

            sz, sy, sx = _image.shape
            patches = list(
                itertools.product(
                    range(0, sz, patch_size - OVERLAP),
                    range(0, sy, patch_size - OVERLAP),
                    range(0, sx, patch_size - OVERLAP),
                )
            )
            random.shuffle(patches)

            for patch in patches:
                sub_image = get_image_patch(_image, patch, patch_size)
                sub_mask = get_image_patch(_mask, patch, patch_size)
                if (sub_mask > 0.5).sum() >= (sub_mask.size * 0.1):
                    print("patch gerado")
                    tmp_filename = mktemp(suffix=".npz")
                    np.savez(tmp_filename, image=sub_image, mask=sub_mask)
                    del sub_image
                    del sub_mask
                    patches_files.append(tmp_filename)
                    patches_added += 1
                if patches_added == num_patches:
                    break

    return patches_files


def gen_all_patches(files):
    patches_files = []
    with Pool() as pool:
        for i, image_patches_files in enumerate(pool.imap(gen_image_patches, files)):
            print(i, len(files)) 
            patches_files.extend(image_patches_files)
    return patches_files


def h5file_from_patches(patches_files, filename, patch_size=SIZE):
    with h5py.File(filename, "w") as f_array:
        size = len(patches_files)
        images = f_array.create_dataset(
            "images", (size, patch_size, patch_size, patch_size, 1), dtype="float32"
        )
        masks = f_array.create_dataset(
            "masks", (size, patch_size, patch_size, patch_size, 1), dtype="float32"
        )

        f_array["bg"] = 0
        f_array["fg"] = 0

        for n, patch_file in enumerate(patches_files):
            print("loading", patch_file)
            arr = np.load(patch_file)
            images[n] = arr["image"].reshape(patch_size, patch_size, patch_size, 1)
            masks[n] = arr["mask"].reshape(patch_size, patch_size, patch_size, 1)
            f_array["bg"][()] += (masks[n] < 0.5).sum()
            f_array["fg"][()] += (masks[n] >= 0.5).sum()
            os.remove(patch_file)
            print(n, size)


def main():
    #deepbrain_folder = pathlib.Path("datasets").resolve()
    deepbrain_folder = pathlib.Path("datasets").resolve()
    files = file_utils.get_lidc_filenames(deepbrain_folder)

    print(files, deepbrain_folder)

    #cc359_files = file_utils.get_cc359_filenames(deepbrain_folder)
    #nfbs_files = file_utils.get_nfbs_filenames(deepbrain_folder)
    #files = cc359_files + nfbs_files

    patches_files = gen_all_patches(files)
    random.shuffle(patches_files)

    training_files = patches_files[: int(len(patches_files) * 0.80)]
    testing_files = patches_files[int(len(patches_files) * 0.80) :]

    h5file_from_patches(training_files, "train_arrays.h5")
    h5file_from_patches(testing_files, "test_arrays.h5")


if __name__ == "__main__":
    main()
