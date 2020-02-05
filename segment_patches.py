import itertools
import os
import sys

import nibabel as nb
import numpy as np
from skimage.transform import resize

import model_var as model
from constants import OVERLAP, SIZE


def save_image(image, filename, spacing=(1.0, 1.0, 1.0)):
    image_nifti = nb.Nifti1Image(image, None)
    image_nifti.header.set_zooms(spacing)
    image_nifti.header.set_dim_info(slice=0)
    nb.save(image_nifti, filename)


def main():
    image_filename = sys.argv[1]
    output_filename = sys.argv[2]

    image = nb.load(image_filename).get_fdata()
    image = model.image_normalize(image)
    #  image = image.swapaxes(2, 0)
    mask = np.zeros_like(image, dtype="float32")
    nn_model = model.load_model()

    patch_size = SIZE
    sz, sy, sx = image.shape
    i_cuts = itertools.product(
        range(0, sz, patch_size - OVERLAP),
        range(0, sy, patch_size - OVERLAP),
        range(0, sx, patch_size - OVERLAP),
    )

    sub_image = np.empty(shape=(patch_size, patch_size, patch_size), dtype="float32")
    sub_mask = np.empty_like(sub_image)
    sums = np.zeros_like(image)
    for iz, iy, ix in i_cuts:
        print(iz, iy, ix)
        sub_image[:] = 0
        sub_mask[:] = 0
        _sub_image = image[
            iz : iz + patch_size, iy : iy + patch_size, ix : ix + patch_size
        ]

        _sub_mask = mask[
            iz : iz + patch_size, iy : iy + patch_size, ix : ix + patch_size
        ]

        sums[iz : iz + patch_size, iy : iy + patch_size, ix : ix + patch_size] += 1

        sz, sy, sx = _sub_image.shape

        sub_image[0:sz, 0:sy, 0:sx] = _sub_image
        #  sub_mask[0:sz, 0:sy, 0:sx] = _sub_mask

        sub_mask[:] = nn_model.predict(
            sub_image.reshape(1, patch_size, patch_size, patch_size, 1)
        ).reshape(sub_mask.shape)

        _sub_mask += sub_mask[0:sz, 0:sy, 0:sx]

    print("min_max", mask.min(), mask.max(), sums.min(), sums.max())
    mask = mask / sums
    print("min_max", mask.min(), mask.max(), sums.min(), sums.max())
    save_image(model.image_normalize(mask, 0, 1000), output_filename)


if __name__ == "__main__":
    main()
