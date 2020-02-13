import argparse
import itertools
import os

os.environ["KERAS_BACKEND"] = "theano"

parser = argparse.ArgumentParser()

parser.add_argument("--gpu", action="store_true", help="use gpu", dest="use_gpu")
parser.add_argument("-p", action="store_true", help="return probability", dest="ret_prob")
parser.add_argument("-i", "--input", help="Input mri image")
parser.add_argument("-o", "--output", help="Output mri image")
args, _ = parser.parse_known_args()

if args.use_gpu:
    os.environ["THEANO_FLAGS"] = "device=cuda0"

import nibabel as nb
import numpy as np
from skimage.transform import resize

import model
from constants import OVERLAP, SIZE


def save_image(image, filename, affine):
    image_nifti = nb.Nifti1Image(image, affine)
    nb.save(image_nifti, filename)


def main():
    image_filename = args.input
    output_filename = args.output

    image = nb.load(image_filename)
    affine = image.affine
    image = image.get_fdata()
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

        print(iz, iy, ix, sub_mask.min(), sub_mask.max())

        _sub_mask += sub_mask[0:sz, 0:sy, 0:sx]

    print("min_max", mask.min(), mask.max(), sums.min(), sums.max())
    mask = mask / sums
    print("min_max", mask.min(), mask.max(), sums.min(), sums.max())
    if args.ret_prob:
        save_image(model.image_normalize(mask, 0, 1000), output_filename, affine)
    else:
        image[mask < 0.5] = image.min()
        save_image(model.image_normalize(image, 0, 1000), output_filename, affine)


if __name__ == "__main__":
    main()
