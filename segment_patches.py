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
from utils import image_normalize


def save_image(image, filename, affine):
    image_nifti = nb.Nifti1Image(image, affine)
    nb.save(image_nifti, filename)


def predict_patch(sub_image, patch, nn_model, patch_size=SIZE):
    (iz, ez), (iy, ey), (ix, ex) = patch
    sub_mask = nn_model.predict(sub_image.reshape(1, patch_size, patch_size, patch_size, 1))
    return sub_mask.reshape(patch_size, patch_size, patch_size)[0:ez-iz, 0:ey-iy, 0:ex-ix]


def gen_patches(image, patch_size, overlap):
    sz, sy, sx = image.shape
    i_cuts = itertools.product(
        range(0, sz, patch_size - OVERLAP),
        range(0, sy, patch_size - OVERLAP),
        range(0, sx, patch_size - OVERLAP),
    )
    sub_image = np.empty(shape=(patch_size, patch_size, patch_size), dtype="float32")
    for iz, iy, ix in i_cuts:
        sub_image[:] = 0
        _sub_image = image[
            iz : iz + patch_size, iy : iy + patch_size, ix : ix + patch_size
        ]
        sz, sy, sx = _sub_image.shape
        sub_image[0:sz, 0:sy, 0:sx] = _sub_image
        ez = iz + sz
        ey = iy + sy
        ex = ix + sx

        yield ((iz, ez), (iy, ey), (ix, ex)), sub_image



def segment_on_cpu(image, nn_model):
    pass

def segment_on_gpu(image, nn_model):
    mask = np.zeros_like(image, dtype="float32")
    sums = np.zeros_like(image)
    for patch, sub_image in gen_patches(image, SIZE, OVERLAP):
        (iz, ez), (iy, ey), (ix, ex) = patch
        sub_mask = predict_patch(sub_image, patch, nn_model, SIZE)
        mask[iz:ez, iy:ey, ix:ex] += sub_mask
        sums[iz:ez, iy:ey, ix:ex] += 1

    return mask / sums



def main():
    image_filename = args.input
    output_filename = args.output

    image = nb.load(image_filename)
    affine = image.affine
    image = image.get_fdata()
    image = image_normalize(image)
    nn_model = model.load_model()

    mask = segment_on_gpu(image, nn_model)
    if args.ret_prob:
        save_image(image_normalize(mask, 0, 1000), output_filename, affine)
    else:
        image[mask < 0.5] = image.min()
        save_image(image_normalize(image, 0, 1000), output_filename, affine)


if __name__ == "__main__":
    main()
