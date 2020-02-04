import itertools
import json
import pathlib
import random

import constants
import file_utils
import nibabel as nb


def gen_patches(filename, patch_size, overlap):
    image = nb.load(filename)
    sz, sy, sx = image.shape
    patches = itertools.product(
        range(0, sz, patch_size - overlap),
        range(0, sy, patch_size - overlap),
        range(0, sx, patch_size - overlap),
    )
    return patches


def create_json(dataset_files, patch_size, overlap, json_filename):
    dataset_patches = []
    for image_filename, mask_filename in dataset_files:
        for patches in gen_patches(str(image_filename), patch_size, overlap):
            dataset_patches.append((
                str(image_filename),
                str(mask_filename),
                patches
            ))

    with open(json_filename, "w") as f:
        json.dump(dataset_patches, f)


def main():
    datasets_folder = pathlib.Path("datasets")
    dataset_files = file_utils.get_cc359_filenames(datasets_folder) + file_utils.get_nfbs_filenames(datasets_folder)
    random.shuffle(dataset_files)

    training_files = dataset_files[:int(len(dataset_files) * 0.8 )]
    testing_files = dataset_files[int(len(dataset_files) * 0.8 ):]

    create_json(training_files, constants.SIZE, constants.OVERLAP, "train_files.json")
    create_json(testing_files, constants.SIZE, constants.OVERLAP, "testing_files.json")


if __name__ == "__main__":
    main()
