import itertools
import pathlib
import random
import sys

import file_utils
import model
import nibabel as nb
from constants import EPOCHS, SIZE
from keras.callbacks import ModelCheckpoint
from skimage.transform import resize


def load_models(files):
    swaps = itertools.cycle(
        ((None, None),) + tuple(itertools.combinations(range(3), 2))
    )
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

            i, j = next(swaps)

            print(image_filename, image.min(), image.max(), mask.min(), mask.max())

            if i is None:
                yield image.astype("float32").reshape(
                    1, SIZE, SIZE, SIZE, 1
                ), mask.astype("float32").reshape(1, SIZE, SIZE, SIZE, 1)
            else:
                yield image.astype("float32").swapaxes(i, j).reshape(
                    1, SIZE, SIZE, SIZE, 1
                ), mask.astype("float32").swapaxes(i, j).reshape(1, SIZE, SIZE, SIZE, 1)


def train(kmodel, deepbrain_folder):
    cc359_files = file_utils.get_cc359_filenames(deepbrain_folder)
    nfbs_files = file_utils.get_nfbs_filenames(deepbrain_folder)
    files = cc359_files + nfbs_files
    random.shuffle(files)

    training_files = files[: int(len(files) * 0.75)]
    testing_files = files[int(len(files) * 0.75) :]
    training_files_gen = load_models(training_files)
    testing_files_gen = load_models(testing_files)

    best_model_file = pathlib.Path("weights/weights.h5").resolve()
    best_model = ModelCheckpoint(
        str(best_model_file), monitor="val_loss", verbose=1, save_best_only=True
    )

    kmodel.fit_generator(
        training_files_gen,
        steps_per_epoch=len(training_files),
        epochs=EPOCHS,
        validation_data=testing_files_gen,
        validation_steps=len(testing_files),
        callbacks=[model.PlotLosses(), best_model],
    )


def main():
    kmodel = model.generate_model()
    train(kmodel, pathlib.Path("datasets").resolve())


if __name__ == "__main__":
    main()
