import sys

import nibabel as nb
from skimage.transform import resize

import model
from constants import SIZE


def save_image(image, filename, spacing=(1.0, 1.0, 1.0)):
    image_nifti = nb.Nifti1Image(image, None)
    image_nifti.header.set_zooms(spacing)
    image_nifti.header.set_dim_info(slice=0)
    nb.save(image_nifti, filename)


def main():
    image_filename = sys.argv[1]
    output_filename = sys.argv[2]

    image = nb.load(image_filename).get_fdata()
    image = image.swapaxes(2, 0)
    nn_model = model.load_model()

    img = resize(image, (SIZE, SIZE, SIZE), mode="constant", anti_aliasing=True)
    img = img / img.max()
    img = img.astype("float32").reshape(1, SIZE, SIZE, SIZE, 1)

    print(img.max(), img.min(), img.shape)
    mask = nn_model.predict(img)
    print(mask.max(), mask.min(), mask.shape)
    mask = resize(
        mask.reshape(SIZE, SIZE, SIZE), image.shape, mode="constant", anti_aliasing=True
    )


    #  mask = (mask > 0.5) * 255
    image[mask < 0.5] = image.min()
    save_image(image, output_filename)


if __name__ == "__main__":
    main()
