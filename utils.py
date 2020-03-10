import numpy as np
from scipy.ndimage import rotate
from skimage.transform import resize
import pathlib
import os


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


def get_LUT_value(data, window, level):
    shape = data.shape
    data_ = data.ravel()
    data = np.piecewise(
        data_,
        [
            data_ <= (level - 0.5 - (window - 1) / 2),
            data_ > (level - 0.5 + (window - 1) / 2),
        ],
        [0, 255, lambda data_: ((data_ - (level - 0.5)) / (window - 1) + 0.5) * (255)],
    )
    data.shape = shape
    return data


def image_normalize(image, min_=0.0, max_=1.0):
    imin, imax = image.min(), image.max()
    return (image - imin) * ((max_ - min_) / (imax - imin)) + min_


def get_plaidml_devices(gpu=False):
    local_user_plaidml = pathlib.Path("~/.local/share/plaidml/").expanduser().absolute()
    if local_user_plaidml.exists():
        os.environ["RUNFILES_DIR"] = str(local_user_plaidml)
        os.environ["PLAIDML_NATIVE_PATH"] = str(pathlib.Path("~/.local/lib/libplaidml.so").expanduser().absolute())

    import plaidml

    ctx = plaidml.Context()
    plaidml.settings._setup_for_test(plaidml.settings.user_settings)
    plaidml.settings.experimental = True
    devices, _ = plaidml.devices(ctx, limit=100, return_all=True)
    if gpu:
        for device in devices:
            if b"cuda" in device.description.lower():
                return device
        for device in devices:
            if b"opencl" in device.description.lower():
                return device
    for device in devices:
        if b"llvm" in device.description.lower():
            return device
