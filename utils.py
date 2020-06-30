import numpy as np
from scipy.ndimage import rotate
from skimage.transform import resize


def apply_transform(image, rot1, rot2):
    if rot1 > 0:
        image = rotate(
            image, angle=rot1, axes=(1, 0), output=np.float32#, order=0, prefilter=False
        )
    if rot2 > 0:
        image = rotate(
            image, angle=rot2, axes=(2, 1), output=np.float32#, order=0, prefilter=False
        )
    return image




def image_normalize(image, min_=0.0, max_=1.0):
    imin, imax = image.min(), image.max()
    if imin == imax:
        print(imin, imax)
    return (image - imin) * ((max_ - min_) / (imax - imin)) + min_


def get_plaidml_devices(gpu=False, _id=0):
    import plaidml
    ctx = plaidml.Context()
    plaidml.settings._setup_for_test(plaidml.settings.user_settings)
    plaidml.settings.experimental = True
    devices, _ = plaidml.devices(ctx, limit=100, return_all=True)
    if gpu:
        for device in devices:
            if b'cuda' in device.description.lower():
                return device
        for device in devices:
            if b'opencl' in device.description.lower() and device.id.endswith(b'%d' % _id):
                return device
    for device in devices:
        if b'llvm' in device.description.lower():
            return device
