import math
import os.path
import warnings
import numpy as np
from medpy.io import load
import skimage
from skimage import io, transform, morphology, measure


def preprocess_image(image_path, mask_path=None):
    """Preprocess an image, optionally using a mask.

    :param image_path: path to the image.
    :param mask_path: path to the mask to use (optional).
    """

    name, ext = os.path.splitext(image_path)
    if ext == ".dcm":
        img = load_synthetic_mammogram(image_path)
    else:
        img = load_real_mammogram(image_path, mask_path)

    msk = None
    if mask_path is not None:
        msk = load_mask(mask_path)
        msk = resize_mask_to_image(msk, img.shape)
        img = img * msk

    return img, msk


def load_synthetic_mammogram(image_path):
    image_data, image_header = load(image_path)
    img = np.invert(image_data)
    img = skimage.img_as_float(img)
    return img


def load_real_mammogram(image_path, mask_path=None):
    img = io.imread(image_path, as_grey=True)
    img = skimage.img_as_float(img)
    return img


def load_mask(mask_path):
    msk = io.imread(mask_path, as_grey=True)
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            msk = skimage.img_as_uint(msk)   # cast from uint32 to unit16
            msk = skimage.img_as_float(msk)  # cast from uint16 to float
        except Warning:
            # Warning precision loss is ok. We only need binary info for mask
            pass
    msk = erode_mask(msk, kernel_size=35)
    return msk


def resize_mask_to_image(msk, img_shape):
    msk = transform.resize(msk, img_shape)
    msk[msk > 0] = 1
    msk[msk == 0] = 0
    return msk


def normalise_image(img, new_min=0, new_max=1):
    """Normalise an image between a range.

    :param new_min: lower bound to normalise to. Default 0
    :param new_min: upper bound to normalise to. Default 1
    :returns: ndarry --  the normalise image
    """

    old_max, old_min = img.max(), img.min()
    img = (img - old_min) * (new_max - new_min) / (old_max - old_min) + new_min
    return img


def binary_image(img, threshold):
    """Create a binary image using a threshold.

    Everything greater than the threshold is set to 1, while everything less
    than is set to zero.

    :param img: the image to threshold
    :param threshold: the value to threshold the image with
    :returns: ndarray -- int64 array representing the binary version of the
              image
    """
    binary_image = np.zeros(img.shape, dtype='uint8')
    binary_image[img > threshold] = 1
    return binary_image


def skeletonize_image(img, min_object_size, dilation_size=3):
    """Convert a binary image to skeleton representation.

    This will remove any small artifacts below min_object_size. Then the
    remaining artifacts will be dilated to produce better connectivity. The
    result is skeletonized to produce the final image.

    :param min_object_size: minimum size of artifact to keep
    :param dilation_size: radius of the disk kernel to use for dilation.
    :returns: ndarray -- skeletonized image.
    """
    img = measure.label(img)
    img = morphology.remove_small_objects(img, min_object_size, connectivity=4)

    # dilate to connect bigger structures
    dilation_kernel = morphology.disk(dilation_size)
    img = morphology.binary_closing(img, dilation_kernel)

    img[img > 0] = 1

    return img


def erode_mask(mask, kernel_func=morphology.disk, kernel_size=30):
    """Erode a mask using a kernel

    Uses binary_erosion to erode the edge of a mask.

    :param mask: the mask to erode
    :param kernel_func: the function used to generate the kernel
                        (default: disk)
    :param kernel_size: the size of the kernel to use (default: 30)
    """
    eroded_mask = np.zeros(mask.shape)
    kernel = kernel_func(kernel_size)
    morphology.binary_erosion(mask, kernel, out=eroded_mask)
    return eroded_mask


def to_polar_coordinates(x, y):
    """Convert the 2D pixel coordinates to polar coordinates

    :param x: x coordinate of the point
    :param y: y coordinate of the point
    :returns: tuple -- (r,phi) of the point as polar coordinates
    """
    theta = math.atan2(y, x)
    if theta < 0:
        theta = theta + 2 * math.pi
    return math.hypot(x, y), theta


def transform_2d(f, grid, *args):
    """Apply a function to every element in a 2d array

    :param f: function to apply
    :param grid: array to apply the function too. Must be 2D.
    :param args: addtional arguments to the function
    :returns: ndarray - of transformed data
    """
    out_grid = []
    for row in grid:
        out_row = []
        for value in row:
            out_value = f(value, *args)
            out_row.append(out_value)
        out_grid.append(out_row)
    return np.array(out_grid)


def vectorize_array(f, array, *args):
    """ Helper function to vectorize across the rows of a 2D numpy array

    :params f: function to vectorize
    :params array: 2darray to iterate over.
    :params args: list of arguments to pass to the function f
    :returns: ndarray of the results of applying the function to each row.
    """
    return np.array([f(row, *args) for row in array])
