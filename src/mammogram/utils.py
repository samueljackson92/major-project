import math
import numpy as np
from skimage import morphology


def normalise_image(img, new_min=0, new_max=1):
    """Normalise an image between a range.

    :param new_min: lower bound to normalise to. Default 0
    :param new_min: upper bound to normalise to. Default 1
    :returns: ndarry --  the normalise image
    """
    old_max, old_min = np.amax(img), np.amin(img)
    img = (img - old_min) * (new_max - new_min) / (old_max - old_min) + new_min
    return img


def binary_image(img, threshold):
    """Create a binary image using a threshold.

    Everything greater than the threshold is set to 1, while everything less
    than is set to zero.

    :param img: the image to threshold
    :param threshold: the value to threshold the image with
    :returns: ndarray -- int64 array representing the binary version of the image
    """
    binary_image = np.zeros(img.shape, dtype='int64')
    binary_image[img>threshold] = 1
    return binary_image


def binary_thinning(img, min_object_size):
    """Thin a binary image

    Objects below the value of min_object_size are discarded

    :param img: the binary image on which to perform thinning
    :param min_object_size: the minium size of object to keep
    :returns: ndarry -- thinned binary image
    """
    skeleton = morphology.skeletonize(img)
    morphology.remove_small_objects(skeleton, min_object_size,
                                    connectivity=2, in_place=True)
    return skeleton

def erode_mask(mask, kernel_func=morphology.disk, kernel_size=30):
    """Erode a mask using a kernel

    Uses binary_erosion to erode the edge of a mask.

    :param mask: the mask to erode
    :param kernel_func: the function used to generate the kernel (default: disk)
    :param kernel_size: the size of the kernel to use (default: 30)
    """
    eroded_mask = np.zeros(mask.shape)
    morphology.binary_erosion(mask, kernel_func(kernel_size), out=eroded_mask)
    return eroded_mask

def to_polar_coordinates(x, y):
    """Convert the 2D pixel coordinates to polar coordinates

    :param x: x coordinate of the point
    :param y: y coordinate of the point
    :returns: tuple -- (r,phi) of the point as polar coordinates
    """
    theta = math.atan2(y, x)
    if theta < 0: theta = theta + 2 * math.pi
    return math.hypot(x, y), theta
