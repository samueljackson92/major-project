import math
import numpy as np
from skimage import morphology

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


def to_polar_coordinates(x, y):
    """Convert the 2D pixel coordinates to polar coordinates

    :param x: x coordinate of the point
    :param y: y coordinate of the point
    :returns: tuple -- (r,phi) of the point as polar coordinates
    """
    theta = math.atan2(y, x)
    if theta < 0: theta = theta + 2 * math.pi
    return math.hypot(x, y), theta
