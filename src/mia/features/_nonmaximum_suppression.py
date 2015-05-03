"""
Non-maximum suppression algorithm for selecting the most prominant line
orinetation from a line strength image.

Reference: Reyer Zwiggelaar, Tim C. Parr, and Christopher J. Taylor.
"Finding Orientated Line Patterns in Digital Mammographic Images." BMVC. 1996.
"""
import numpy as np
from scipy.ndimage import filters

__all__ = ['nonmaximum_suppression']


def nonmaximum_suppression(line_strength, line_orientation, nbins,
                           kernel_size=3):
    """Non-maximum suppression of the line strength images.

    :param line_strength: the line strength image to process
    :param line_orientation: the line orientation image to use
    :param nbins: number of bins used to create the line_orientation image
    :param kernel_size: size of the kernel neighbourhood (default 3)
    :returns: ndarray -- the suppressed line strength image
    """
    if (nbins % 4) != 0:
        raise ValueError("nbins must be a multiple of 4")

    kernels = generate_kernels(kernel_size)

    # Convert the line orientation image to the parallel orientation
    parallel_orientation = (line_orientation + (nbins/2)) % nbins
    parallel_orientation = parallel_orientation % len(kernels)

    return filter_for_maximum_direction(kernels, line_strength,
                                        line_orientation)


def filter_for_maximum_direction(kernels, line_strength, parallel_orientation):
    """Filter the line_strength image for the maximum local direction

    :param kernels: the list of kernels to filter the image
    :param line_strength: the line strength image to process
    :param parallel_orientation: the parallel orientation image to the original
                                 line orientation image
    :returns: ndarray -- the suppressed line strength image
    """

    def func(window):
        if np.argmax(window) != 1:
            return 0
        return 1

    filtered_images = []
    for kernel in kernels:
        filtered_image = np.zeros(line_strength.shape)
        filters.generic_filter(line_strength, func, footprint=kernel,
                               output=filtered_image)
        filtered_images.append(filtered_image)

    line_strength_suppressed = np.zeros(line_strength.shape)
    np.choose(parallel_orientation, filtered_images,
              out=line_strength_suppressed)
    return line_strength_suppressed


def generate_kernels(kernel_size):
    """ Make 4 kernels in the horizontal, vertical and diagonal directions.

    :param kernel_size: the size of the kernels to create.
    :returns: list -- the generated kernels
    """
    horizontal = np.zeros(shape=(kernel_size, kernel_size), dtype='int8')
    horizontal[1] = np.ones(kernel_size)

    vertical = np.zeros(shape=(kernel_size, kernel_size), dtype='int8')
    vertical[:, 1] = np.ones(kernel_size)

    left_diagonal = np.eye(kernel_size, dtype='int8')
    right_diagonal = np.fliplr(left_diagonal)

    kernels = np.array([
        horizontal,
        left_diagonal,
        vertical,
        right_diagonal
    ])

    return kernels
