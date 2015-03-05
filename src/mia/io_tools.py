"""
Various IO utility functions.
"""

import os
import re


def iterate_directory(directory, mask_directory=None):
    """ Iterate of a directory of images

    :param directory: the directory to iterate over.
    :param mask_directory: the directory in which to find the corresponding
                           image mask.
    :returns: iterator to the image paths in the directory
    """
    regex = re.compile("p(\d{3}-\d{3}-\d{5}-[a-z]{2})\.png")
    check_is_directory(directory)

    for img_name in os.listdir(directory):
        match = re.match(regex, img_name)
        if match is not None:
            img_path = os.path.join(directory, img_name)
            check_is_image(img_path, ".png")

            msk_path = None
            if mask_directory is not None:
                msk_name = "f%s_mask.png" % match.group(1)
                msk_path = os.path.join(mask_directory, msk_name)
                check_is_image(msk_path, ".png")

            yield img_path, msk_path


def check_is_directory(directory):
    """Check that the specified path is a directory

    :param directory: path to check if it is a directory
    :raises: ValueError
    """
    if not os.path.isdir(directory):
        raise ValueError("%s is not a directory" % directory)


def check_is_image(img_path, ext):
    """Check that the specified path is an image with the expected extension

    :param directory: path to check if it is a image
    :raises: ValueError
    """
    if not os.path.isfile(img_path):
        raise ValueError("%s is not a file" % img_path)

    if not img_path.endswith(ext):
        raise ValueError("%s does not have the expected file extension"
                         % img_path)
