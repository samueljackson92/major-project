"""
Various IO utility functions.
"""

import os
import re
import json


def _image_name_filter(img_name):
    _, ext = os.path.splitext(img_name)
    if ext == ".png":
        regex = re.compile("p(\d{3}-\d{3}-\d{5}-[a-z]{2})\.png")
    else:
        regex = re.compile("([a-zA-Z_]+\d+_c_\d)\.dcm")

    return re.match(regex, img_name) is not None


def _mask_name_filter(img_name):
    _, ext = os.path.splitext(img_name)
    if ext == ".png":
        regex = re.compile("f(\d{3}-\d{3}-\d{5}-[a-z]{2})_mask\.png")
    else:
        regex = re.compile("([a-zA-Z_]+\d+_c_\d)_mask\.dcm")

    return re.match(regex, img_name) is not None


def iterate_directories(image_directory, mask_directory,
                        image_filter=_image_name_filter,
                        mask_filter=_mask_name_filter):
    """ Iterate of a directory of images

    :param image_directory: the directory to iterate over.
    :param mask_directory: the directory in which to find the corresponding
                           image mask.
    :returns: iterator to the image paths in the directory
    """

    img_iterator = iterate_directory(image_directory, image_filter)
    msk_iterator = iterate_directory(mask_directory, mask_filter)
    for values in zip(img_iterator, msk_iterator):
        yield values


def iterate_directory(directory, filter_func=None):
    check_is_directory(directory)

    for img_name in os.listdir(directory):
        if filter_func is None or filter_func(img_name):
            img_path = os.path.join(directory, img_name)
            yield img_path


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


def dump_mapping_to_json(mapping, columns, output_file):
    """Dump a mapping to JSON data.

    :param mapping: the data frame to store
    :param columns: the columns to use
    :param output_file: the output file to store it in
    """
    groups = mapping.groupby('class')
    json_data = []
    for index, group in groups:
        points = group[columns].as_matrix().tolist()
        img_names = group.index.values.tolist()
        point_data = zip(points, img_names)
        data = [{'x': x, 'y': y, 'name': name} for (x, y), name in point_data]

        g = {
            'name': "BI-RADS Class %d" % index,
            'data': data,
        }
        json_data.append(g)

    with open(output_file, 'wb') as fp:
        json.dump(json_data, fp)
