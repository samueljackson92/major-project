"""
Various IO utility functions.
"""

import os
import json


def iterate_directories(image_directory, mask_directory):
    """ Iterate of a directory of images

    :param image_directory: the directory to iterate over.
    :param mask_directory: the directory in which to find the corresponding
                           image mask.
    :returns: iterator to the image paths in the directory
    """

    img_iterator = iterate_directory(image_directory)
    msk_iterator = iterate_directory(mask_directory)
    for values in zip(img_iterator, msk_iterator):
        yield values


def iterate_directory(directory, filter_func=None):
    check_is_directory(directory)

    for img_name in sorted(os.listdir(directory)):
        img_path = os.path.join(directory, img_name)
        if check_is_file(img_path, '.png', '.dcm'):
            yield img_path


def check_is_directory(directory):
    """Check that the specified path is a directory

    :param directory: path to check if it is a directory
    :raises: ValueError
    """
    if not os.path.isdir(directory):
        raise ValueError("%s is not a directory" % directory)


def check_is_file(img_path, *exts):
    """Check that the specified path has the expected extension

    :param img_path: path to check if it is a image
    :param exts: expected extensions of the file
    """
    return os.path.isfile(img_path) and any([img_path.endswith(ext)
                                             for ext in exts])


def dump_mapping_to_json(mapping, columns, class_labels, output_file):
    """Dump a mapping to JSON data.

    :param mapping: the data frame to store
    :param columns: the columns to use
    :param output_file: the output file to store it in
    """
    colours = ['#0035E5', '#00D86E', '#97CB00', '#BF0000']
    groups = mapping.groupby(class_labels)
    json_data = []
    for i, (class_label, group) in enumerate(groups):
        points = group[columns].as_matrix().tolist()
        img_names = group.index.values.tolist()
        point_data = zip(points, img_names)
        data = [{'x': x, 'y': y, 'name': name} for (x, y), name in point_data]

        g = {
            'name': "Class: %d" % class_label,
            'color': colours[i],
            'data': data,
            'marker': {
                'symbol': 'circle'
            }
        }
        json_data.append(g)

    with open(output_file, 'wb') as fp:
        json.dump(json_data, fp)
