import os.path
import logging
import datetime
import time
import re
import numpy as np
import pandas as pd
import multiprocessing

from mia.features.blobs import blob_features, blob_props
# from mia.features.texture import blob_texture_props, GLCM_FEATURES
from mia.io_tools import iterate_directory
from mia.utils import preprocess_image

logger = logging.getLogger(__name__)


def process_image(image_path, mask_path, scale_to_mask=False):
    """Process a single image.

    :param image_path: the absolute file path to the image
    :param mask_path: the absolute file path to the mask
    :param scale_to_mask: whether to downscale the image to the mask
    :returns: statistics of the blobs in the image
    """
    img_name = os.path.basename(image_path)

    # orientations = np.arange(0, np.pi, np.pi/8)
    # distances = [1, 3, 5]

    logger.info("Processing image %s" % img_name)
    start = time.time()

    img, msk = preprocess_image(image_path, mask_path,
                                scale_to_mask=scale_to_mask)
    blobs = blob_features(img, msk)
    shape_props = blob_props(blobs)
    # tex_props = blob_texture_props(img, blobs, GLCM_FEATURES,
    #                                distances, orientations)
    # props = np.hstack([shape_props, tex_props])

    end = time.time()
    logger.info("%d blobs found in image %s" % (blobs.shape[0], img_name))
    logger.debug("%.2f seconds to process" % (end-start))

    return shape_props


def add_BIRADS_class(feature_matrix, class_labels_file):
    """Add the BIRADS classes to the data frame given a file with the class
    labels

    :param feature_matrix: DataFrame for features where the index is the image
                           names
    :param class_labels_file: csv file containg the class labels
    :returns: DataFrame with the class labels added under the column 'class'
    """
    class_labels = pd.DataFrame().from_csv(class_labels_file)
    class_labels = class_labels['BI-RADS']

    name_regex = re.compile(r'p(\d{3}-\d{3}-\d{5})-[a-z]{2}.png')

    class_hash = {}
    for img, c in class_labels.iteritems():
        class_hash[img] = c

    def transform_name_to_index(name):
        return int(re.match(name_regex, name).group(1).replace('-', ''))

    img_names = [transform_name_to_index(v)
                 for v in feature_matrix.index.values]
    img_classes = [class_hash[key] for key in img_names]

    feature_matrix['class'] = pd.Series(img_classes,
                                        index=feature_matrix.index)
    return feature_matrix


def create_feature_matrix(features, img_names, class_labels_file):
    """Create a pandas DataFrame for the features

    :param features: numpy array for features
    :param img_names: list of image names to use as the index
    :returns: DataFrame representing the features
    """
    # texture_prop_names = ["%s_%s" % (prefix, name) for name in GLCM_FEATURES
    #                       for prefix in ['avg', 'std', 'min', 'max']]

    column_names = ['blob_count', 'avg_radius', 'std_radius',
                    'min_radius', 'max_radius']
    # column_names += texture_prop_names

    feature_matrix = pd.DataFrame(features,
                                  index=img_names,
                                  columns=column_names)
    feature_matrix.index.name = 'image_name'

    if class_labels_file is not None:
        feature_matrix = add_BIRADS_class(feature_matrix, class_labels_file)

    return feature_matrix


def multiprocess_images(args):
    """Helper method for multiprocessing images.

    Pass the function arguments to the functions running in the child process
    :param args: arguments to the process_image function
    :returns: result of the process image function
    """
    return process_image(*args)


def run_multi_process(image_dir, mask_dir, num_processes=4,
                      class_labels_file=None):
    """Process a collection of images using multiple process

    :param image_dir: image directory where the data set is stored
    :param mask_dir: mask directory where the data set is stored
    :returns: pandas DataFrame with the features for each image
    """
    paths = [p for p in iterate_directory(image_dir, mask_dir)]
    img_names = [os.path.basename(img_path) for img_path, msk_path in paths]

    multiprocessing.freeze_support()
    pool = multiprocessing.Pool(num_processes)
    features = np.array(pool.map(multiprocess_images, paths))

    return create_feature_matrix(features, img_names, class_labels_file)


def run_reduction(image_directory, masks_directory, output_file, birads_file,
                  num_processes):
    logger.debug("Hi")
    start_time = time.time()

    if birads_file is None:
        logger.warning("No BIRADS file supplied. Output will not contain "
                       "risk class labels")
    if output_file is None:
        logger.warning("No output file supplied. Data will not be saved "
                       "to file.")

    feature_matrix = run_multi_process(image_directory, masks_directory,
                                       num_processes, birads_file)

    if output_file is not None:
        feature_matrix.to_csv(output_file, Header=False)
    else:
        logger.info(feature_matrix)

    end_time = time.time()
    total_time = end_time - start_time
    total_time = str(datetime.timedelta(seconds=total_time))
    logger.info("TOTAL REDUCTION TIME: %s" % total_time)
