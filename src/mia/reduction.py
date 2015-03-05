import os.path
import logging
import datetime
import time
import re
import numpy as np
import pandas as pd
import multiprocessing

from mia.features.blobs import blob_features, blob_props
from mia.features.intensity import blob_intensity_props
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
    intensity_props = np.array([blob_intensity_props(blob, img)
                               for blob in blobs])
    props = np.hstack([blobs, intensity_props])
    # tex_props = blob_texture_props(img, blobs, GLCM_FEATURES,
    #                                distances, orientations)
    # props = np.hstack([shape_props, tex_props])

    end = time.time()
    logger.info("%d blobs found in image %s" % (blobs.shape[0], img_name))
    logger.debug("%.2f seconds to process" % (end-start))

    column_names = ['x', 'y', 'radius', 'avg_intensity', 'std_intensity',
                    'skew_intensity', 'kurtosis_intensity']
    blob_df = pd.DataFrame(props, columns=column_names)
    blob_df['image_name'] = pd.Series(np.repeat(img_name, len(blobs)),
                                      index=blob_df.index)

    return blob_df


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

    img_names = feature_matrix['image_name']
    img_classes = [class_hash[transform_name_to_index(v)] for v in img_names]
    feature_matrix['class'] = pd.Series(img_classes,
                                        index=feature_matrix.index)
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

    multiprocessing.freeze_support()
    pool = multiprocessing.Pool(num_processes)
    features = pool.map(multiprocess_images, paths)
    features = pd.concat(features, ignore_index=True)

    return features


def run_reduction(image_directory, masks_directory, output_file, birads_file,
                  num_processes):
    """Run a redcution on an image dataset

    :param image_directory: directory containing the images to process
    :param masks_directory: directory containing the masks for the images
    :param output_file: name of the file to output the results to
    :param birads_file: name fof the file containing the class data from each
                        image
    :param num_processes: number of processes to use in a multiprocess reduction
    """
    start_time = time.time()

    if birads_file is None:
        logger.warning("No BIRADS file supplied. Output will not contain "
                       "risk class labels")
    if output_file is None:
        logger.warning("No output file supplied. Data will not be saved "
                       "to file.")

    feature_matrix = run_multi_process(image_directory, masks_directory,
                                       num_processes, birads_file)

    feature_matrix = add_BIRADS_class(feature_matrix, birads_file)

    if output_file is not None:
        feature_matrix.to_csv(output_file, Header=False)
    else:
        print feature_matrix

    end_time = time.time()
    total_time = end_time - start_time
    total_time = str(datetime.timedelta(seconds=total_time))
    logger.info("TOTAL REDUCTION TIME: %s" % total_time)


def blob_feature_statistics(csv_file, output_file):
    """Create blob features from a file of blob detections

    :param csv_file: file containing the detected blobs
    :param output_file: name to output the resulting features to.
    """
    blobs = pd.DataFrame.from_csv(csv_file)
    image_names = blobs['image_name'].unique()

    shape_props = blob_props(blobs)
    feature_matrix = _create_feature_matrix(shape_props, image_names, None)
    feature_matrix.to_csv(output_file)


def _create_feature_matrix(features, img_names, class_labels_file):
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
