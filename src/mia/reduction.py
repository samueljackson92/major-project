import os.path
import logging
import datetime
import time
import re
import numpy as np
import pandas as pd
import multiprocessing

from mia.features.blobs import detect_blobs, blob_props
from mia.features.intensity import detect_intensity
from mia.io_tools import iterate_directory
from mia.utils import preprocess_image

logger = logging.getLogger(__name__)


def process_image(*args, **kwargs):
    """Process a single image.

    :param image_path: the absolute file path to the image
    :param mask_path: the absolute file path to the mask
    :param scale_to_mask: whether to downscale the image to the mask
    :returns: statistics of the blobs in the image
    """
    img_name = os.path.basename(args[0])

    logger.info("Processing image %s" % img_name)
    start = time.time()

    props = _find_features(*args, **kwargs)

    end = time.time()
    logger.info("%d blobs found in image %s" % (props[0].shape[0], img_name))
    logger.debug("%.2f seconds to process" % (end-start))

    return _make_image_data_frame(img_name, props)


def _find_features(image_path, mask_path, scale_to_mask=False):
    img, msk = preprocess_image(image_path, mask_path,
                                scale_to_mask=scale_to_mask)
    blobs_df = detect_blobs(img, msk)
    intensity_df = detect_intensity(blobs_df, img)
    return [blobs_df, intensity_df]


def _make_image_data_frame(img_name, data_frames):
    img_df = pd.concat(data_frames, axis=1)
    info_df = _make_image_info_data_frame(img_name)
    info_df = pd.concat([info_df]*img_df.shape[0], ignore_index=True)
    return pd.concat([img_df, info_df], axis=1)


def _make_image_info_data_frame(img_name):
    name_regex = re.compile(r'p(\d{3}-\d{3}-\d{5})-([a-z]{2})\.png')

    def find_image_info(name):
        m = re.match(name_regex, name)
        patient_id = m.group(1).replace('-', '')
        view, side = list(m.group(2))
        return [img_name, patient_id, view, side]

    image_info = find_image_info(img_name)
    return pd.DataFrame([image_info], columns=['image_name', 'patient_id',
                                               'view', 'side'])


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

    class_hash = {}
    for img, c in class_labels.iteritems():
        class_hash[img] = c

    img_classes = [class_hash[int(name)]
                   for name in feature_matrix['patient_id']]
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
    :param num_processes: number of processes to use in a multiprocess
                          reduction
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


def reduction_feature_statistics(csv_file, output_file):
    """Create blob features from a file of blob detections

    :param csv_file: file containing the detected blobs
    :param output_file: name to output the resulting features to.
    """
    blobs = pd.DataFrame.from_csv(csv_file)
    image_names = blobs['image_name'].unique()

    info_columns = ['patient_id', 'view', 'side', 'class']
    info_df = blobs[info_columns].drop_duplicates()
    info_df.index = image_names

    shape_props = blob_props(blobs)
    feature_matrix = _create_feature_matrix(shape_props, image_names, None)
    feature_matrix = pd.concat([info_df, feature_matrix], axis=1)
    feature_matrix.to_csv(output_file)


def _create_feature_matrix(features, img_names, class_labels_file):
    """Create a pandas DataFrame for the features

    :param features: numpy array for features
    :param img_names: list of image names to use as the index
    :returns: DataFrame representing the features
    """
    column_names = ['blob_count', 'avg_radius', 'std_radius',
                    'small_radius_count', 'med_radius_count',
                    'large_radius_count', 'density',
                    'avg_avg_intensity', 'avg_std_intensity',
                    'avg_skew_intensity', 'avg_kurt_intensity',
                    'std_avg_intensity', 'std_std_intensity',
                    'std_skew_intensity', 'std_kurt_intensity']

    feature_matrix = pd.DataFrame(features,
                                  index=img_names,
                                  columns=column_names)
    feature_matrix.index.name = 'image_name'
    return feature_matrix
