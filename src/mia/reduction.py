import os.path
import logging
import datetime
import time
import re
import pandas as pd
import multiprocessing

from mia.features.blobs import detect_blobs, blob_props
from mia.features.intensity import detect_intensity, intensity_props
from mia.io_tools import iterate_directory
from mia.utils import preprocess_image

logger = logging.getLogger(__name__)


def process_image(*args, **kwargs):
    """Process a single image.

    :param image_path: the absolute file path to the image
    :param mask_path: the absolute file path to the mask
    :returns: statistics of the blobs in the image
    """
    img_name = os.path.basename(args[0])

    logger.info("Processing image %s" % img_name)
    start = time.time()

    blobs, intensity = _find_features(*args, **kwargs)

    end = time.time()
    logger.info("%d blobs found in image %s" % (blobs.shape[0], img_name))
    logger.debug("%.2f seconds to process" % (end-start))

    blobs = _make_image_data_frame(img_name, blobs)
    intensity = _make_image_data_frame(img_name, intensity)
    return [blobs, intensity]


def _find_features(image_path, mask_path):
    img, msk = preprocess_image(image_path, mask_path)
    blobs_df = detect_blobs(img, msk)
    intensity_df = detect_intensity(blobs_df, img)
    return [blobs_df, intensity_df]


def _make_image_data_frame(img_name, df):
    info_df = _make_image_info_data_frame(img_name)
    info_df = pd.concat([info_df]*df.shape[0], ignore_index=True)
    frame = pd.concat([df, info_df], axis=1)
    return frame


def _make_image_info_data_frame(img_name):
    _, ext = os.path.splitext(img_name)

    if ext == '.dcm':
        return _make_synthetic_info_data_frame(img_name)
    else:
        return _make_mammogram_info_data_frame(img_name)


def _make_synthetic_info_data_frame(img_name):
    return pd.DataFrame([img_name], columns=['img_name'])


def _make_mammogram_info_data_frame(img_name):
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
    features = list(zip(*features))

    blobs = pd.concat(features[0], ignore_index=True)
    intensity = pd.concat(features[1], ignore_index=True)
    return blobs, intensity


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

    blobs, intensity = run_multi_process(image_directory, masks_directory,
                                         num_processes, birads_file)

    if output_file is not None:
        blobs.to_csv(output_file + '_blobs.csv')
        intensity.to_csv(output_file + '_intenstiy.csv')
    else:
        print blobs
        print intensity

    end_time = time.time()
    total_time = end_time - start_time
    total_time = str(datetime.timedelta(seconds=total_time))
    logger.info("TOTAL REDUCTION TIME: %s" % total_time)


def feature_statistics(raw_detections):
    """Create blob features from a file of blob detections

    :param csv_file: file containing the detected blobs
    :param output_file: name to output the resulting features to.
    """
    # raw_detections = pd.DataFrame.from_csv(csv_file)
    image_names = raw_detections['image_name'].unique()

    info_df = _create_info_data_frame(raw_detections, image_names)
    feature_matrix = _create_feature_matrix(raw_detections, image_names)

    feature_matrix = pd.concat([feature_matrix, info_df], axis=1)
    # feature_matrix.to_csv(output_file)
    return feature_matrix


def _create_info_data_frame(raw_detections, index_names):
    info_columns = ['patient_id', 'view', 'side', 'class']
    info_df = raw_detections[info_columns].drop_duplicates()
    info_df.index = index_names
    return info_df


def _create_feature_matrix(raw_detections, index_names):
    feature_matrix = pd.DataFrame()
    for index, frame in raw_detections.groupby('image_name'):
        shape_props = blob_props(frame)
        int_props = intensity_props(frame)
        row = pd.concat([shape_props, int_props], axis=1)
        feature_matrix = pd.concat([feature_matrix, row], ignore_index=True)

    feature_matrix.index = index_names
    return feature_matrix
