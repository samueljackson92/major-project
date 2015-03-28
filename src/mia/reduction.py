import os.path
import logging
import datetime
import time
import re
import functools
import numpy as np
import pandas as pd
import multiprocessing
from skimage import transform

from convolve_tools import deformable_covolution
from mia.features.blobs import detect_blobs
from mia.features.intensity import detect_intensity
from mia.features.texture import texture_from_clusters
from mia.io_tools import iterate_directories
from mia.utils import (preprocess_image, log_kernel, cluster_image,
                       clusters_from_labels, sort_clusters_by_density)

logger = logging.getLogger(__name__)


def _time_reduction(func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        start_time = time.time()
        value = func(*args, **kwargs)
        end_time = time.time()

        total_time = end_time - start_time
        total_time = str(datetime.timedelta(seconds=total_time))
        logger.info("TOTAL REDUCTION TIME: %s" % total_time)
        return value
    return inner


def _time_image_processing(func):
    @functools.wraps(func)
    def inner(img_path, msk_path):
        logger.info("Processing image %s" % os.path.basename(img_path))

        start_time = time.time()
        value = func(img_path, msk_path)
        end_time = time.time()

        logger.info("%.2f seconds to process" % (end_time - start_time))

        return value
    return inner


@_time_image_processing
def blob_features(image_path, mask_path):
    """Process a single image.

    :param image_path: the absolute file path to the image
    :param mask_path: the absolute file path to the mask
    :returns: statistics of the blobs in the image
    """
    img_name = os.path.basename(image_path)

    img, msk = preprocess_image(image_path, mask_path)
    props = detect_blobs(img, msk)
    props['img_name'] = pd.Series([img_name] * props.shape[0])

    logger.info("%d blobs found in image %s" % (props.shape[0], img_name))

    return props


def func_star(func, args):
    """Helper method for multiprocessing images.

    Pass the function arguments to the functions running in the child process

    :param args: arguments to the process_image function
    :returns: result of the process image function
    """
    return func(*args)


def multi_process_images(image_function, paths, num_processes):
    func = functools.partial(func_star, image_function)
    multiprocessing.freeze_support()
    pool = multiprocessing.Pool(num_processes)
    return pool.map(func, paths)


@_time_reduction
def run_blob_reduction(image_dir, mask_dir, num_processes=4):
    """Process a collection of images using multiple process

    :param image_dir: image directory where the data set is stored
    :param mask_dir: mask directory where the data set is stored
    :returns: pandas DataFrame with the features for each image
    """
    paths = [p for p in iterate_directories(image_dir, mask_dir)]
    features = multi_process_images(blob_features, paths, num_processes)
    return pd.concat(features)


@_time_reduction
def run_texture_reduction(image_dir, mask_dir, num_processes=4):

    image_dirs = iterate_directories(image_dir, mask_dir)
    texture_features = []

    for img_path, msk_path in image_dirs:
        img, msk = preprocess_image(img_path, msk_path)
        img_name = os.path.basename(img_path)

        logger.info("Processing image %s" % img_name)
        start = time.time()

        labels = cluster_image(img)
        clusters = clusters_from_labels(img, labels)
        # discard the first cluster which is all zero
        clusters = sort_clusters_by_density(clusters)[1:]
        tex_features = texture_from_clusters(clusters)
        tex_features.index = [img_name]
        texture_features.append(tex_features)

        end = time.time()
        logger.info("%.2f seconds to process" % (end-start))

    return pd.concat(texture_features)


@_time_image_processing
def intensity_features(img_path, msk_path):
    img_name = os.path.basename(img_path)

    img, msk = preprocess_image(img_path, msk_path)
    img = img[msk == 1]
    img_series = pd.Series(img)
    img_described = img_series.describe()

    # create dataframe of histogram features from the described series
    stats = pd.DataFrame(img_described.as_matrix()).T
    stats.columns = img_described.index
    stats.index = [img_name]

    stats['skew'] = img_series.skew()
    stats['kurtosis'] = img_series.kurtosis()

    return stats


@_time_reduction
def run_intensity_reduction(image_dir, mask_dir, num_processes=4):
    paths = [p for p in iterate_directories(image_dir, mask_dir)]
    features = multi_process_images(intensity_features, paths, num_processes)
    return pd.concat(features)


@_time_reduction
def run_raw_reduction(image_directory, masks_directory):
    """Run a raw reduction on an image dataset.

    :param image_directory: directory containing the images to process
    :param masks_directory: directory containing the masks for the images
    :param output_file: name of the file to output the results to
    """
    IMG_SHAPE = (3328, 2560)
    rname = re.compile(r"p(\d{3}-\d{3}-\d{5})-([a-z]{2})\.png")
    kernel = log_kernel(8.0)

    feature_matrix = []
    image_dirs = iterate_directories(image_directory, masks_directory)
    for img_path, msk_path in image_dirs:
        name = os.path.basename(img_path)
        logger.info(name)

        img, msk = preprocess_image(img_path, msk_path)
        # flip images so they follow the same orientation
        match = re.match(rname, name)
        if match and 'r' in match.group(2):
            img = np.fliplr(img)
            msk = np.fliplr(msk)

        img = transform.resize(img, IMG_SHAPE)
        img = transform.pyramid_reduce(img, np.sqrt(2)*7)

        msk = transform.resize(msk, IMG_SHAPE)
        msk = transform.pyramid_reduce(msk, np.sqrt(2)*7)

        msk[msk < 1] = 0
        msk[msk == 1] = 1
        img = img * msk

        img = -deformable_covolution(img, msk, kernel)
        img = img.flatten()

        feature_matrix.append(img)

    return np.vstack(feature_matrix)
