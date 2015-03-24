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
from mia.io_tools import iterate_directories
from mia.utils import preprocess_image, log_kernel

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


def process_image(*args, **kwargs):
    """Process a single image.

    :param image_path: the absolute file path to the image
    :param mask_path: the absolute file path to the mask
    :returns: statistics of the blobs in the image
    """
    img_name = os.path.basename(args[0])

    logger.info("Processing image %s" % img_name)
    start = time.time()

    props = _find_features(*args, **kwargs)

    end = time.time()
    logger.info("%d blobs found in image %s" % (props[0].shape[0], img_name))
    # logger.info("%d lines found in image %s" % (props[1].shape[0], img_name))
    logger.debug("%.2f seconds to process" % (end-start))

    for df in props:
        df['img_name'] = pd.Series([img_name]*df.shape[0])

    return props


def _find_features(image_path, mask_path):
    img, msk = preprocess_image(image_path, mask_path)
    blobs_df = detect_blobs(img, msk)
    # linear_df, _ = detect_linear(img, msk)
    intensity_df = detect_intensity(blobs_df, img)
    return [blobs_df, intensity_df]


def multiprocess_images(args):
    """Helper method for multiprocessing images.

    Pass the function arguments to the functions running in the child process

    :param args: arguments to the process_image function
    :returns: result of the process image function
    """
    return process_image(*args)


@_time_reduction
def run_multi_process(image_dir, mask_dir, num_processes=4):
    """Process a collection of images using multiple process

    :param image_dir: image directory where the data set is stored
    :param mask_dir: mask directory where the data set is stored
    :returns: pandas DataFrame with the features for each image
    """
    paths = [p for p in iterate_directories(image_dir, mask_dir)]
    multiprocessing.freeze_support()
    pool = multiprocessing.Pool(num_processes)
    feature_frames = pool.map(multiprocess_images, paths)
    feature_frames = list(zip(*feature_frames))

    props = [pd.concat(frames, ignore_index=True) for frames in feature_frames]
    return props


@_time_reduction
def run_raw_reduction(image_directory, masks_directory, img_filter, msk_filter):
    """Run a raw reduction on an image dataset.

    :param image_directory: directory containing the images to process
    :param masks_directory: directory containing the masks for the images
    :param output_file: name of the file to output the results to
    """
    IMG_SHAPE = (3328, 2560)
    rname = re.compile(r"p(\d{3}-\d{3}-\d{5})-([a-z]{2})\.png")
    kernel = log_kernel(8.0)

    feature_matrix = []
    image_dirs = iterate_directories(image_directory, masks_directory,
                                     img_filter, msk_filter)
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
