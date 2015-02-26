"""Mammogram Image Analysis.

Usage:
  mia.py IMAGE [MASK] [--scale-to-mask, --output-dir=<output>, --verbose]
  mia.py (-h | --help)
  mia.py --version
Options:
  -h --help                  Show this screen.
  --version                  Show version.
  --verbose                  Turn on debug logging
  --scale-to-mask            Scale the image to the mask.
  -o --output-dir=<output>   Directory to output the results to.

"""
import os.path
import logging
import time
import numpy as np
import pandas as pd
import multiprocessing
import itertools

from docopt import docopt
from mammogram.plotting import plot_blobs
from mammogram.blob_detection import blob_detection, blob_props
from mammogram.io import iterate_directory
from mammogram.utils import preprocess_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MIA Pipeline")


def process_image(image_path, mask_path, scale_to_mask=False):
    """Process a single image.

    :param image_path: the absolute file path to the image
    :param mask_path: the absolute file path to the mask
    :param scale_to_mask: whether to downscale the image to the mask
    :returns: statistics of the blobs in the image
    """
    img_name = os.path.basename(image_path)

    logger.info("Processing image %s" % img_name)

    start = time.time()
    img, msk = preprocess_image(image_path, mask_path,
                                scale_to_mask=scale_to_mask)
    b = blob_detection(img, msk)
    props = blob_props(b)
    end = time.time()

    logger.info("%d blobs found in image" % b.shape[0])
    logger.debug("%.2f seconds to process" % (end-start))

    return props


def create_feature_matrix(features, img_names):
    """Create a pandas DataFrame for the features

    :param features: numpy array for features
    :param img_names: list of image names to use as the index
    :returns: DataFrame representing the features
    """
    column_names = ['blob_count', 'mean_radius', 'std_radius',
                    'min_radius', 'max_radius']

    feature_matrix = pd.DataFrame(features,
                                  index=img_names,
                                  columns=column_names)
    feature_matrix.index.name = 'image_name'
    return feature_matrix


def multiprocess_images(args):
    """Helper method for multiprocessing images.

    Pass the function arguments to the functions running in the child process
    :param args: arguments to the process_image function
    :returns: result of the process image function
    """
    return process_image(*args)


def run_multi_process(image_dir, mask_dir):
    """Process a collection of images using multiple process

    :param image_dir: image directory where the data set is stored
    :param mask_dir: mask directory where the data set is stored
    :returns: pandas DataFrame with the features for each image
    """
    paths = [p for p in iterate_directory(image_dir, mask_dir)]
    img_names = [os.path.basename(img_path) for img_path, msk_path in paths]

    multiprocessing.freeze_support()
    pool = multiprocessing.Pool(4)
    features = np.array(pool.map(multiprocess_images, paths))

    return create_feature_matrix(features, img_names)


def main():
    arguments = docopt(__doc__, version='0.4.0')
    image_dir = arguments["IMAGE"]
    mask_dir = arguments["MASK"]
    output_directory = arguments['--output-dir']

    if arguments['--verbose']:
        logger.setLevel(logging.DEBUG)

    s = time.time()
    feature_matrix = run_multi_process(image_dir, mask_dir)
    e = time.time()
    logger.info("TOTAL PROCESSING TIME: %.2f" % (e-s))

    feature_matrix.to_csv(output_directory, Header=False)


if __name__ == "__main__":
    main()
