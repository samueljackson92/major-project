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
import numpy as np
import pandas as pd

from docopt import docopt
from mammogram.blob_detection import blob_detection, blob_props
from mammogram.io import iterate_directory
from mammogram.utils import preprocess_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MIA Pipeline")


def process_blobs(image_dir, mask_dir, scale_to_mask=False):
    features = np.empty((0,5))
    img_names = []
    for image_path, mask_path in iterate_directory(image_dir, mask_dir):
        img_name = os.path.basename(image_path)
        img_names.append(img_name)

        logger.info("Processing image %s" % img_name)

        img, msk = preprocess_image(image_path, mask_path,
                                    scale_to_mask=scale_to_mask)

        b = blob_detection(img, msk)
        props = blob_props(b)
        props = props.reshape(1, props.size)
        features = np.vstack([features, props])

        logger.info("%d blobs found in image" % b.shape[0])

    column_names = ['blob_count', 'mean_radius', 'std_radius',
                    'min_radius', 'max_radius']
    feature_matrix = pd.DataFrame(features,
                                  index = img_names,
                                  columns=column_names)
    feature_matrix.index.name = 'image_name'
    return feature_matrix


def main():
    arguments = docopt(__doc__, version='0.4.0')
    image_dir = arguments["IMAGE"]
    mask_dir = arguments["MASK"]
    output_directory = arguments['--output-dir']

    if arguments['--verbose']:
        logger.setLevel(logging.DEBUG)

    feature_matrix = process_blobs(image_dir, mask_dir,
                                   arguments['--scale-to-mask'])
    feature_matrix.to_csv(output_directory)
