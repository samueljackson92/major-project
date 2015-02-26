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

from docopt import docopt
from mammogram.plotting import plot_blobs
from mammogram.blob_detection import blob_detection, blob_props
from mammogram.io import iterate_directory
from mammogram.utils import preprocess_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MIA Pipeline")


def process_blobs(image_dir, mask_dir, output_directory, scale_to_mask=False):
    img_names = []
    features = np.empty((0,5))
    for image_path, mask_path in iterate_directory(image_dir, mask_dir):

        img_name = os.path.basename(image_path)
        img_names.append(img_name)

        logger.info("Processing image %s" % img_name)

        start = time.time()
        img, msk = preprocess_image(image_path, mask_path,
                                    scale_to_mask=scale_to_mask)

        b = blob_detection(img, msk)
        end = time.time()

        props = blob_props(b)
        props = props.reshape(1, props.size)
        features = np.vstack([features, props])

        logger.info("%d blobs found in image" % b.shape[0])
        logger.debug("%.2f seconds to process" % (end-start))

    feature_matrix = create_feature_matrix(features, img_names)
    return feature_matrix

def create_feature_matrix(features, img_names):
    column_names = ['blob_count', 'mean_radius', 'std_radius',
                    'min_radius', 'max_radius']

    feature_matrix = pd.DataFrame(features,
                                  index=img_names,
                                  columns=column_names)
    feature_matrix.index.name = 'image_name'
    feature_matrix.to_csv(output_directory, Header=False)
    return feature_matrix


def process_image(img_path, mask_path):
    img, msk = preprocess_image(img_path, mask_path)
    import time
    start = time.time()
    blobs = blob_detection(img,msk)
    end = time.time()
    logger.debug(end-start)
    plot_blobs(img, blobs)


def main():
    arguments = docopt(__doc__, version='0.4.0')
    image_dir = arguments["IMAGE"]
    mask_dir = arguments["MASK"]
    output_directory = arguments['--output-dir']

    if arguments['--verbose']:
        logger.setLevel(logging.DEBUG)

    # process_image(image_dir, mask_dir)
    feature_matrix = process_blobs(image_dir, mask_dir, output_directory,
                                   arguments['--scale-to-mask'])

if __name__ == "__main__":
    main()
