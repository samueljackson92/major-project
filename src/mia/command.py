"""
The comannd module provides a small CLI utility that allows detection of image
features directly in the command line. This module is built using the Click
library.
"""

import click
import logging
import pandas as pd

from skimage import transform
import mia

from mia.reduction.reducers import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mia")

LOG_LEVELS = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING
}


@click.group()
@click.version_option(version='0.9.0', prog_name='mia')
@click.option('--log-level', default='info',
              type=click.Choice(['debug', 'info', 'warning']),
              help="Level of logging to report")
def cli(log_level):
    """MIA: Mammogram Image Analysis

    MIA is a python library for analysing mammographic image data.
    """
    logger.setLevel(LOG_LEVELS[log_level])


###############################################################################
# MIA Reduction
###############################################################################


@cli.group()
def reduction():
    """Collection of commands for detecting features across a whole dataset of
    images. Each reduction provides multi-processing support.
    """
    pass


@reduction.command()
@click.argument('image-directory', type=click.Path())
@click.argument('masks-directory', type=click.Path())
@click.argument('output-file', type=click.Path())
@click.option('--num-processes', default=2,
              help="Num of processes to use for the reduction.")
def blob(image_directory, masks_directory, output_file, num_processes):
    """Detect blobs over a dataset of images"""
    reduction = BlobFeaturesReduction(image_directory, masks_directory)
    features = reduction.reduce(num_processes=num_processes)
    features.to_csv(output_file)


@reduction.command()
@click.argument('image-directory', type=click.Path())
@click.argument('masks-directory', type=click.Path())
@click.argument('output-file', type=click.Path())
@click.option('--num-processes', default=2,
              help="Num of processes to use for the reduction.")
def line(image_directory, masks_directory, output_file, num_processes):
    """Detect lines over a dataset of images"""
    reduction = LineFeaturesReduction(image_directory, masks_directory)
    features = reduction.reduce(num_processes=num_processes)
    features.to_csv(output_file)


@reduction.command()
@click.argument('image-directory', type=click.Path())
@click.argument('masks-directory', type=click.Path())
@click.argument('patch_file', type=click.Path())
@click.argument('output-file', type=click.Path())
@click.option('--num-processes', default=2,
              help="Num of processes to use for the reduction.")
def intensity_from_patch(image_directory, masks_directory, patch_file,
                         output_file, num_processes):
    """Detect intensity properties from the histogram of a patch defined by
    blobs/lines.
    """
    patch_frame = pd.DataFrame.from_csv(patch_file)
    reduction = PatchIntensityFeaturesReduction(image_directory,
                                                masks_directory)
    features = reduction.reduce(patch_frame, num_processes=num_processes)
    features.to_csv(output_file)


@reduction.command()
@click.argument('image-directory', type=click.Path())
@click.argument('masks-directory', type=click.Path())
@click.argument('patch_file', type=click.Path())
@click.argument('output-file', type=click.Path())
@click.option('--num-processes', default=2,
              help="Num of processes to use for the reduction.")
def texture_from_patch(image_directory, masks_directory, patch_file,
                       output_file, num_processes):
    """Detect texture properties from the histogram of a patch defined by
    blobs/lines.
    """
    patch_frame = pd.DataFrame.from_csv(patch_file)
    reduction = PatchTextureFeaturesReduction(image_directory,
                                              masks_directory)
    features = reduction.reduce(patch_frame, num_processes=num_processes)
    features.to_csv(output_file)


@cli.command()
@click.argument('image-file', type=click.Path())
@click.argument('mask-file', required=False, type=click.Path())
def detect_blobs(image_file, mask_file):
    """Detect blobs from a single image and display the detected blobs plotted
    on an image.
    """
    img, msk = mia.utils.preprocess_image(image_file, mask_file)
    blob_df = mia.features.blobs.detect_blobs(img, msk)
    mia.plotting.plot_blobs(img, blob_df[['x', 'y', 'radius']].as_matrix())


@cli.command()
@click.argument('image-file', type=click.Path())
@click.argument('mask-file', required=False, type=click.Path())
def detect_lines(image_file, mask_file):
    """Detect lines from a single image and display the detected lines plotted
    on an image.
    """
    img, msk = mia.utils.preprocess_image(image_file, mask_file)
    _, line_image = mia.features.linear_structure.detect_linear(img, msk)
    img = transform.pyramid_reduce(img, 4)
    mia.plotting.plot_linear_structure(img, line_image)


if __name__ == '__main__':
    cli(obj={})
