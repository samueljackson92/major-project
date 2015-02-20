"""Mammogram Image Analysis.

Usage:
  mia.py IMAGE [MASK] [--scale-to-mask]
  mia.py (-h | --help)
  mia.py --version
Options:
  -h --help         Show this screen.
  --version         Show version.
  --scale-to-mask   Scale the image to the mask.

"""
from docopt import docopt

import math
import numpy as np
from skimage import exposure, measure, transform, io, morphology
import skimage.filter as filters

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from mammogram.blob_detection import blob_detection
from mammogram.orientated_bins import orientated_bins
from mammogram.nonmaximum_suppression import nonmaximum_suppression
import mammogram.plotting as plotting
from mammogram.utils import *

def linear_features(img, radius, nbins, threshold):
    """Compute linear features from an image

    Uses orientated bins with nonmaximum suppression and binary thinning.

    :param img: image to find linear features in
    :param radius: radius of the bins to use
    :param nbins: number of bins to use
    :returns: tuple -- containing (line_image, regions)
    """
    line_strength, line_orientation = orientated_bins(img, radius, nbins=nbins)
    line_strength_suppressed = nonmaximum_suppression(line_strength, line_orientation, nbins)

    line_image = binary_image(line_strength_suppressed, threshold)
    line_image = skeletonize_image(line_image, 50, dilation_size=1)

    #find image regions
    line_image = measure.label(line_image)
    regions = measure.regionprops(line_image)

    return line_strength_suppressed, regions

def main():
    arguments = docopt(__doc__, version='0.3.0')

    image_path = arguments["IMAGE"]
    mask_path = arguments["MASK"]

    img = io.imread(image_path, as_grey=True)

    if arguments['--scale-to-mask']:
        img = transform.pyramid_reduce(img, downscale=4)

    #mask image
    if mask_path:
        msk = io.imread(mask_path, as_grey=True)
        msk = erode_mask(msk, kernel_size=35)
        if not arguments['--scale-to-mask']:
            msk = transform.rescale(msk,4)
        img = img * msk

    img = normalise_image(img)

    import time
    start = time.time()
    blobs = blob_detection(img, msk)
    end = time.time()
    print "Execution time: %f" % (end-start)

    fig, ax = plt.subplots(1,1)
    ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
        ax.add_patch(c)

    plt.show()

    #
    # import matplotlib.cm as cm
    # io.imshow(img)
    # io.imshow(line_image, cmap=cm.autumn)
    # io.show()

    # plotting.plot_blobs(img, blobs)
    # plot_multiple_images([img, line_image])
    # plot_region_props(line_image, regions)
