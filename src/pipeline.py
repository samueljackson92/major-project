"""Mammogram Processing Pipeline.

Usage:
  pipeline.py IMAGE [MASK]
  pipeline.py (-h | --help)
  pipeline.py --version
Options:
  -h --help     Show this screen.
  --version     Show version.

"""
from docopt import docopt

import math
import numpy as np
from skimage import measure, transform, io

from mammogram.orientated_bins import orientated_bins
from mammogram.nonmaximum_suppression import nonmaximum_suppression
from mammogram.plotting import plot_multiple_images, plot_region_props
from mammogram.utils import binary_image, binary_thinning, erode_mask

def linear_features(img, radius, nbins):
    """Compute linear features from an image

    Uses orientated bins with nonmaximum suppression and binary thinning.

    :param img: image to find linear features in
    :param radius: radius of the bins to use
    :param nbins: number of bins to use
    :returns: tuple -- containing (line_image, regions)
    """
    line_strength, line_orientation = orientated_bins(img, radius, nbins=nbins)
    line_strength_suppressed = nonmaximum_suppression(line_strength, line_orientation, nbins)

    line_image = binary_image(line_strength_suppressed, 0.05)
    line_image = binary_thinning(line_image, 8)

    labelled_image = measure.label(line_image)
    regions = measure.regionprops(labelled_image)

    return line_image, regions

if __name__ == '__main__':
    arguments = docopt(__doc__, version='0.1.0')

    image_path = arguments["IMAGE"]
    mask_path = arguments["MASK"]
    img = io.imread(image_path, as_grey=True)

    #scale with gaussian pyramid
    pyramid = transform.pyramid_gaussian(img, downscale=4)
    img = pyramid.next()
    img = pyramid.next()

    #mask image
    if mask_path:
        msk = io.imread(mask_path, as_grey=True)
        msk = erode_mask(msk)
        img = img * msk

    nbins, size = 12, 5
    line_image, regions = linear_features(img, size, nbins)

    plot_region_props(line_image, regions)
