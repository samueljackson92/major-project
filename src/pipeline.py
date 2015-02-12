"""Mammogram Processing Pipeline.

Usage:
  pipeline.py IMAGE [MASK] [--scale-to-mask]
  pipeline.py (-h | --help)
  pipeline.py --version
Options:
  -h --help         Show this screen.
  --version         Show version.
  --scale-to-mask   Scale the image to the mask.

"""
from docopt import docopt

import math
import numpy as np
from skimage import measure, transform, io

from mammogram.orientated_bins import orientated_bins
from mammogram.nonmaximum_suppression import nonmaximum_suppression
import mammogram.plotting as plotting
from mammogram.utils import binary_image, binary_thinning, erode_mask, normalise_image

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

    nbins, size, threshold = 12, 5, 5.0e-2
    line_image, regions = linear_features(img, size, nbins, threshold)
    line_image = np.ma.masked_where(line_image == 0, line_image)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    ax.imshow(line_image, cmap=plt.cm.autumn)
    plt.show()
