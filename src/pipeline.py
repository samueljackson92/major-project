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
from skimage import exposure, measure, transform, io

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from mammogram.blob_detection import blob_detection
from mammogram.orientated_bins import orientated_bins
from mammogram.nonmaximum_suppression import nonmaximum_suppression
import mammogram.plotting as plotting
from mammogram.utils import binary_image, binary_thinning, erode_mask

def linear_features(img, radius, nbins, threshold=15.0):
    """Compute linear features from an image

    Uses orientated bins with nonmaximum suppression and binary thinning.

    :param img: image to find linear features in
    :param radius: radius of the bins to use
    :param nbins: number of bins to use
    :returns: tuple -- containing (line_image, regions)
    """
    line_strength, line_orientation = orientated_bins(img, radius, nbins=nbins)
    line_strength_suppressed = nonmaximum_suppression(line_strength, line_orientation, nbins)

    bimage = binary_image(line_strength_suppressed, threshold)
    line_image = binary_thinning(bimage, 8)

    labelled_image = measure.label(line_image)
    regions = measure.regionprops(labelled_image)

    return line_image, regions

if __name__ == '__main__':
    arguments = docopt(__doc__, version='0.1.0')

    image_path = arguments["IMAGE"]
    mask_path = arguments["MASK"]
    img = io.imread(image_path, as_grey=True)
    # img = exposure.equalize_adapthist(img)

    #scale with gaussian pyramid
    # img = transform.pyramid_reduce(img, downscale=4)

    #mask image
    if mask_path:
        msk = io.imread(mask_path, as_grey=True)
        msk = erode_mask(msk, kernel_size=35)
        msk = transform.rescale(msk,4)
        img = img * msk

    # blob_detection(img, msk)
    # ax.set_title(title)
    # ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    # ax.imshow(line_image, cmap=cm.autumn)
    # for blob in blobs:
    #     y, x, r = blob
    #     c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
    #     ax.add_patch(c)

    # plt.show()



    #
    # import matplotlib.cm as cm
    # io.imshow(img)
    # io.imshow(line_image, cmap=cm.autumn)
    # io.show()

    # plotting.plot_blobs(img, blobs)
    # plot_multiple_images([img, line_image])
    # plot_region_props(line_image, regions)
