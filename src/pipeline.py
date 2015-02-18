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

if __name__ == '__main__':
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

    import scipy.ndimage.filters as filters

    def gaussian_kernel(size, fwhm = 3):
        """ Make gaussian kernel."""

        fwhm = 2.355*fwhm

        x = np.arange(0, size, 1, float)
        y = x[:,np.newaxis]

        x0 = y0 = size // 2
        return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

    def log_kernel(size, sigma):
        g = gaussian_kernel(size+1, sigma)
        log = filters.laplace(g, mode='wrap')
        log = log[1:-1,1:-1] #remove the edge crap
        return log

    log = log_kernel(5, 1.5)

    msk = msk.astype('int')
    blob_detection(img, msk)
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
