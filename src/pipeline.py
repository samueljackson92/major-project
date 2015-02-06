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

import numpy as np
from skimage import measure, transform, io

from mammogram.orientated_bins import orientated_bins
from mammogram.nonmaximum_suppression import nonmaximum_suppression
from mammogram.plotting import plot_multiple_images

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
        img = img * msk

    nbins, size = 12, 5
    line_strength, line_orientation = orientated_bins(img, size, nbins=nbins)
    line_strength_suppressed = nonmaximum_suppression(line_strength, line_orientation, nbins)

    from skimage import morphology
    threshold = 0.05
    binary_image = np.zeros(line_strength_suppressed.shape)
    binary_image[line_strength_suppressed>threshold] = 1

    skeleton = morphology.skeletonize(binary_image)
    morphology.remove_small_objects(skeleton, 8, connectivity=2, in_place=True)

    labelled_image = measure.label(skeleton)
    props = measure.regionprops(labelled_image)

    print len(props)
    print props[0].perimeter

    plot_multiple_images([img, line_strength_suppressed, skeleton])
