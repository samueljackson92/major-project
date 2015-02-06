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
from mammogram.plotting import plot_multiple_images
from mammogram.utils import non_maximal_suppression

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

    nbins = 12
    size = 5
    line_strength, line_orientation = orientated_bins(img, size, nbins=nbins)
    parallel_orientation = (line_orientation + (nbins/2)) % nbins
    np.max(parallel_orientation)

    from scipy.ndimage import filters

    horizontal = np.zeros(shape=(3,3), dtype='int64')
    horizontal[1] = np.ones(3)

    vertical = np.zeros(shape=(3,3), dtype='int64')
    vertical[:,1] = np.ones(3, dtype='int64')

    left_diagonal = np.eye(3, dtype='int64')
    right_diagonal = np.fliplr(left_diagonal)

    kernels = np.array([
        np.where(horizontal),
        np.where(left_diagonal),
        np.where(vertical),
        np.where(right_diagonal)
    ])

    filterd_images = []
    for kernel in kernels:
        filterd_image = np.zeros(line_strength.shape)
        filters.maximum_filter(line_strength, footprint=kernel, output=filterd_image)
        filterd_images.append(filterd_image)

    parallel_orientation = parallel_orientation % len(kernels)
    line_strength_suppressed = np.zeros(line_strength.shape)
    np.choose(parallel_orientation, filterd_images, out=line_strength_suppressed)

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
