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
from skimage import transform, io

from mammogram.orientated_bins import orientated_bins
from mammogram.utils import non_maximal_suppression

if __name__ == '__main__':
    arguments = docopt(__doc__, version='0.1.0')

    image_path = arguments["IMAGE"]
    mask_path = arguments["MASK"]
    img = io.imread(image_path, as_grey=True)

    #scale with gaussian pyramid
    # pyramid = transform.pyramid_gaussian(img, downscale=4)
    # img = pyramid.next()
    # img = pyramid.next()

    #mask image
    if mask_path:
        msk = io.imread(mask_path, as_grey=True)
        img = img * msk

    line_strength, line_orientation = orientated_bins(img, 5, nbins=4)
    # img = non_maximal_suppression(img, kernel=np.ones((5,5)))

    io.imshow(line_strength)
    io.show()
