"""Mammogram Processing Pipeline.
Usage:
  pipeline.py IMAGE MASK
  pipeline.py (-h | --help)
  pipeline.py --version
Options:
  -h --help     Show this screen.
  --version     Show version.
"""
from docopt import docopt

import skimage.io as io
from skimage.transform import pyramid_gaussian

from mammogram.orientated_bins import orientated_bins

if __name__ == '__main__':
    arguments = docopt(__doc__, version='0.1.0')
    image_path = arguments["IMAGE"]
    mask_path = arguments["MASK"]

    img = io.imread(image_path)
    msk = io.imread(mask_path, as_grey=True)

    pyramid = pyramid_gaussian(img, downscale=4)
    img = pyramid.next()
    img = pyramid.next()

    img = img * msk
    line_strength = orientated_bins(img, 5, nbins=12)

    io.imshow(line_strength)
    io.show()
