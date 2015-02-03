"""Masking.

Usage:
  masking.py IMAGE MASK
  masking.py (-h | --help)
  masking.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.
"""
from docopt import docopt
import scipy
import numpy as np
import math

import skimage.io as io
from skimage import transform

def mask_image(image, mask):
    idx = (mask!=0)
    masked_image = np.zeros(image.shape)
    masked_image[idx] = image[idx]
    return masked_image

if __name__ == '__main__':
    arguments = docopt(__doc__, version='Masking')
    image_path = arguments["IMAGE"]
    mask_path = arguments["MASK"]

    image = io.imread(image_path)

    mask = io.imread(mask_path, as_grey=True)
    mask = transform.rescale(mask, 4) #mammogram is x4 mask

    masked_image = mask_image(image, mask)

    io.imshow(masked_image)
    io.show()
