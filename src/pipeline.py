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

def main():
    arguments = docopt(__doc__, version='0.3.0')
    image_path = arguments["IMAGE"]
    mask_path = arguments["MASK"]
    img, msk = preprocess_image(image_path, mask_path, scale_to_mask=arguments['--scale-to-mask'])
