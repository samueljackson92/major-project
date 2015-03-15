import unittest
import nose.tools

from scipy.ndimage import filters
import skimage
from skimage import io, transform
from mia.utils import preprocess_image
from ..test_utils import get_file_path
import numpy as np


class UtilsRegressionTest(unittest.TestCase):

    def test_load_real_image_and_mask(self):
        img_path = get_file_path("../../../data/p214-010-60001-cr.png")
        msk_path = get_file_path("../../../data/masks/f214-010-60001-cr_mask.png")
        img, msk = preprocess_image(img_path, msk_path)

        nose.tools.assert_equal(img.shape, (3328, 2560))
        nose.tools.assert_equal(msk.shape, img.shape)

    def test_load_syntehtic_image_and_mask(self):
        img_path = get_file_path("../../../data/test_Mix_DPerc0_c_0.dcm")
        msk_path = get_file_path("../../../data/masks/test_Mix_DPerc0_c_0.png")
        img, msk = preprocess_image(img_path, msk_path)

        nose.tools.assert_equal(img.shape, (1792, 2048))
        nose.tools.assert_equal(msk.shape, img.shape)
