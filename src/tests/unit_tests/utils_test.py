import unittest
import nose.tools
import numpy as np

from mia.utils import normalise_image, preprocess_image
from ..test_utils import get_file_path


class UtilsTest(unittest.TestCase):

    def test_load_image(self):
        img_path = get_file_path("texture_patches/texture1.png")
        img, msk = preprocess_image(img_path)

        nose.tools.assert_equal(img.shape, (100, 100))
        nose.tools.assert_equal(msk, None)

    def test_normalise_image(self):
        img = np.zeros((10, 10))
        img[5, 5] = 255
        img = normalise_image(img)

        nose.tools.assert_equal(np.amax(img), 1)
        nose.tools.assert_equal(np.amin(img), 0)

    def test_normalise_image_specified_range(self):
        img = np.zeros((10, 10))
        img = img-1
        img[5, 5] = 255
        img = normalise_image(img, new_max=100, new_min=-10)

        nose.tools.assert_equal(np.amax(img), 100)
        nose.tools.assert_equal(np.amin(img), -10)
