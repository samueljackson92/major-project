import unittest
import nose.tools
import numpy as np

from mia.utils import *

class UtilsTest(unittest.TestCase):

    def test_normalise_image(self):
        img = np.zeros((10,10))
        img[5,5] = 255
        img = normalise_image(img)

        nose.tools.assert_equal(np.amax(img), 1)
        nose.tools.assert_equal(np.amin(img), 0)

    def test_normalise_image(self):
        img = np.zeros((10,10))
        img = img-1
        img[5,5] = 255
        img = normalise_image(img, new_max=100, new_min=-10)

        nose.tools.assert_equal(np.amax(img), 100)
        nose.tools.assert_equal(np.amin(img), -10)
