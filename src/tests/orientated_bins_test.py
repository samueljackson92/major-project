import unittest
import nose.tools
import numpy as np

import skimage.io as io
from scipy.ndimage.filters import gaussian_filter

from mammogram.orientated_bins import orientated_bins

class OrientatedBinsTest(unittest.TestCase):

    def test_with_pure_structure(self):
        size = 20
        linear_structure = np.zeros(shape=(size,size))
        linear_structure[:,size/2] = np.ones(size)

        line_strength = orientated_bins(linear_structure, 7)
        # line_strength[line_strength<0.15] = 0 #threshold
        io.imshow(line_strength)
        io.show()
        nose.tools.assert_equal(np.count_nonzero(line_strength), size)

    def test_with_burred_structure(self):
        linear_structure = np.identity(20)
        img = gaussian_filter(linear_structure, 0.8)

        line_strength = orientated_bins(img, 10)
        line_strength[line_strength<0.15] = 0 #threshold

        nose.tools.assert_equal(np.count_nonzero(line_strength), 20)

    def test_with_multiple_windows(self):
        size = 100
        linear_structure = np.zeros(shape=(size,size))
        linear_structure[:,15] = np.ones(size)

        linear_structure = np.identity(size)
        noise = np.random.rand(size, size) * 0.5
        linear_structure += noise
        img = gaussian_filter(linear_structure, 1.5)

        line_strength = orientated_bins(img, 10)

    def test_with_more_bins(self):
        size = 100
        linear_structure = np.zeros(shape=(size,size))
        linear_structure[:,15] = np.ones(size)

        linear_structure = np.identity(size)
        noise = np.random.rand(size, size) * 0.5
        linear_structure += noise
        img = gaussian_filter(linear_structure, 1.5)

        line_strength = orientated_bins(img, 5, nbins=8)

    # def test_real(self):
    #     path="../../data/p214-010-60001-cl.png"
    #     mask="../../data/f214-010-60001-cl_mask.png"
    #
    #     img = io.imread(path)
    #     msk = io.imread(mask, as_grey=True)
    #
    #     pyramid = pyramid_gaussian(img, downscale=4)
    #     pyramid.next()
    #     img = pyramid.next()
    #
    #     img = img * msk
    #
    #     line_strength = orientated_bins(img, 5, nbins=12)
    #
    #     # io.imshow(line_strength)
    #     # io.show()
