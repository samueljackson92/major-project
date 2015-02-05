import unittest
import nose.tools
import numpy as np

from skimage import io, transform
import skimage.filter as filters

from mammogram.orientated_bins import orientated_bins

class OrientatedBinsTest(unittest.TestCase):

    def test_with_pure_structure(self):
        linear_structure = generate_linear_structure(20)
        line_strength, line_orinetation = orientated_bins(linear_structure, 7)
        line_strength[line_strength<0.15] = 0

        nose.tools.assert_equal(np.count_nonzero(line_strength), 20)
        nose.tools.assert_equal(np.count_nonzero(line_orinetation), 60)

    def test_with_noisy_structure(self):
        linear_structure = generate_linear_structure(100, with_noise=True)
        line_strength, line_orinetation = orientated_bins(linear_structure, 10)
        line_strength[line_strength<0.05] = 0
        nose.tools.assert_almost_equal(np.count_nonzero(line_strength), 7, delta=2)

    def test_with_multiple_many_bins(self):
        linear_structure = generate_linear_structure(100, with_noise=True)
        line_strength, line_orinetation = orientated_bins(linear_structure, 10)
        line_strength[line_strength<0.05] = 0
        nose.tools.assert_almost_equal(np.count_nonzero(line_strength), 7, delta=2)


def generate_linear_structure(size, with_noise=False):
    """Generate a basic linear structure, possibly with noise"""
    linear_structure = np.zeros(shape=(size,size))
    linear_structure[:,size/2] = np.ones(size)

    if with_noise:
        linear_structure = np.identity(size)
        noise = np.random.rand(size, size) * 0.1
        linear_structure += noise
        linear_structure = filters.gaussian_filter(linear_structure, 1.5)

    return linear_structure
