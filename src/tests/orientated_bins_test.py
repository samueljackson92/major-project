import unittest
import nose.tools
import numpy as np

from skimage import io, transform
from test_utils import generate_linear_structure
from mammogram.orientated_bins import orientated_bins

class OrientatedBinsTest(unittest.TestCase):

    def test_with_pure_structure(self):
        linear_structure = generate_linear_structure(20)
        line_strength, line_orinetation = orientated_bins(linear_structure, 7)
        line_strength[line_strength<0.3] = 0

        nose.tools.assert_equal(np.count_nonzero(line_strength), 20)
        nose.tools.assert_equal(np.count_nonzero(line_orinetation), 100)

    def test_with_noisy_structure(self):
        linear_structure = generate_linear_structure(100, with_noise=True)
        line_strength, line_orinetation = orientated_bins(linear_structure, 10)
        line_strength[line_strength<0.05] = 0
        nose.tools.assert_almost_equal(np.count_nonzero(line_strength), 312, delta=10)

    def test_with_multiple_many_bins(self):
        linear_structure = generate_linear_structure(100, with_noise=True)
        line_strength, line_orinetation = orientated_bins(linear_structure, 10)
        line_strength[line_strength<0.05] = 0
        nose.tools.assert_almost_equal(np.count_nonzero(line_strength), 312, delta=10)
