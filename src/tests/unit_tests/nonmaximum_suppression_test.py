import unittest
import nose.tools
import numpy as np

from skimage import io, transform
from ..test_utils import generate_linear_structure
from mia.orientated_bins import orientated_bins
from mia.nonmaximum_suppression import nonmaximum_suppression

class NonMaximumSuppressionTest(unittest.TestCase):

    def test_suppression(self):
        linear_structure = generate_linear_structure(20)
        nbins = 8
        line_strength, line_orinetation = orientated_bins(linear_structure, nbins)
        suppressed = nonmaximum_suppression(line_strength, line_orinetation, nbins)

        nose.tools.assert_equal(np.count_nonzero(suppressed), 200)

    def test_suppression_fails_incorrect_bins(self):
        linear_structure = generate_linear_structure(20)
        nbins = 6
        line_strength, line_orinetation = orientated_bins(linear_structure, 7,
                                                            nbins=nbins)

        args = (line_strength, line_orinetation, nbins)
        nose.tools.assert_raises(ValueError, nonmaximum_suppression, *args)

    def test_different_kernel_size(self):
        linear_structure = generate_linear_structure(20)
        nbins = 8
        line_strength, line_orinetation = orientated_bins(linear_structure, nbins)
        suppressed = nonmaximum_suppression(line_strength, line_orinetation,
                                            nbins, kernel_size=6)

        nose.tools.assert_equal(np.count_nonzero(suppressed), 260)
