import unittest
import nose.tools
import numpy.testing

from mia.features.intensity import _intensity_stats
from mia.utils import *
from ..test_utils import get_file_path

class IntensityTests(unittest.TestCase):

    @classmethod
    def setupClass(cls):
        img_path = get_file_path("texture_patches/texture1.png")
        cls._img, msk = preprocess_image(img_path, normalise=False)

    def test_intensity_stats(self):
        expected_output = np.array([222.22222222, 286.25310392,
                                    1.33091438, 0.75751295])
        stats = _intensity_stats(self._img)
        np.testing.assert_allclose(stats, expected_output)
