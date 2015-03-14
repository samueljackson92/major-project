import unittest
import numpy as np

from mia.features.intensity import _intensity_stats
from mia.utils import preprocess_image
from ..test_utils import get_file_path


class IntensityTests(unittest.TestCase):

    @classmethod
    def setupClass(cls):
        img_path = get_file_path("texture_patches/texture1.png")
        cls._img, msk = preprocess_image(img_path)

    def test_intensity_stats(self):
        expected_output = np.array([39.0625, 146.827585, 4.559662, 21.860318])
        stats = _intensity_stats(self._img)
        np.testing.assert_allclose(stats, expected_output)
