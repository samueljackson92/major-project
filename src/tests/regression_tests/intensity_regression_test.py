import unittest
import pandas as pd
import nose.tools

from mia.features.blobs import detect_blobs
from mia.features.intensity import detect_intensity
from mia.utils import preprocess_image
from ..test_utils import get_file_path


class IntensityTests(unittest.TestCase):

    @classmethod
    def setupClass(cls):
        img_path = get_file_path("mias/mdb154.png")
        msk_path = get_file_path("mias/masks/mdb154_mask.png")
        cls._img, cls._msk = preprocess_image(img_path, msk_path)

    # def test_detect_intensity(self):
    #     blobs = detect_blobs(self._img, self._msk)
    #     intensity = detect_intensity(self._img, blobs)
    #
    #     nose.tools.assert_true(isinstance(intensity, pd.DataFrame))
    #     nose.tools.assert_equal(intensity.shape[1], 10)
