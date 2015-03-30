import unittest
import numpy as np
import pandas as pd
import nose.tools

from mia.features.intensity import intensity_props
from mia.utils import preprocess_image
from ..test_utils import get_file_path


class IntensityTests(unittest.TestCase):

    @classmethod
    def setupClass(cls):
        img_path = get_file_path("texture_patches/texture1.png")
        cls._img, msk = preprocess_image(img_path)

    def test_intensity_props(self):
        props = intensity_props(self._img)

        expected_result = [[1.00000000e+04, 7.95705882e-02, 2.25583335e-02,
                            3.52941176e-02, 6.27450980e-02, 7.45098039e-02,
                            9.41176471e-02, 2.07843137e-01, 8.10502678e-01,
                            7.94866829e-01]]
        nose.tools.assert_true(isinstance(props, pd.DataFrame))
        nose.tools.assert_equal(props.shape, (1, 10))
        np.testing.assert_almost_equal(props.as_matrix(), expected_result)
