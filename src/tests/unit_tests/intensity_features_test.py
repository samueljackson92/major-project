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

        nose.tools.assert_true(isinstance(props, pd.DataFrame))
        nose.tools.assert_equal(props.shape, (1, 10))

        path_name = "reference_results/tex1_intensity_features.npy"
        result_path = get_file_path(path_name)
        expected_result = np.load(result_path)

        np.testing.assert_almost_equal(props.as_matrix(), expected_result)
