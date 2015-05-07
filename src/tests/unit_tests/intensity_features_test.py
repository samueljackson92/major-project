import unittest
import numpy as np
import pandas as pd
import nose.tools

from mia.features.intensity import intensity_props, detect_intensity
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

    def test_detect_intensity_with_blobs(self):
        img_path = get_file_path("mias/mdb154.png")
        img, msk = preprocess_image(img_path)

        path = get_file_path("reference_results/mias-blobs.csv")
        df = pd.DataFrame.from_csv(path)

        intensity = detect_intensity(img, df.loc["mdb154.png"])

        nose.tools.assert_true(isinstance(intensity, pd.DataFrame))
        nose.tools.assert_equal(intensity.shape, (25, 14))

        path_name = "reference_results/mdb154_intensity_blobs.npy"
        result_path = get_file_path(path_name)
        expected_result = np.load(result_path)
        np.testing.assert_almost_equal(intensity.as_matrix(), expected_result)

    def test_detect_intensity_with_lines(self):
        img_path = get_file_path("mias/mdb154.png")
        img, msk = preprocess_image(img_path)

        path = get_file_path("reference_results/mias-lines.csv")
        df = pd.DataFrame.from_csv(path)

        intensity = detect_intensity(img, df.loc["mdb154.png"])

        nose.tools.assert_true(isinstance(intensity, pd.DataFrame))
        nose.tools.assert_equal(intensity.shape, (2, 16))

        path_name = "reference_results/mdb154_intensity_lines.npy"
        result_path = get_file_path(path_name)
        expected_result = np.load(result_path)
        np.testing.assert_almost_equal(intensity.as_matrix(), expected_result)
