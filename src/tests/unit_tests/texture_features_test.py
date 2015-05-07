import unittest
import nose.tools

from mia.features.texture import *
from mia.utils import *
from ..test_utils import get_file_path


class TextureTests(unittest.TestCase):

    @classmethod
    def setupClass(cls):
        img_path = get_file_path("texture_patches/texture1.png")
        cls._img, msk = preprocess_image(img_path)

    def test_detect_intensity_with_blobs(self):
        img_path = get_file_path("mias/mdb154.png")
        img, msk = preprocess_image(img_path)

        path = get_file_path("reference_results/mias-blobs.csv")
        df = pd.DataFrame.from_csv(path)

        texture = detect_texture(img, df.loc["mdb154.png"])

        nose.tools.assert_true(isinstance(texture, pd.DataFrame))
        nose.tools.assert_equal(texture.shape, (25, 8))

        path_name = "reference_results/mdb154_texture_blobs.npy"
        result_path = get_file_path(path_name)
        expected_result = np.load(result_path)
        np.testing.assert_almost_equal(texture.as_matrix(), expected_result)

    def test_detect_intensity_with_lines(self):
        img_path = get_file_path("mias/mdb154.png")
        img, msk = preprocess_image(img_path)

        path = get_file_path("reference_results/mias-lines.csv")
        df = pd.DataFrame.from_csv(path)

        texture = detect_texture(img, df.loc["mdb154.png"])

        nose.tools.assert_true(isinstance(texture, pd.DataFrame))
        nose.tools.assert_equal(texture.shape, (2, 10))

        path_name = "reference_results/mdb154_texture_lines.npy"
        result_path = get_file_path(path_name)
        expected_result = np.load(result_path)
        np.testing.assert_almost_equal(texture.as_matrix(), expected_result)

    def test_gabor_bank_features(self):
        orientations = np.arange(0, np.pi, np.pi/8)
        frequencies = np.arange(0.1, 5.0)

        gabor_magnitudes = gabor_features(self._img, frequencies, orientations)
        stats = image_orthotope_statistics(gabor_magnitudes)

        nose.tools.assert_equals(gabor_magnitudes.shape, (5, 8, 100, 100))
        nose.tools.assert_equals(stats.shape, (3, 5))

        results_path = get_file_path('reference_results/tex1_gabor_features.npy')
        expected_stats = np.load(results_path)
        np.testing.assert_allclose(stats, expected_stats)

    def test_glcm_features(self):
        properties = ['contrast', 'dissimilarity']
        orientations = np.arange(0, np.pi, np.pi/8)
        distances = [1, 2, 4, 8]

        features = glcm_features(self._img, distances, orientations, properties)
        nose.tools.assert_equals(features.shape, (2, 4, 8))

        results_path = get_file_path('reference_results/tex1_glcm_features.npy')
        expected_result = np.load(results_path)
        np.testing.assert_array_equal(features, expected_result)

    def test_filter_image_for_texture(self):
        props = filter_image_for_texture(self._img, 0, 'dissimilarity')
        results_path = get_file_path('reference_results/tex1_filter_for_texture.npy')
        expected_result = np.load(results_path)
        np.testing.assert_array_equal(props, expected_result)
