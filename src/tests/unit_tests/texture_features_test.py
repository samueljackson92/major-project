import unittest
import nose.tools

from mia.texture_features import *
from mia.utils import *
from ..test_utils import get_file_path

class TextureFeatureTests(unittest.TestCase):

    @classmethod
    def setupClass(cls):
        img_path = get_file_path("texture_patches/texture1.png")
        cls._img, msk = preprocess_image(img_path, normalise=False)

    def test_gabor_bank_features(self):
        orientations = np.arange(0, np.pi, np.pi/8)
        frequencies = np.arange(0.1,5.0)

        gabor_magnitudes = gabor_features(self._img, frequencies, orientations)
        stats = image_orthotope_statistics(gabor_magnitudes)

        nose.tools.assert_equals(gabor_magnitudes.shape, (5,8,100,100))
        nose.tools.assert_equals(stats.shape, (3,5))

        results_path = get_file_path('reference_results/tex1_gabor_features.npy')
        expected_stats = np.load(results_path)
        np.testing.assert_array_equal(stats, expected_stats)

    def test_glcm_features(self):
        properties = ['contrast', 'dissimilarity']
        orientations = np.arange(0, np.pi, np.pi/8)
        distances = [1,2,4,8]

        features = glcm_features(self._img, distances, orientations, properties)
        nose.tools.assert_equals(features.shape, (2,4,8))

        results_path = get_file_path('reference_results/tex1_glcm_features.npy')
        expected_result = np.load(results_path)
        np.testing.assert_array_equal(features, expected_result)
