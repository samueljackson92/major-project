import unittest
import test_utils
import nose.tools

from mammogram.texture_features import *
from mammogram.utils import *
from mammogram.plotting import plot_image_cube
from test_utils import get_file_path

from skimage import io, feature

class TextureFeatureTests(unittest.TestCase):

    @classmethod
    def setupClass(cls):
        img_path = get_file_path("texture_patches/texture1.png")
        cls._img = preprocess_image(img_path, normalise=False)

    def test_gabor_bank_features(self):
        orientations = np.arange(0, np.pi, np.pi/8)
        frequencies = np.arange(0.1,5.0)

        gabor_magnitudes = gabor_features(self._img, frequencies, orientations)
        stats = image_orthotope_statistics(gabor_magnitudes)

        nose.tools.assert_equals(gabor_magnitudes.shape, (5,8,100,100))
        nose.tools.assert_equals(stats.shape, (3,5))


    def test_glcm_features(self):
        properties = ['contrast', 'dissimilarity']
        orientations = np.arange(0, np.pi, np.pi/8)
        distances = [1,2,4,8]

        features = glcm_features(self._img, distances, orientations, properties)
        nose.tools.assert_equals(features.shape, (2,4,8))
