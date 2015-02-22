import unittest
import nose.tools
import os.path

from mammogram.linear_features import *
from mammogram.blob_detection import *
from mammogram.texture_features import *
from mammogram.utils import *

class TextureFeaturesRegressionTest(unittest.TestCase):

    @classmethod
    def setupClass(cls):
		img_path = os.path.abspath(os.path.join('data', 'p214-010-60001-cl.png'))
		msk_path = os.path.abspath(os.path.join('data', 'f214-010-60001-cl_mask.png'))
		cls._img, cls._msk = preprocess_image(img_path, msk_path)

    def test_blob_and_gabor(self):
        blobs = blob_detection(self._img, self._msk)

        orientations = np.arange(0, np.pi, np.pi/8)
        frequencies = np.arange(0.1,5.0)

        for blob in blobs:
            image_section = extract_blob(blob, self._img)
            gabor_magnitudes = gabor_features(image_section, frequencies, orientations)
            stats = image_orthotope_statistics(gabor_magnitudes)

            h,w = image_section.shape
            nose.tools.assert_equals(gabor_magnitudes.shape, (5,8, h, w))
            nose.tools.assert_equals(stats.shape, (3,5))

    def test_linear_and_gabor(self):
        nbins, size, threshold = 12, 5, 4.0e-2
        line_strength, regions = linear_features(self._img, size, nbins, threshold)

        orientations = np.arange(0, np.pi, np.pi/8)
        frequencies = np.arange(0.1,5.0)

        for props in regions:
            image_section = extract_feature(props, self._img)
            gabor_magnitudes = gabor_features(image_section, frequencies, orientations)
            stats = image_orthotope_statistics(gabor_magnitudes)

            h,w = image_section.shape
            nose.tools.assert_equals(gabor_magnitudes.shape, (5,8, h, w))
            nose.tools.assert_equals(stats.shape, (3,5))

    def test_blob_and_glcm(self):
        properties = ['contrast', 'dissimilarity']
        orientations = np.arange(0, np.pi, np.pi/8)
        distances = [1,2,4,8]

        blobs = blob_detection(self._img, self._msk)

        for blob in blobs:
            image_section = extract_blob(blob, self._img)
            features = glcm_features(image_section, distances, orientations, properties)
            nose.tools.assert_equals(features.shape, (2,4,8))

    def test_linear_and_glcm(self):
        nbins, size, threshold = 12, 5, 4.0e-2
        line_strength, regions = linear_features(self._img, size, nbins, threshold)

        properties = ['contrast', 'dissimilarity']
        orientations = np.arange(0, np.pi, np.pi/8)
        distances = [1,2,4,8]

        for props in regions:
            image_section = extract_feature(props, self._img)
            features = glcm_features(image_section, distances, orientations, properties)
            nose.tools.assert_equals(features.shape, (2,4,8))
