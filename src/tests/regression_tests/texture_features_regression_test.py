import unittest
import nose.tools

from mia.features.linear_structure import detect_linear, extract_line
from mia.features.blobs import detect_blobs
from mia.features.texture import *
from mia.utils import *
from ..test_utils import get_file_path


class TextureRegressionTest(unittest.TestCase):

    @classmethod
    def setupClass(cls):
        img_path = get_file_path("mias/mdb154.png")
        msk_path = get_file_path("mias/masks/mdb154_mask.png")
        cls._img, cls._msk = preprocess_image(img_path, msk_path)

    def test_blob_and_gabor(self):
        blobs = detect_blobs(self._img, self._msk).as_matrix()

        orientations = np.arange(0, np.pi, np.pi/8)
        frequencies = np.arange(0.1, 5.0)

        for blob in blobs:
            image_section = extract_blob(blob, self._img)
            gabor_magnitudes = gabor_features(image_section, frequencies, orientations)
            stats = image_orthotope_statistics(gabor_magnitudes)

            h, w = image_section.shape
            nose.tools.assert_equals(gabor_magnitudes.shape, (5, 8, h, w))
            nose.tools.assert_equals(stats.shape, (3, 5))

    def test_linear_and_gabor(self):
        nbins, size, threshold = 12, 5, 1.0e-2
        regions, line_strength = detect_linear(self._img, self._msk,
                                               size, nbins, threshold)

        orientations = np.arange(0, np.pi, np.pi/8)
        frequencies = np.arange(0.1, 5.0)

        for i, props in regions.iterrows():
            image_section = extract_line(props, self._img)
            gabor_magnitudes = gabor_features(image_section, frequencies, orientations)
            stats = image_orthotope_statistics(gabor_magnitudes)

            h, w = image_section.shape
            nose.tools.assert_equals(gabor_magnitudes.shape, (5, 8, h, w))
            nose.tools.assert_equals(stats.shape, (3, 5))

    def test_blob_and_glcm(self):
        properties = ['contrast', 'dissimilarity']
        orientations = np.arange(0, np.pi, np.pi/8)
        distances = [1,2,4,8]

        blobs = detect_blobs(self._img, self._msk).as_matrix()

        for blob in blobs:
            image_section = extract_blob(blob, self._img)
            features = glcm_features(image_section, distances, orientations, properties)
            nose.tools.assert_equals(features.shape, (2,4,8))

    def test_linear_and_glcm(self):
        nbins, size, threshold = 12, 5, 1.0e-2
        regions, line_strength = detect_linear(self._img, self._msk,
                                               size, nbins, threshold)

        properties = ['contrast', 'dissimilarity']
        orientations = np.arange(0, np.pi, np.pi/8)
        distances = [1,2,4,8]

        for i, props in regions.iterrows():
            image_section = extract_line(props, self._img)
            features = glcm_features(image_section, distances, orientations, properties)
            nose.tools.assert_equals(features.shape, (2,4,8))

    def test_compute_glcm_from_blob(self):
        properties = ['contrast', 'dissimilarity', 'homogeneity',
                      'energy', 'correlation']
        orientations = np.arange(0, np.pi, np.pi/8)
        distances = [1, 3, 5]

        blobs = detect_blobs(self._img, self._msk).as_matrix()
        tex_props = blob_texture_props(self._img, blobs, properties,
                                       distances, orientations)
        nose.tools.assert_equal(tex_props.shape, (20,))

    def test_texture_from_clusters(self):
        img_path = get_file_path("mias/mdb154.png")
        msk_path = get_file_path("mias/masks/mdb154_mask.png")
        img, msk = preprocess_image(img_path, msk_path)

        labels = cluster_image(img)
        clusters = clusters_from_labels(img, labels)
        srtd_clusters = sort_clusters_by_density(clusters)[1:]
        tex_features = texture_from_clusters(srtd_clusters)

        props = ['contrast', 'dissimilarity', 'homogeneity', 'energy']
        expected_cols = [prop + '_cluster_%d' % (i+1)
                         for i in range(clusters[1:].shape[0]) for prop in props]
        nose.tools.assert_equal(tex_features.shape, (1, 16))
        np.testing.assert_array_equal(tex_features.columns.values, expected_cols)
