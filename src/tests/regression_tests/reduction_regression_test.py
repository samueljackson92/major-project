import unittest
import nose.tools
import numpy as np

from mia.reduction import (raw_reduction, blob_features,
                           intensity_features, texture_features,
                           texture_cluster_features)
from ..test_utils import get_file_path, assert_lists_equal


class ReductionRegressionTest(unittest.TestCase):

    @classmethod
    def setupClass(cls):
        cls._img_path = get_file_path("mias/mdb154.png")
        cls._msk_path = get_file_path("mias/masks/mdb154_mask.png")

    def test_process_image_real_mammogram(self):
        blobs_df = blob_features(self._img_path, self._msk_path)

        nose.tools.assert_equal(blobs_df.shape[1], 13)
        blob_columns = ['x', 'y', 'radius', 'count', 'mean',
                        'std', 'min', '25%', '50%', '75%', 'max',
                        'skew', 'kurtosis']

        assert_lists_equal(blobs_df.columns.values, blob_columns)

    def test_raw_reduction(self):
        img_dir = get_file_path("mias/")
        msk_dir = get_file_path("mias/masks/")
        features = raw_reduction(img_dir, msk_dir)

        nose.tools.assert_equal(features.shape, (2, 87283))

    def test_texture_features(self):
        img_dir = get_file_path("mias/mdb154.png")
        msk_dir = get_file_path("mias/masks/mdb154_mask.png")
        features = texture_features(img_dir, msk_dir)

        nose.tools.assert_equal(features.shape, (1, 4))
        np.testing.assert_array_equal(features.index.values, ['mdb154.png'])

    def test_texture_cluster_features(self):
        img_dir = get_file_path("mias/mdb154.png")
        msk_dir = get_file_path("mias/masks/mdb154_mask.png")
        features = texture_cluster_features(img_dir, msk_dir)

        nose.tools.assert_equal(features.shape, (1, 16))
        np.testing.assert_array_equal(features.index.values, ['mdb154.png'])

    def test_intensity_features(self):
        features = intensity_features(self._img_path, self._msk_path)
        nose.tools.assert_equal(features.shape, (1, 10))
