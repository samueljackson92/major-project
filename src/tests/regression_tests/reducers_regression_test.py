import unittest
import nose.tools

from mia.reduction.reducers import *
from ..test_utils import *


class ReducersRegressionTest(unittest.TestCase):

    @classmethod
    def setupClass(cls):
        cls._img_dir = get_file_path("mias")
        cls._msk_dir = get_file_path("mias/masks")

    def test_blob_feature_reduction(self):
        reduction = BlobFeaturesReduction(self._img_dir, self._msk_dir)
        blobs_df = reduction.reduce()

        nose.tools.assert_equal(blobs_df.shape[1], 4)
        expected_columns = ['x', 'y', 'radius', 'breast_area']
        assert_data_frame_columns_match(blobs_df, expected_columns)

    def test_line_feature_reduction(self):
        reduction = LineFeaturesReduction(self._img_dir, self._msk_dir)
        lines_df = reduction.reduce()

        nose.tools.assert_equal(lines_df.shape[1], 6)
        expected_columns = ['area', 'min_row', 'min_col', 'max_row',
                            'max_col', 'breast_area']
        assert_data_frame_columns_match(lines_df, expected_columns)

    def test_patch_intensity_feature_reduction(self):
        reduction = BlobFeaturesReduction(self._img_dir, self._msk_dir)
        blobs_df = reduction.reduce()

        reduction = PatchIntensityFeaturesReduction(self._img_dir,
                                                    self._msk_dir)
        intensity_df = reduction.reduce(blobs_df)

        expected_columns = ['x', 'y', 'radius', 'breast_area', 'count',
                            'mean', 'std', 'min', '25%', '50%', '75%',
                            'max', 'skew', 'kurtosis']

        nose.tools.assert_equal(intensity_df.shape[1], 14)
        assert_data_frame_columns_match(intensity_df, expected_columns)

    def test_patch_texture_feature_reduction(self):
        reduction = BlobFeaturesReduction(self._img_dir, self._msk_dir)
        blobs_df = reduction.reduce()

        reduction = PatchTextureFeaturesReduction(self._img_dir,
                                                  self._msk_dir)
        texture_df = reduction.reduce(blobs_df)

        expected_columns = ['x', 'y', 'radius', 'breast_area', 'contrast',
                            'dissimilarity', 'homogeneity', 'energy']

        nose.tools.assert_equal(texture_df.shape[1], 8)
        assert_data_frame_columns_match(texture_df, expected_columns)
