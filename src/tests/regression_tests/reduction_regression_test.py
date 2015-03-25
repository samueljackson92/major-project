import unittest
import nose.tools

from mia.reduction import process_image, run_raw_reduction
from ..test_utils import get_file_path, assert_lists_equal


class ReductionRegressionTest(unittest.TestCase):

    @classmethod
    def setupClass(cls):
        cls._img_path = get_file_path("mias/mdb154.png")
        cls._msk_path = get_file_path("mias/masks/mdb154_mask.png")

    def test_process_image_real_mammogram(self):
        blobs_df, intensity_df = process_image(self._img_path, self._msk_path)

        nose.tools.assert_equal(blobs_df.shape[1], 4)
        nose.tools.assert_equal(intensity_df.shape[1], 9)

        blob_columns = ['x', 'y', 'radius', 'img_name']
        assert_lists_equal(blobs_df.columns.values, blob_columns)

        intensity_columns = ['count', 'mean', 'std', 'min', '25%', '50%',
                             '75%', 'max', 'img_name']
        assert_lists_equal(intensity_df.columns.values, intensity_columns)

    def test_run_raw_reduction(self):
        img_dir = get_file_path("mias/")
        msk_dir = get_file_path("mias/masks/")
        features = run_raw_reduction(img_dir, msk_dir)

        nose.tools.assert_equal(features.shape, (2, 87283))
