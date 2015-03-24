import unittest
import nose.tools

from mia.utils import preprocess_image
from ..test_utils import get_file_path


class UtilsRegressionTest(unittest.TestCase):

    def test_load_real_image_and_mask(self):
        img_path = get_file_path("mias/mdb154.png")
        msk_path = get_file_path("mias/masks/mdb154_mask.png")
        img, msk = preprocess_image(img_path, msk_path)

        nose.tools.assert_equal(img.shape, (1024, 980))
        nose.tools.assert_equal(msk.shape, img.shape)
