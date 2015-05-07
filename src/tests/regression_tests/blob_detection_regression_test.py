import unittest
import nose.tools

from mia.features.blobs import detect_blobs
from mia.utils import preprocess_image
from ..test_utils import get_file_path


class BlobsRegressionTests(unittest.TestCase):

    def test_detect_blobs(self):
        img_path = get_file_path("mias/mdb154.png")
        msk_path = get_file_path("mias/masks/mdb154_mask.png")
        img, msk = preprocess_image(img_path, msk_path)

        blobs = detect_blobs(img, msk)
        nose.tools.assert_equal(blobs.shape[1], 3)
