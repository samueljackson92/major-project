import unittest
import nose.tools
import os.path

from mia.features.blobs import detect_blobs
from mia.utils import preprocess_image


class BlobsRegressionTests(unittest.TestCase):

    def test_detect_blobs(self):
        img_path = os.path.abspath(os.path.join('data',
                                   'p214-010-60001-cl.png'))
        msk_path = os.path.abspath(os.path.join('data/masks',
                                   'f214-010-60001-cl_mask.png'))
        img, msk = preprocess_image(img_path, msk_path)

        blobs = detect_blobs(img, msk)
        nose.tools.assert_equal(blobs.shape[1], 3)
