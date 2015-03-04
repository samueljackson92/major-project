import unittest
import nose.tools
import os.path

from mia.blob_features import blob_features
from mia.utils import preprocess_image


class BlobDetectionRegressionTests(unittest.TestCase):

    def test_blob_features(self):
        img_path = os.path.abspath(os.path.join('data',
                                   'p214-010-60001-cl.png'))
        msk_path = os.path.abspath(os.path.join('data/masks',
                                   'f214-010-60001-cl_mask.png'))
        img, msk = preprocess_image(img_path, msk_path)

        blobs = blob_features(img, msk)
        nose.tools.assert_equal(blobs.shape[1], 3)
