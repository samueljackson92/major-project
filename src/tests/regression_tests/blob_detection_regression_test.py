import unittest
import nose.tools
import os.path

from mammogram.blob_detection import blob_detection
from mammogram.utils import *
from ..test_utils import get_file_path

class BlobDetectionRegressionTests(unittest.TestCase):

    def test_blob_detection(self):
        img_path = os.path.abspath(os.path.join('data', 'p214-010-60001-cl.png'))
        msk_path = os.path.abspath(os.path.join('data', 'f214-010-60001-cl_mask.png'))
        img, msk = preprocess_image(img_path, msk_path)

        blobs = blob_detection(img, msk)
        nose.tools.assert_equal(blobs.shape[1], 3)
