import unittest
import nose.tools
import numpy as np

from skimage import io, transform
from test_utils import generate_blob
from mammogram.blob_detection import blob_detection
from mammogram.plotting import plot_blobs

class BlobDetectionTests(unittest.TestCase):

    def test_detect_blob(self):
        img = generate_blob()
        blob = blob_detection(img)
        plot_blobs(img, blob)
        assert False
