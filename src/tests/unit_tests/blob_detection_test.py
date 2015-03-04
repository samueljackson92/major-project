import nose.tools
import numpy as np
import unittest
import os.path

from mammogram.blob_features import blob_features, blob_props
from mammogram.plotting import plot_blobs
from skimage import io, transform
from ..test_utils import *
from mammogram import utils

class BlobDetectionTests(unittest.TestCase):

    def test_detect_blob(self):
        pass

    def test_blob_props(self):
        blobs = np.array([[1.0, 1.0, 1.0], [1.0, 3.0, 2.0], [1.0, 3.0, 1.0]])
        expected_result = np.array([ 3., 1.33333333, 0.47140452, 1., 2.])
        props = blob_props(blobs)
        np.testing.assert_allclose(props, expected_result)
