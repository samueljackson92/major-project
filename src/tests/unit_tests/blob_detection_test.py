import nose.tools
import numpy as np
import unittest
import os.path

from mammogram.blob_detection import blob_detection, blob_props
from mammogram.plotting import plot_blobs
from skimage import io, transform
from ..test_utils import *
from mammogram import utils

class BlobDetectionTests(unittest.TestCase):

    def test_detect_blob(self):
        pass

    def test_blob_props(self):
        blobs1 = np.array([[1.0, 1.0, 1.0], [1.0, 3.0, 2.0], [1.0, 3.0, 1.0]])
        blobs2 = np.array([[3.0, 2.0, 0.1], [4.0, 0.5, 0.2], [1.0, 2.0, 1.0], [1.0, 2.0, 1.0]])

        expected_result = np.array([[ 3., 1.33333333, 0.47140452, 1., 2.],
                                    [ 4., 0.575, 0.42646805, 0.1, 1.]])

        blobs = [blobs1[:,2], blobs2[:,2]]

        feature_matrix = blob_props(blobs)
        np.testing.assert_allclose(feature_matrix.as_matrix(), expected_result)
