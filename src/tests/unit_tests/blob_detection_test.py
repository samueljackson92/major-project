import nose.tools
import numpy as np
import pandas as pd
import unittest
import os.path

from mia.features.blobs import blob_features, blob_props
from mia.plotting import plot_blobs
from skimage import io, transform
from ..test_utils import *
from mia import utils

class BlobDetectionTests(unittest.TestCase):

    def test_detect_blob(self):
        pass

    # def test_blob_props(self):
    #     blobs = np.array([[1.0, 1.0, 1.0], [1.0, 3.0, 2.0], [1.0, 3.0, 1.0]])
    #     blobs = pd.DataFrame(blobs, columns=['x', 'y', 'radius'])
    #     expected_result = np.array([ 3., 1.33333333, 0.47140452, 1., 2.])
    #     props = blob_props(blobs)
    #     np.testing.assert_allclose(props, expected_result)
