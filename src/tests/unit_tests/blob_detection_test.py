import nose.tools
import pandas as pd
import unittest

from mia.features.blobs import _blob_density
from ..test_utils import *


class BlobDetectionTests(unittest.TestCase):

    @classmethod
    def setupClass(cls):
        # load a single patient's blobs
        csv_file = get_file_path("2015-03-05-results.csv")
        patient_id = 21401060001
        cls._blobs = pd.DataFrame.from_csv(csv_file)
        cls._blobs = cls._blobs[cls._blobs['patient_id'] == patient_id]
        cls._blobs = cls._blobs[['x', 'y', 'radius']]

    def test_blob_density_measure(self):
        density = _blob_density(self._blobs[['x', 'y']], 4)
        nose.tools.assert_equal(density.shape, (self._blobs.shape[0],))
