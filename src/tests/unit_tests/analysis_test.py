import nose.tools
import pandas as pd
import numpy as np
import unittest

from mia.analysis import *
from ..test_utils import get_file_path


class AnalysisTests(unittest.TestCase):

    @classmethod
    def setupClass(cls):
        csv_file = get_file_path("blob_detection.csv")
        cls._features = pd.DataFrame.from_csv(csv_file)

    def test_standard_scaler(self):
        data = [np.arange(0, 1, 0.1), np.arange(0, 5, 0.5), np.arange(-5, 5)]
        df = pd.DataFrame(data)

        df = standard_scaler(df)

        for x in df.max(axis=1):
            nose.tools.assert_less(x, 1.5)
        for x in df.min(axis=1):
            nose.tools.assert_greater(x, -1.5)

    def test_normalise_data_frame(self):
        data = [np.arange(0, 1, 0.1), np.arange(0, 5, 0.5), np.arange(-5, 5)]
        df = pd.DataFrame(data)

        df = normalize_data_frame(df)

        for x in df.max(axis=1):
            nose.tools.assert_less(x, 1)
        for x in df.min(axis=1):
            nose.tools.assert_greater(x, -1)

    def test_tsne_with_data_frame(self):
        feature_names = ['avg_radius', 'std_radius', 'blob_count']
        df = self._features[feature_names]
        mapping = tSNE(df)

        nose.tools.assert_equal(mapping.shape, (360, 2))

    def test_tsne_with_matrix(self):
        feature_names = ['avg_radius', 'std_radius', 'blob_count']
        df = self._features[feature_names]
        matrix = df.as_matrix()
        mapping = tSNE(matrix)

        nose.tools.assert_equal(mapping.shape, (360, 2))

    def test_measure_closeness(self):
        feature_names = ['avg_radius', 'std_radius', 'blob_count']
        df = self._features[feature_names]
        mapping = tSNE(df)

        distances = measure_closeness(mapping, self._features['patient_id'])

        nose.tools.assert_equal(mapping.shape, (360, 2))
        nose.tools.assert_equal(distances.shape, (90, ))
        nose.tools.assert_almost_equal(distances.mean(), 5, delta=1.0)
