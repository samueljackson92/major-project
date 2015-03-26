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
        mapping = tSNE(df, learning_rate=400, perplexity=45)

        distances = measure_closeness(mapping, self._features['patient_id'])

        nose.tools.assert_equal(mapping.shape, (360, 2))
        nose.tools.assert_equal(distances.shape, (90, ))
        nose.tools.assert_almost_equal(distances.mean(), 5, delta=1.0)

    def test_create_hologic_meta_data(self):
        BIRADS_file = get_file_path("BIRADS.csv")
        md = create_hologic_meta_data(self._features, BIRADS_file)

        expected_columns = ['patient_id', 'side', 'view', 'img_name',
                            'BIRADS', 'img_number'],
        nose.tools.assert_equal(md.shape, (self._features.shape[0], 6))
        nose.tools.assert_true(all([x == y for x, y in
                                    zip(md.index, self._features.index)]))
        for col in expected_columns:
            nose.tools.assert_true(col in md.columns.values)

    def test_create_hologic_meta_data_raw(self):
        path = get_file_path("2015-03-05-results.csv")
        BIRADS_file = get_file_path("BIRADS.csv")

        df = pd.DataFrame.from_csv(path)
        df.index = df.image_name
        md = create_hologic_meta_data(df, BIRADS_file)

        expected_columns = ['patient_id', 'side', 'view', 'img_name',
                            'BIRADS', 'img_number'],
        nose.tools.assert_equal(md.shape, (df.shape[0], 6))
        nose.tools.assert_true(all([x == y for x, y in
                                    zip(md.index, df.index)]))
        for col in expected_columns:
            nose.tools.assert_true(col in md.columns.values)

    def test_create_synthetic_meta_data(self):
        blobs_path = get_file_path("synthetic_blobs.csv")
        df = pd.DataFrame.from_csv(blobs_path)
        df.index = df['img_name']

        md_path = get_file_path("synthetic_meta_data_cleaned.csv")
        md = create_synthetic_meta_data(df, md_path)

        nose.tools.assert_equal(md.shape, (df.shape[0], 16))
        np.testing.assert_equal(md.index.values, df.index.values)

    def test_features_from_blobs(self):
        path = get_file_path("2015-03-05-results.csv")
        df = pd.DataFrame.from_csv(path)
        df.index = df.image_name

        features = features_from_blobs(df)

        nose.tools.assert_equal(features.shape, (360, 12))

    def test_features_from_intensity(self):
        path = get_file_path("2015-03-05-results.csv")
        df = pd.DataFrame.from_csv(path)
        df.index = df.image_name

        features = features_from_intensity(df)

        nose.tools.assert_equal(features.shape, (360, 8))
