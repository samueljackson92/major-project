import nose.tools
import pandas as pd
import numpy as np
import unittest

from mia.analysis import *


class AnalysisTests(unittest.TestCase):

    def test_normalise_data_frame(self):
        data = [np.arange(0, 1, 0.1), np.arange(0, 5, 0.5), np.arange(-5, 5)]
        df = pd.DataFrame(data)

        df = normalize_data_frame(df)

        for x in df.max(axis=1):
            nose.tools.assert_less(x, 1)
        for x in df.min(axis=1):
            nose.tools.assert_greater(x, -1)

    def test_normalise_data_frame_meta_data(self):
        data = [np.arange(0, 1, 0.1), np.arange(0, 5, 0.5), np.arange(-5, 5)]
        df = pd.DataFrame(data)
        df['meta'] = np.repeat('wordswords', df.shape[0])
        df = normalize_data_frame(df, ['meta'])

        nose.tools.assert_true('meta' in df.columns)
        df.drop('meta', axis=1, inplace=True)

        for x in df.max(axis=1):
            nose.tools.assert_less(x, 1)
        for x in df.min(axis=1):
            nose.tools.assert_greater(x, -1)
