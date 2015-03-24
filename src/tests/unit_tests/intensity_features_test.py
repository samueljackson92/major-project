import unittest
import pandas as pd
import nose.tools

from mia.features.intensity import intensity_props
from ..test_utils import get_file_path, load_data_frame


class IntensityTests(unittest.TestCase):

    @classmethod
    def setupClass(cls):
        # load a single patient's blobs
        csv_file = get_file_path("2015-03-05-results.csv")
        patient_id = 21401060001
        cls._results = pd.DataFrame.from_csv(csv_file)
        cls._results = cls._results[cls._results['patient_id'] == patient_id]

    def test_intensity_props(self):
        props = intensity_props(self._results)

        nose.tools.assert_true(isinstance(props, pd.DataFrame))
        nose.tools.assert_equal(props.shape, (1, 8))

        ref_result = load_data_frame("reference_results/"
                                     "2015-03-05-results-intensity.csv")

        pd.util.testing.assert_frame_equal(props, ref_result)
