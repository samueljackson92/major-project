import unittest
import nose.tools
from sklearn import manifold, datasets

from mia.coranking import trustworthiness, continuity
from mia.utils import *


class CorankingTest(unittest.TestCase):

    def setUp(self):
        self._high_data, color \
            = datasets.samples_generator.make_swiss_roll(n_samples=1500,
                                                         random_state=1)

        isomap = manifold.Isomap(n_neighbors=12, n_components=2)
        self._low_data = isomap.fit_transform(self._high_data)

    def test_trustworthiness(self):
        t = trustworthiness(self._high_data, self._low_data, 5)
        nose.tools.assert_true(isinstance(t, float))
        nose.tools.assert_almost_equal(t, 0.99, places=1)

    def test_continuity(self):
        c = continuity(self._high_data, self._low_data, 5)
        nose.tools.assert_true(isinstance(c, float))
        nose.tools.assert_almost_equal(c, 0.99, places=1)

        c2 = trustworthiness(self._low_data, self._high_data, 5)
        nose.tools.assert_true(c, c2)
