import unittest
import nose.tools


class ConvolveToolsTests(unittest.TestCase):

    def test_import(self):
        from convolve_tools import deformable_covolution
        nose.tools.assert_true(deformable_covolution is not None)
