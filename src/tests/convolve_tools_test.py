import unittest
import nose.tools
import numpy as np

class ConvolveToolsTests(unittest.TestCase):

    def test_import(self):
        from convolve_tools import deformable_covolution
        nose.tools.assert_true(deformable_covolution != None)

    def test_run(self):
        from convolve_tools import deformable_covolution
        print deformable_covolution(2)
