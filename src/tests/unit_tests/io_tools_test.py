import nose.tools
import unittest
import os
import json
import pandas as pd
import numpy as np

import mia
from mia.io_tools import *
from ..test_utils import get_file_path


class IOTests(unittest.TestCase):

    @classmethod
    def setupClass(cls):
        cls._output_files = []

    @classmethod
    def teardownClass(cls):
        for f in cls._output_files:
            if os.path.isfile(f):
                os.remove(f)

    def test_iterate_directory(self):
        img_directory = get_file_path("texture_patches")
        expected_files = ['texture1.png', 'texture2.png', 'texture3.png',
                          'texture4.png', 'texture5.png']

        expected_files = [os.path.join(img_directory, p) for p in expected_files]

        dirs = list(iterate_directory(img_directory))
        nose.tools.assert_equal(len(dirs), len(expected_files))

        for img_path, expected in zip(dirs, expected_files):
            nose.tools.assert_equal(img_path, expected)

    def test_iterate_directories(self):
        img_directory = get_file_path("texture_patches")
        expected_files = ['texture1.png', 'texture2.png', 'texture3.png',
                          'texture4.png', 'texture5.png']

        expected_files = [os.path.join(img_directory, p) for p in expected_files]

        dirs = list(iterate_directories(img_directory, img_directory))
        nose.tools.assert_equal(len(dirs), len(expected_files))

        for (img_path, msk_path), expected in zip(dirs, expected_files):
            nose.tools.assert_equal(img_path, expected)
            nose.tools.assert_equal(msk_path, expected)

    def test_check_is_file(self):
        img_path = get_file_path("texture_patches/texture1.png")
        nose.tools.assert_true(check_is_file(img_path, ".png"))

    def test_check_is_file_multiple_images(self):
        img_path = get_file_path("synthetic_patch.dcm")
        nose.tools.assert_true(check_is_file(img_path, ".png", ".dcm"))

    def test_check_is_file_wrong_extension(self):
        img_path = get_file_path("blob_detection.csv")
        nose.tools.assert_false(check_is_file(img_path, ".png", ".dcm"))

    def test_check_is_image_raises_on_not_a_file(self):
        img_path = get_file_path("texture_patches")
        nose.tools.assert_false(check_is_file(img_path, ".png", ".dcm"))

    def test_check_is_directory(self):
        directory = get_file_path("texture_patches")
        try:
            check_is_directory(directory)
        except:
            self.fail("check_is_directory raised when it shouldn't have.")

    def test_check_is_directory_raises(self):
        img_path = get_file_path("texture_patches/not_a_directory")
        nose.tools.assert_raises(ValueError, check_is_directory, img_path)

    def test_dump_mapping_to_json(self):
        output_file = 'test_data.json'
        mapping = pd.DataFrame(np.ones((10, 2)), columns=['x', 'y'])
        dump_mapping_to_json(mapping, ['x', 'y'], np.zeros(10), output_file)

        nose.tools.assert_true(os.path.isfile(output_file))

        with open(output_file, 'rb') as f:
            data = json.load(f)

        nose.tools.assert_equal(len(data), 1)
        nose.tools.assert_equal(data[0]['name'], 'Class: 0')
        nose.tools.assert_equal(len(data[0]['data']), 10)

        self._output_files.append(output_file)
