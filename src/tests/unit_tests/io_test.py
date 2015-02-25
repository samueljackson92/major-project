import nose.tools
import unittest
import os.path

from mammogram.io import *
from ..test_utils import *

class IOTests(unittest.TestCase):

    def test_iterate_dir(self):
        directory = get_file_path("texture_patches")
        expected_files = ['texture1.png', 'texture2.png', 'texture3.png',
                          'texture4.png', 'texture5.png']

        expected_files = [os.path.join(directory, p) for p in expected_files]

        for img_path, expected in zip(iterate_directory(directory), expected_files):
            nose.tools.assert_equal(img_path, expected)

    def test_check_is_image(self):
        img_path = get_file_path("texture_patches/texture1.png")
        try:
            check_is_image(img_path, ".png")
        except:
            self.fail("check_is_image raised when it shouldn't have.")

    def test_check_is_image_raises(self):
        img_path = get_file_path("texture_patches/texture1.png")
        nose.tools.assert_raises(ValueError, check_is_image, img_path, ".jpg")

    def test_check_is_directory(self):
        directory = get_file_path("texture_patches")
        try:
            check_is_directory(directory)
        except:
            self.fail("check_is_directory raised when it shouldn't have.")

    def test_check_is_directory_raises(self):
        img_path = get_file_path("texture_patches/not_a_directory")
        nose.tools.assert_raises(ValueError, check_is_directory, img_path)
