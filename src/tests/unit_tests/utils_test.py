import unittest
import nose.tools
import numpy as np

from skimage import morphology
from mia.utils import *
from ..test_utils import get_file_path


class UtilsTest(unittest.TestCase):

    def test_load_real_mammogram(self):
        img_path = get_file_path("texture_patches/texture1.png")
        img = load_real_mammogram(img_path)

        nose.tools.assert_equal(img.dtype, 'float64')
        nose.tools.assert_equal(img.shape, (100, 100))
        nose.tools.assert_less_equal(img.max(), 1.0)
        nose.tools.assert_greater_equal(img.min(), 0.0)

    def test_load_synthetic_mammogram(self):
        img_path = get_file_path("synthetic_patch.dcm")
        img = load_synthetic_mammogram(img_path)

        nose.tools.assert_equal(img.dtype, 'float64')
        nose.tools.assert_equal(img.shape, (100, 100))
        nose.tools.assert_less_equal(img.max(), 1.0)
        nose.tools.assert_greater_equal(img.min(), 0.0)

    def test_load_synthetic_mask(self):
        img_path = get_file_path("synthetic_mask_patch.png")
        img = load_mask(img_path)

        nose.tools.assert_equal(img.dtype, 'float64')
        nose.tools.assert_equal(img.shape, (100, 100))
        nose.tools.assert_less_equal(img.max(), 1.0)
        nose.tools.assert_greater_equal(img.min(), 0.0)

    def test_load_image_real(self):
        img_path = get_file_path("texture_patches/texture1.png")
        img, msk = preprocess_image(img_path)

        nose.tools.assert_equal(img.shape, (100, 100))
        nose.tools.assert_equal(msk, None)
        nose.tools.assert_less_equal(img.max(), 1.0)
        nose.tools.assert_greater_equal(img.min(), 0.0)

    def test_load_image_synthetic(self):
        img_path = get_file_path("synthetic_patch.dcm")
        img, msk = preprocess_image(img_path)

        nose.tools.assert_equal(img.shape, (100, 100))
        nose.tools.assert_equal(msk, None)
        nose.tools.assert_less_equal(img.max(), 1.0)
        nose.tools.assert_greater_equal(img.min(), 0.0)

    def test_load_image_and_mask_real(self):
        img_path = get_file_path("texture_patches/texture1.png")
        msk_path = get_file_path("mask_patch.png")
        img, msk = preprocess_image(img_path, msk_path)

        nose.tools.assert_equal(img.shape, (100, 100))
        nose.tools.assert_less_equal(img.max(), 1.0)
        nose.tools.assert_greater_equal(img.min(), 0.0)

        nose.tools.assert_equal(msk.shape, (100, 100))
        nose.tools.assert_less_equal(msk.max(), 1.0)
        nose.tools.assert_greater_equal(msk.min(), 0.0)

    def test_load_image_and_mask_synthetic(self):
        img_path = get_file_path("synthetic_patch.dcm")
        msk_path = get_file_path("synthetic_mask_patch.png")
        img, msk = preprocess_image(img_path, msk_path)

        nose.tools.assert_equal(img.shape, (100, 100))
        nose.tools.assert_less_equal(img.max(), 1.0)
        nose.tools.assert_greater_equal(img.min(), 0.0)

        nose.tools.assert_equal(msk.shape, (100, 100))
        nose.tools.assert_less_equal(msk.max(), 1.0)
        nose.tools.assert_greater_equal(msk.min(), 0.0)

    def test_resize_mask_to_image(self):
        msk = np.ones((50, 50))
        new_msk = resize_mask_to_image(msk, (100, 100))
        nose.tools.assert_equal(new_msk.shape, (100, 100))

    def test_normalise_image(self):
        img = np.zeros((10, 10))
        img[5, 5] = 255
        img = normalise_image(img)

        nose.tools.assert_equal(np.amax(img), 1)
        nose.tools.assert_equal(np.amin(img), 0)

    def test_normalise_image_specified_range(self):
        img = np.zeros((10, 10))
        img = img-1
        img[5, 5] = 255
        img = normalise_image(img, new_max=100, new_min=-10)

        nose.tools.assert_equal(np.amax(img), 100)
        nose.tools.assert_equal(np.amin(img), -10)

    def test_binary_image(self):
        img = np.zeros((10, 10))
        img[5, 5] = 0.5

        bimg = binary_image(img, threshold=0.49)

        nose.tools.assert_equal(bimg.shape, (10, 10))
        nose.tools.assert_equal(bimg.dtype, 'uint8')
        nose.tools.assert_equal(bimg[5, 5], 1)
        nose.tools.assert_equal(np.count_nonzero(bimg), 1)

    def test_erode_mask(self):
        img = np.zeros((10, 10))
        img[3:7, 3:7] = morphology.square(4)

        msk = erode_mask(img, kernel_size=1)

        nose.tools.assert_equal(msk.shape, (10, 10))
        nose.tools.assert_equal(msk.dtype, 'float64')
        nose.tools.assert_equal(np.count_nonzero(msk), 4)
        nose.tools.assert_equal(np.count_nonzero(msk[4:6, 4:6]), 4)

    def test_erode_mask_different_kernel(self):
        img = np.zeros((10, 10))
        img[3:7, 3:7] = morphology.square(4)

        msk = erode_mask(img, kernel_func=morphology.square, kernel_size=3)

        nose.tools.assert_equal(msk.shape, (10, 10))
        nose.tools.assert_equal(msk.dtype, 'float64')

        nose.tools.assert_equal(np.count_nonzero(msk), 4)
        nose.tools.assert_equal(np.count_nonzero(msk[4:6, 4:6]), 4)

    def test_erode_mask_larger_kernel(self):
        img = np.zeros((10, 10))
        img[3:7, 3:7] = morphology.square(4)

        msk = erode_mask(img, kernel_func=morphology.square, kernel_size=4)

        nose.tools.assert_equal(msk.shape, (10, 10))
        nose.tools.assert_equal(msk.dtype, 'float64')

        nose.tools.assert_equal(np.count_nonzero(msk), 1)
        nose.tools.assert_equal(np.count_nonzero(msk[5, 5]), 1)

    def test_to_polar_coordinates(self):
        # zero degrees
        r, t = to_polar_coordinates(1, 0)
        nose.tools.assert_equal(t, 0)
        nose.tools.assert_equal(r, 1)

        # 180 degrees
        r, t = to_polar_coordinates(-1, 0)
        nose.tools.assert_equal(t, np.pi)
        nose.tools.assert_equal(r, 1)

        # 90 degrees
        r, t = to_polar_coordinates(0, 1)
        nose.tools.assert_equal(t, np.pi/2)
        nose.tools.assert_equal(r, 1)

        # 270 degrees
        r, t = to_polar_coordinates(0, -1)
        nose.tools.assert_equal(t, 3*np.pi/2)
        nose.tools.assert_equal(r, 1)

        # 45 degrees
        r, t = to_polar_coordinates(1, 1)
        nose.tools.assert_equal(t, np.pi/4)
        nose.tools.assert_almost_equal(r, np.cos(t) + np.sin(t))

        # 135 degrees
        r, t = to_polar_coordinates(-1, 1)
        nose.tools.assert_equal(t, np.radians(135))
        nose.tools.assert_almost_equal(r, np.sin(t) - np.cos(t))

        # 315 degrees
        r, t = to_polar_coordinates(1, -1)
        nose.tools.assert_equal(t, np.radians(315))
        nose.tools.assert_almost_equal(r, np.cos(t) - np.sin(t))

    def test_transform_2d(self):
        grid = np.ones((100, 100))
        out = transform_2d(lambda x: x+1, grid)

        nose.tools.assert_equal(out.shape, grid.shape)
        nose.tools.assert_equal(out.dtype, grid.dtype)
        nose.tools.assert_equal(out.max(), grid.max()+1)
        nose.tools.assert_equal(out.sum(), grid.sum()*2)

    def test_transform_2d_with_args(self):
        grid = np.ones((100, 100), dtype='int64')
        out = transform_2d(lambda x, f: x*f, grid, (3,))

        nose.tools.assert_equal(out.shape, grid.shape)
        nose.tools.assert_equal(out.dtype, grid.dtype)
        nose.tools.assert_equal(out.max(), grid.max()+2)
        nose.tools.assert_equal(out.sum(), grid.sum()*3)

    def test_vectorize_array(self):
        grid = np.vstack([np.arange(10) for _ in range(10)])
        out = vectorize_array(lambda x: x+1, grid)

        nose.tools.assert_equal(out.shape, grid.shape)
        nose.tools.assert_equal(out.dtype, grid.dtype)
        for row, original in zip(out, grid):
            np.testing.assert_array_equal(row, original+1)

    def test_vectorize_array_with_args(self):
        grid = np.vstack([np.arange(10) for _ in range(10)])
        out = vectorize_array(lambda x, f: x+f, grid, (3))

        nose.tools.assert_equal(out.shape, grid.shape)
        nose.tools.assert_equal(out.dtype, grid.dtype)
        for row, original in zip(out, grid):
            np.testing.assert_array_equal(row, original+3)

    def test_gaussian_kernel(self):
        kernel = gaussian_kernel(5)
        nose.tools.assert_equal(kernel[2, 2], 1)
        nose.tools.assert_almost_equal(kernel.std(), 0.1048897)

    def test_gaussian_kernel_bigger_size(self):
        kernel = gaussian_kernel(5, sigma=8)
        nose.tools.assert_equal(kernel[2, 2], 1)
        nose.tools.assert_almost_equal(kernel.std(), 0.0178893)

    def test_laplacian_of_gaussian_kernel(self):
        kernel = log_kernel(8)

        nose.tools.assert_equal(kernel.shape, (23, 23))
        nose.tools.assert_almost_equal(kernel.min(), -0.0311235)
        nose.tools.assert_almost_equal(kernel[11, 11], -0.0311235)

    def test_make_mask(self):
        img = np.zeros((10, 10))
        img[5:10] = 256

        out = make_mask(img)

        nose.tools.assert_equal(out.shape, img.shape)
        nose.tools.assert_equal(np.count_nonzero(out), 50)
        nose.tools.assert_equal(img[:5].sum(), 0)
