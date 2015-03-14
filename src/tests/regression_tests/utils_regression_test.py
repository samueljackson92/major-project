import unittest
import nose.tools

from scipy.ndimage import filters
import skimage
from skimage import io
from mia.utils import preprocess_image
from ..test_utils import get_file_path
import numpy as np

def gaussian_kernel(size, fwhm = 3):
    """ Make gaussian kernel."""

    fwhm = 2.355*fwhm

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    x0 = y0 = size // 2
    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

def log_kernel(size, sigma):
    g = gaussian_kernel(size+1, sigma)
    log = filters.laplace(g, mode='wrap')
    log = log[1:-1,1:-1] #remove the edge crap
    return log

class UtilsRegressionTest(unittest.TestCase):

    # def test_load_image(self):
    #     img_path = get_file_path("../../../data/p214-010-60001-cr.png")
    #     msk_path = get_file_path("../../../data/masks/f214-010-60001-cr_mask.png")
    #     img, msk = preprocess_image(img_path, msk_path)
    #
    #     # log = log_kernel(10.0, 8.0)
    #     #
    #     # from convolve_tools import deformable_covolution
    #     # msk = skimage.img_as_float(msk)
    #     # a = -deformable_covolution(img, msk, log)
    #     #
    #     # # img = -filters.gaussian_laplace(msk, 8.0)
    #     # io.imshow(a)
    #     # io.show()
    #
    #
    #     nose.tools.assert_equal(img.shape, (100, 100))
    #     nose.tools.assert_equal(msk, None)

    def test_load_syntehtic_image(self):
        img_path = get_file_path("../../../data/test_Mix_DPerc0_c_0.dcm")
        msk_path = get_file_path("../../../data/masks/test_Mix_DPerc0_c_0.png")
        img, msk = preprocess_image(img_path, msk_path)


        log = log_kernel(24.0, 8.0)
        io.imshow(log)
        io.show()

        from convolve_tools import deformable_covolution
        msk = skimage.img_as_float(msk)
        import time
        s = time.time()
        a = -deformable_covolution(img, msk, log)
        # a = -filters.gaussian_laplace(msk, 8.0)
        e = time.time()
        print e-s
        io.imshow(a)
        io.show()

        nose.tools.assert_equal(img.shape, (100, 100))
        nose.tools.assert_equal(msk, None)
