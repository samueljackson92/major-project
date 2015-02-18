import unittest
import nose.tools
import numpy as np

from skimage import io, transform
from test_utils import generate_blob
from mammogram.blob_detection import blob_detection
from mammogram.plotting import plot_blobs

class BlobDetectionTests(unittest.TestCase):

    def test_detect_blob(self):
        img = generate_blob()
        mask = np.zeros(img.shape)
        mask[50:] = np.ones((50,100))

        # blob = blob_detection(img, mask)
        # plot_blobs(img, blob)
        # assert False


    def test_laplace_of_gauss(self):
        import scipy.ndimage.filters as filters

        def gaussian_kernel(size, fwhm = 3):
            """ Make gaussian kernel."""

            fwhm = 2.355*fwhm

            x = np.arange(0, size, 1, float)
            y = x[:,np.newaxis]

            x0 = y0 = size // 2
            return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

        def log_kernel(size, sigma):
            g = gaussian_kernel(size+2, sigma)
            l = np.array([[0,1,0],
                          [1,-4,1],
                          [0,1,0]])
            log = filters.convolve(g, l)
            log = log[1:-1,1:-1] #remove the edge crap
            return log


        log = log_kernel(100, 20)

        # msk = np.ones((100,100))
        # msk[50:,:] = np.zeros((50,100))
        # log = log*msk
        # io.imshow(log)
        # io.show()

        # assert False
