"""
Blob Detection
"""
import numpy as np
from skimage import feature, transform, io
import skimage.filter as filters
from scipy.ndimage.filters import laplace
from mammogram.utils import normalise_image

def blob_detection(image, mask, min_sigma=8, max_sigma=10*np.sqrt(2), num_sigma=10, threshold=.1, overlap=.5):

    for img in laplacian_pyramid(image, mask, max_layer=10, downscale=np.sqrt(2), sigma=1.0):
        pass
        # io.imshow(img)
        # io.show()

def laplacian_pyramid(image, mask, max_layer, downscale, sigma):
    import scipy.ndimage.filters as filters

    def gaussian_kernel(size, fwhm = 3):
        """ Make gaussian kernel."""

        fwhm = 2.355*fwhm

        x = np.arange(0, size, 1, float)
        y = x[:,np.newaxis]

        x0 = y0 = size // 2
        return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

    def log_kernel(size, sigma):
        g = gaussian_kernel(size+1, sigma)
        log = laplace(g, mode='wrap')
        log = log[1:-1,1:-1] #remove the edge crap
        return log

    log = log_kernel(5, sigma)
    layer = 0
    while layer != max_layer:
        io.imshow(image)
        io.show()

        # from convolve_tools import deformable_covolution
        # smoothed = deformable_covolution(image, mask, log)
        # yield smoothed
        image = normalise_image(image)
        image = transform.rescale(image, 1./downscale)
        # mask = transform.rescale(mask.astype('float64'), 1/downscale)
        # mask = mask.astype('int')
        layer += 1


def vanilla_log(img):
    num_sigma = 10
    sigma_min = 8.0
    sigma_max = sigma_min * np.sqrt(2) * num_sigma
    blobs = feature.blob_log(img,
                             min_sigma=sigma_min,
                             max_sigma=sigma_max,
                             overlap=0.8,
                             num_sigma=num_sigma,
                             threshold=0.01)
    return blobs
