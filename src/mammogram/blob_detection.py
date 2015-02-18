"""
Blob Detection
"""
import numpy as np
from skimage import feature, transform, io
import skimage.filter as filters
from scipy.ndimage.filters import laplace, gaussian_laplace
from mammogram.utils import normalise_image

def blob_detection(image):
    return multiscale_pyramid_detection(image)

def multiscale_pyramid_detection(image):
    factor = np.sqrt(2)
    maxima = np.empty((0,3))
    for i, img in enumerate(laplacian_pyramid(image, max_layer=10, downscale=factor, sigma=8.0)):

        local_maxima = feature.peak_local_max(img, min_distance=0, threshold_abs=0.0,
                                              footprint=np.ones((5, 5)),
                                              threshold_rel=0.0,
                                              exclude_border=False)

        #Generate array of sigma sizes for this level.
        local_sigma = 8.0*factor**i
        sigmas = np.empty((local_maxima.shape[0], 1))
        sigmas.fill(local_sigma)

        #stack detections together into single list of blobs.
        local_maxima = np.hstack((local_maxima, sigmas))
        maxima = np.vstack((maxima, local_maxima))

    return maxima

def laplacian_pyramid(image, max_layer, downscale, sigma):
    image = normalise_image(image)

    layer = 0
    while layer != max_layer:

        log_filtered = -gaussian_laplace(image, sigma)

        #upscale to original image size
        if layer > 0:
            log_filtered = transform.rescale(log_filtered, downscale**layer)

        yield log_filtered

        #downscale image, but keep sigma the same.
        image = transform.rescale(image, 1./downscale)
        layer += 1
