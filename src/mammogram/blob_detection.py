"""
Blob Detection
"""
import numpy as np
from skimage import feature, transform, io
import skimage.filter as filters
from scipy.ndimage.filters import gaussian_laplace, generic_filter, laplace, gaussian_filter

def blob_detection(image, mask, min_sigma=8, max_sigma=10*np.sqrt(2), num_sigma=10, threshold=.1, overlap=.5):

    for img in laplacian_pyramid(image, mask, max_layer=10, downscale=np.sqrt(2), sigma=8):
        io.imshow(img)
        io.show()

    # sigma_list = np.linspace(min_sigma, max_sigma, num_sigma)
    #
    # # computing gaussian laplace
    # # s**2 provides scale invariance
    # gl_images = [-gaussian_laplace(image, s) * s ** 2 for s in sigma_list]
    # image_cube = np.dstack(gl_images)
    #
    # local_maxima = feature.peak_local_max(image_cube, threshold_abs=threshold,
    #                               footprint=np.ones((3, 3, 3)),
    #                               threshold_rel=0.0,
    #                               exclude_border=False)
    #
    # # Convert the last index to its corresponding scale value
    # local_maxima[:, 2] = sigma_list[local_maxima[:, 2]]
    # return local_maxima

def laplacian_pyramid(image, mask, max_layer, downscale, sigma):
    layer = 0
    while layer != max_layer:
        smoothed = filter_image(image, mask, sigma)
        yield smoothed

        image = transform.rescale(image, 1/downscale)
        layer += 1


def filter_image(image, mask, sigma):

    filtered_image = np.zeros(image.shape)
    laplace(image, filtered_image)
    filtered_image[mask!=1]=0
    generic_filter(filtered_image, func, output=filtered_image, extra_arguments=(sigma,))
    return filtered_image


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
