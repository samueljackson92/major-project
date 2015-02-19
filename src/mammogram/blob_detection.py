"""
Blob Detection
"""
import math
import numpy as np
from sklearn import cluster
from skimage import feature, transform, io, morphology
import skimage.filter as filters
from scipy.ndimage.filters import laplace, gaussian_laplace
from mammogram.utils import normalise_image

def blob_detection(image, mask, max_layer=10, downscale=np.sqrt(2), sigma=8.0):
    blobs = multiscale_pyramid_detection(image, max_layer, downscale, sigma)
    blobs = remove_edge_blobs(blobs, image.shape)
    blobs = remove_false_positives(blobs, image, mask)
    return blobs


def multiscale_pyramid_detection(image, *args):
    factor = np.sqrt(2)
    maxima = np.empty((0,3))
    for i, img in enumerate(laplacian_pyramid(image, *args)):

        local_maxima = feature.peak_local_max(img, min_distance=0, threshold_abs=0.001,
                                              footprint=np.ones((5, 5)),
                                              threshold_rel=0.0,
                                              exclude_border=False)

        if len(local_maxima) > 0:
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

        log_filtered = -gaussian_laplace(image, sigma, mode='reflect')

        #upscale to original image size
        if layer > 0:
            log_filtered = transform.rescale(log_filtered, downscale**layer)

        yield log_filtered

        #downscale image, but keep sigma the same.
        image = transform.rescale(image, 1./downscale)
        layer += 1


def merge_blobs(blobs, image_shape):
    return blobs


def remove_edge_blobs(blobs, image_shape):
    img_height, img_width = image_shape

    def check_within_image(blob):
        y,x,r = blob
        return not (x - r < 0 or x + r > img_width) or (y - r < 0 or y + r > img_height)

    return np.array(filter(check_within_image, blobs))

def remove_false_positives(blobs, image, mask):
    #Find breast tissue for clustering
    tissue = image[mask==1]
    tissue = tissue.reshape(tissue.size, 1)

    clusters = cluster_image(tissue)
    threshold = compute_mean_intensity_threshold(clusters)

    print "Threshold: %f" % threshold

    #Filter blobs by mean intensity using threshold
    return filter_blobs_by_mean_intensity(blobs, image, threshold)

def cluster_image(image, num_clusters=9):
    #Segment the breast tissue into clusters
    k_means = cluster.KMeans(n_clusters=num_clusters)
    labels = k_means.fit_predict(image)

    clusters = []
    for i in range(num_clusters):
        c = image[labels==i]
        clusters.append(c)

    return clusters

def filter_blobs_by_mean_intensity(blobs, image, threshold):
    filtered_blobs = []
    for blob in blobs:
        y,x,r = blob

        kernel = morphology.disk(math.ceil(r))
        hs, he = y - math.ceil(r), y + math.ceil(r)+1
        ws, we = x - math.ceil(r), x + math.ceil(r)+1

        image_section = image[hs:he,ws:we]
        image_section = image_section[kernel==1]
        image_section = image_section.reshape(image_section.size, 1)

        if np.mean(image_section) > threshold:
            filtered_blobs.append(blob)

    return filtered_blobs

def compute_mean_intensity_threshold(clusters):
    #Find the high density clusters
    avg_cluster_intensity = np.array([np.average(c) for c in clusters])
    std_cluster_intensity = np.array([np.std(c) for c in clusters])

    indicies = avg_cluster_intensity.argsort()[-3:]
    hdc_avg = avg_cluster_intensity[indicies].reshape(3,1)
    hdc_std = std_cluster_intensity[indicies].reshape(3,1)

    #Compute threshold from the high density cluster intensity
    return np.mean(hdc_avg) - np.std(hdc_std)
