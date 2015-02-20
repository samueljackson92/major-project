"""
Multi-scale blob detection.

Uses a Laplacian of Gaussian pyramid to detect blobs over multiple scales.

References:

Chen, Zhili, et al. "A multiscale blob representation of mammographic
parenchymal patterns and mammographic risk assessment." Computer Analysis of
Images and Patterns. Springer Berlin Heidelberg, 2013.
"""

import math
import numpy as np
import skimage.filter as filters

from mammogram._adjacency_graph import Graph
from mammogram.utils import normalise_image
from scipy.ndimage.filters import gaussian_laplace, gaussian_filter
from sklearn import cluster
from skimage import feature, transform, io, morphology


def blob_detection(image, mask=None, max_layer=10, downscale=np.sqrt(2), sigma=8.0):
    """Performs multi-scale blob detection

    :param image: image to detect blobs in.
    :param mask: mask used on the image. (Optional)
    :param max_layer: maximum depth of image to produce
    :param downscale: factor to downscale the image by
    :param sigma: sigma of the gaussian used as part of the filter
    :yields: ndarry - filtered images at each scale in the pyramid.
    """
    blobs = multiscale_pyramid_detection(image, max_layer, downscale, sigma)
    blobs = remove_edge_blobs(blobs, image.shape)
    blobs = remove_false_positives(blobs, image, mask)
    blobs = merge_blobs(blobs, image)
    return blobs


def multiscale_pyramid_detection(image, *args):
    """ Detects blobs over multiple scales using an LoG pyramid

    :param image: the image to detect blobs in
    :param args: arguments passed to create the LoG pyramid
    :returns: list of blobs detected over multiple scales in format (y,x,sigma)
    """
    factor = np.sqrt(2)
    maxima = np.empty((0,3))
    for i, img in enumerate(log_pyramid(image, *args)):

        local_maxima = feature.peak_local_max(img, min_distance=0,
                                              threshold_abs=0.001,
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


def log_pyramid(image, max_layer, downscale, sigma):
    """Generator for a laplacian of gaussian pyramid.

    Due to the fact that mammograms are large, the pyramid is generated by
    downscaling the image for filtering, then upsampling to the original size
    to find peaks.

    :param image: image apply the LoG filter to.
    :param max_layer: maximum depth of image to produce
    :param downscale: factor to downscale the image by
    :param sigma: sigma of the gaussian used as part of the filter
    :yields: ndarry - filtered images at each scale in the pyramid.
    """
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

def remove_edge_blobs(blobs, image_shape):
    """Remove blobs detected around the edge of the image.

    :param blobs: list of blobs detected from the image
    :param image_shape: shape of the image. Provides the bounds to check.
    :returns: list of filtered blobs
    """
    img_height, img_width = image_shape

    def check_within_image(blob):
        y,x,r = blob
        return not (x - r < 0 or x + r > img_width) \
               or (y - r < 0 or y + r > img_height)

    return filter(check_within_image, blobs)


def remove_false_positives(blobs, image, mask):
    """Remove false positives from the detected blobs

    :param blobs: list of blobs detected from the image
    :param image: image that the blobs came from
    :param mask: mask used to filter the image tissue
    """
    #Find breast tissue for clustering
    tissue = image[mask==1] if mask is not None else image
    tissue = tissue.reshape(tissue.size, 1)

    clusters = cluster_image(tissue)
    threshold = compute_mean_intensity_threshold(clusters)

    print "Threshold: %f" % threshold

    #Filter blobs by mean intensity using threshold
    return filter_blobs_by_mean_intensity(blobs, image, threshold)


def cluster_image(image, num_clusters=9):
    """Segement the image into clusters using K-Means

    :param image: image to segement
    :param num_clusters: the number of clusters to use
    :returns: list of clusters. Each cluster is an array of intensity values
              belonging to a particular cluster.
    """
    k_means = cluster.KMeans(n_clusters=num_clusters)
    labels = k_means.fit_predict(image)
    return [image[labels==i] for i in range(num_clusters)]


def filter_blobs_by_mean_intensity(blobs, image, threshold):
    """Remove blobs whose mean intensity falls below a threshold

    :param blobs: list of blobs detected from the image
    :param image: image that the blobs came from
    :param threshold: threshold below which blobs are removed
    :returns: list of blobs filtered by their mean intensity
    """
    filtered_blobs = []
    for blob in blobs:
        image_section = extract_radial_blob(blob, image)
        if np.mean(image_section) > threshold:
            filtered_blobs.append(blob)

    return filtered_blobs


def compute_mean_intensity_threshold(clusters, k_largest=3):
    """Compute a threshold based on the mean intensity for tissue in a mammogram

    The threshold is the average intensity from the k most dense clusters less
    the standard deviation of those clusters.

    :param clusters: list of clusters of image segements. The k largest clusters
                     will be used to compute the average intensity threshold.
    :param k_largest: number of clusters to use to compute the threshold.
                      (Default is 3)
    :returns: int - threshold based on the mean intensity
    """
    #Find the high density clusters
    avg_cluster_intensity = np.array([np.average(c) for c in clusters])
    std_cluster_intensity = np.array([np.std(c) for c in clusters])

    indicies = avg_cluster_intensity.argsort()[-k_largest:]
    hdc_avg = avg_cluster_intensity[indicies].reshape(k_largest,1)
    hdc_std = std_cluster_intensity[indicies].reshape(k_largest,1)

    #Compute threshold from the high density cluster intensity
    return np.mean(hdc_avg) - np.std(hdc_std)


def merge_blobs(blobs, image):
    """Merge blobs found from the LoG pyramid

    :param blobs: list of blobs detected from the image
    :param image: image the blobs were found in
    :returns: a filtered list of blobs remaining after merging
    """
    #reverse so largest blobs are at the start
    blobs = np.array(blobs[::-1])
    blob_graph, remove_list = build_graph(blobs)
    remove_list += merge_intersecting_blobs(blobs, blob_graph, image)
    blobs = remove_blobs(blobs, remove_list)
    return blobs


def build_graph(blobs):
    """Build a directed graph of blobs from the largest scale to the smallest

    This will also return a list of blobs to remove because they are entirely
    contianed within a larger blob.

    :param blobs: blobs to build the graph with
    :returns: tuple containing the graph and a list of nodes to remove
    """
    g = Graph()

    remove_list = set()
    for index, blob in enumerate(blobs):

        g.add_node(index, blob)

        #check if blob has been marked as entirely within a larger blob
        if index in remove_list:
            continue

        for neighbour_index, neighbour in enumerate(blobs):
            if index != neighbour_index:
                if is_external(blob, neighbour):
                    continue
                elif is_intersecting(blob, neighbour):
                    g.add_adjacent(index, neighbour_index)
                elif is_internal(blob, neighbour):
                    remove_list.add(neighbour_index)

    return g, list(remove_list)


def merge_intersecting_blobs(blobs, blob_graph, image):
    """Merge the intersecting blobs using a directed graph

    :param blobs: list of blobs detected from the image to merge
    :param blob_graph: directed graph of blobs from largest to smallest
    :param image: image that the blobs were detected in
    :returns: list of indicies of detected blobs to remove
    """
    remove_list = set()

    for index, neighbours_indicies in blob_graph.iterate():
        blob = blob_graph.get_node(index)
        blob_section = extract_blob(blob, image)

        for neighbour_index in neighbours_indicies:
            neighbour = blob_graph.get_node(neighbour_index)

            if is_close(blob, neighbour):
                neighbour_section = extract_blob(neighbour, image)
                blob_gss = np.sum(gaussian_filter(blob_section, blob[2]))
                neighbour_gss = np.sum(gaussian_filter(neighbour_section, neighbour[2]))

                if blob_gss > neighbour_gss:
                    remove_list.add(neighbour_index)
                elif blob_gss < neighbour_gss:
                    remove_list.add(index)

    return list(remove_list)


def is_intersecting(a,b):
    """ Check if two blobs intersect each other

    :param a: first blob. This is larger than b.
    :param b: second blob. This is smaller than a.
    :returns: if the radius of b overlaps with the radius of a
    """
    ay, ax, ar = a
    by, bx, br = b

    d = math.sqrt((ax - bx)**2 + (ay - by)**2)
    return ar - br < d and d < ar + br


def is_internal(a,b):
    """ Check if blob b is within blob a

    :param a: first blob. This is larger than b.
    :param b: second blob. This is smaller than a.
    :returns: if b is inside the radius of a
    """
    ay, ax, ar = a
    by, bx, br = b

    d = math.sqrt((ax - bx)**2 + (ay - by)**2)
    return d <= ar - br


def is_external(a,b):
    """ Check if blob b is outside blob a

    :param a: first blob. This is larger than b.
    :param b: second blob. This is smaller than a.
    :returns: if b is outside the radius of a
    """
    ay, ax, ar = a
    by, bx, br = b

    d = math.sqrt((ax - bx)**2 + (ay - by)**2)
    return d >= ar + br


def is_close(a,b,alpha=0.0):
    """ Check if two blobs are close to one another

    :param a: first blob. This is larger than b.
    :param b: second blob. This is smaller than a.
    :param alpha: The amount of overlap allowed between blobs
    :returns: if blobs are close
    """
    if alpha < 0 or alpha > 1:
        raise ValueError("Value of alpha must be between 0 and 1.")

    ay, ax, ar = a
    by, bx, br = b

    d = math.sqrt((ax - bx)**2 + (ay - by)**2)
    return d <= ar - br*alpha


def extract_blob(blob, image):
    """ Extract the pixels that make up the blob's neighbourhood

    :param blob: the blob to extract
    :param image: the image to extract the blob from
    :returns: extracted square neighbourhood
    """
    y,x,r = blob
    hs, he = y - math.ceil(r), y + math.ceil(r)+1
    ws, we = x - math.ceil(r), x + math.ceil(r)+1
    image_section = image[hs:he,ws:we]
    return image_section


def extract_radial_blob(blob, image):
    """ Extract the pixels that make up the blob's neighbourhood

    This uses a disk to extract only the pixels within the radius of the blob
    :param blob: the blob to extract
    :param image: the image to extract the blob from
    :returns: extracted disk neighbourhood
    """
    image_section = extract_blob(blob, image)
    kernel = morphology.disk(math.ceil(blob[2]))
    image_section = image_section[kernel==1]
    image_section = image_section.reshape(image_section.size, 1)
    return image_section


def remove_blobs(blobs, remove_list):
    """Remove blobs corresponding to the indicies in remove_list

    :param blobs: list of blobs to filter
    :param remove_list: list of indicies to remove from the blob list
    :returns: filtered list of blobs
    """
    remove_list = np.array(remove_list)
    mask = np.ones_like(blobs,dtype=bool)
    mask[remove_list] = False
    blobs = blobs[mask]
    blobs = blobs.reshape(blobs.size/3, 3)
    return blobs
