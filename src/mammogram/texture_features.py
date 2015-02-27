"""Texture filters

"""
import numpy as np
import scipy.stats as stats
import skimage.filter as filters
from skimage import feature

def gabor_features(image, frequencies, orientations):
    """Filter a image using a Gabor filter bank

    :param image: image to filter with the Gabor bank
    :param frequencies: list of frequency indicies to use. These will be
                        converted using the formula 2**(-(f+2)/2) * pi
    :param orientations: list of orientations to use in rad
    :returns: image cube of shape (f,o,h,w) where f is the number of
              frequencies, o is the number of orientations and h and w are the
              height and width of the image
    """
    frequencies = 2**(-(frequencies+2)/2) * np.pi

    gabor_magnitudes = []
    for i, freq in enumerate(frequencies):
        row = []
        for j, theta in enumerate(orientations):
            real, imag = filters.gabor_filter(image, freq, theta)
            magnitude = np.sqrt(real**2 + imag**2)
            row.append(magnitude)
        gabor_magnitudes.append(row)

    return np.array(gabor_magnitudes)


def glcm_features(image, distances, orientations, properties):
    """ Compute properties of the Grey-Level Co-occurance Matrix for an image

    :param image: image to compute the GLCM for
    :param distances: distances to compute the GLCM across
    :param orientations: angles to computes the orientations over
    :param properties: list of properties to compute from the GLCM
    :returns: matrix of features computed from the GLCM
    """
    glcm = feature.greycomatrix(image, distances, orientations)
    return np.array([feature.greycoprops(glcm, prop) for prop in properties])



def image_orthotope_statistics(image_orthotope):
    """ Calculate the mean, std, and skew of an image orthotope

    :param image_orthotope: the image orthotope
    :returns: feature matrix containing the column vectors [mean, std, skew] the
              components of which first order statics of a single image.
    """
    means = np.array([np.mean(image) for image in image_orthotope])
    stds = np.array([np.std(image) for image in image_orthotope])
    skew = np.array([stats.skew(image.flatten()) for image in image_orthotope])
    return np.array([means, stds, skew])
