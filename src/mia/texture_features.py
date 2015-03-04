"""Texture filters

"""
import numpy as np
import scipy.stats as stats
import skimage.filter as filters
from skimage import feature

from mia.blob_features import extract_blob
from mia.utils import normalise_image

GLCM_FEATURES = ['contrast', 'dissimilarity', 'homogeneity', 'energy',
                 'correlation']


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


def compute_texture_features_from_blob(blob, image, properties,
                                       distances, orientations):
    """ Compute texture features from the patch of image defined by a blob

    :param blob: the blob defining the patch of image to use.
    :param image: the image to extract the blob from.
    :param properties: list of properties to use during the extraction
    :param orientations: list of orientations to use to compute the GLCM.
    :param distances: list of distances to use to compute the GLCM.
    :returns: ndarray of the mean value for each property over all angles
              and distances
    """
    img_section = extract_blob(blob, image)
    # GCLM only supports images that have values not in the 0-1 range
    img_section = normalise_image(img_section, 0, 255)
    texture_props = glcm_features(img_section, distances, orientations,
                                  properties)

    # Compute the mean of all orientations and distances
    avg_tex_features = np.mean(texture_props.mean(axis=2), axis=1)
    return avg_tex_features


def vectorize_array(f, array, *args):
    """ Helper function to vectorize across the rows of a 2D numpy array

    :params f: function to vectorize
    :params array: 2darray to iterate over.
    :params args: list of arguments to pass to the function f
    :returns: ndarray of the results of applying the function to each row.
    """
    return np.array([f(row, *args) for row in array])


def blob_texture_props(image, blobs, properties, distances, orientations):
    tex_props = vectorize_array(compute_texture_features_from_blob, blobs,
                                image, properties, distances, orientations)
    tex_props = np.hstack([tex_props.mean(axis=0), tex_props.std(axis=0),
                           tex_props.max(axis=0), tex_props.min(axis=0)])
    return tex_props
