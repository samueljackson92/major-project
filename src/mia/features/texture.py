"""Texture feature detection

This module provides a collection of functions for running texture based
features on patches of images.

"""
import pandas as pd
import numpy as np
import scipy.stats as stats
import skimage
from skimage import feature, filters, transform
from scipy.ndimage.filters import generic_filter

from mia.features.blobs import extract_blob
from mia.features.linear_structure import extract_line
from mia.utils import vectorize_array

GLCM_FEATURES = ['contrast', 'dissimilarity', 'homogeneity', 'energy',
                 'correlation']


def detect_texture(img, patches):

    def _extract_texture(row):
        _, patch = row
        if 'radius' in patch:
            img_patch = extract_blob(patch[['x', 'y', 'radius']], img)
        else:
            small_img = transform.pyramid_reduce(img, 4)
            img_patch = extract_line(patch, small_img)
        return texture_props(img_patch)

    frames = map(_extract_texture, patches.iterrows())
    features = pd.concat(frames)
    features.index = patches.index
    return pd.concat([patches, features], axis=1)


def texture_props(img_blob):
    thetas = np.arange(0, np.pi, np.pi/8)
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy']

    features = glcm_features(img_blob, [1], thetas, props)
    # compute mean across all orientations
    features = np.mean(features, axis=2)

    return pd.DataFrame(features.T, columns=props)


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
    image = skimage.img_as_ubyte(image)
    glcm = feature.greycomatrix(image, distances, orientations, normed=True)
    return np.array([feature.greycoprops(glcm, prop) for prop in properties])


def glcm_feature(image, distance, orientation, prop):
    glcm = feature.greycomatrix(image, [distance], [orientation], normed=True)
    return feature.greycoprops(glcm, prop)


def image_orthotope_statistics(image_orthotope):
    """ Calculate the mean, std, and skew of an image orthotope

    :param image_orthotope: the image orthotope
    :returns: feature matrix containing the column vectors [mean, std, skew]
              the components of which first order statics of a single image.
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
    texture_props = glcm_features(img_section, distances, orientations,
                                  properties)

    # Compute the mean of all orientations and distances
    avg_tex_features = np.mean(texture_props.mean(axis=2), axis=1)
    return avg_tex_features


def blob_texture_props(image, blobs, properties, distances, orientations):
    """ Compute the GLCM texture properties of blobs in an image

    :param image: the image the blobs were found in
    :param blobs: list of blobs found in the image
    :param properties: list of properties to extract using the GLCM
    :param distances: list of distances to use to compute the GLCM
    :param orientations: list of orientations to use to compute the GLCM
    :returns: array containg the mean, std, max and min of the properties
    """
    tex_props = vectorize_array(compute_texture_features_from_blob, blobs,
                                image, properties, distances, orientations)
    tex_props = np.hstack([tex_props.mean(axis=0), tex_props.std(axis=0),
                           tex_props.max(axis=0), tex_props.min(axis=0)])
    return tex_props


def texture_from_clusters(clusters):
    thetas = np.arange(0, np.pi, np.pi/8)
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy']

    tex_features = []
    for i, cluster in enumerate(clusters):
        prop_suffix = '_cluster_%d' % (i+1)
        col_names = [name + prop_suffix for name in props]

        features = glcm_features(cluster, [1], thetas, props)
        # compute mean across all orientations
        features = np.mean(features, axis=2)

        df = pd.DataFrame(features.T, columns=col_names)
        tex_features.append(df)

    return pd.concat(tex_features, axis=1)


def filter_image_for_texture(img, orientation, prop, kernel_size=5):

    def filter_texture_non_zero(window, *args):
        window = window.reshape((kernel_size, kernel_size))
        if np.count_nonzero(window) > 0:
            return glcm_feature(window, *args)
        else:
            return 0

    return generic_filter(img, filter_texture_non_zero,
                          size=kernel_size,
                          extra_arguments=(1, orientation, prop))
