"""This module provides a collection of functions for running intensity based
features on patches of images.
"""
import pandas as pd
import numpy as np

from mia.features.blobs import extract_blob
from mia.features.linear_structure import extract_line
from skimage import transform


def detect_intensity(img, patches):
    """Detect intensity features from a collection of patches.

    :param img: the image to detect intensity features in.
    :param patches: the patches of blobs to look at. Either defined by a radius
        or by a min/max col/row.

    :returns: DataFrame -- of intensity stats for each patch of image.
    """
    def _extract_intensity(row):
        _, patch = row
        if 'radius' in patch.index:
            img_patch = extract_blob(patch[['x', 'y', 'radius']], img)
        else:
            small_img = transform.pyramid_reduce(img, 4)
            img_patch = extract_line(patch, small_img)
        return intensity_props(img_patch)

    frames = map(_extract_intensity, patches.iterrows())
    features = pd.concat(frames)
    features.index = patches.index
    return pd.concat([patches, features], axis=1)


def intensity_props(img):
    """Return the intensity properties for a patch of image

    :param img: patch of image to create statistics for.
    :returns: DataFrame -- containing the statistics for the patch.
    """
    # create dataframe of histogram features from the described series
    img_series = pd.Series(img.flatten())

    img_described = img_series.describe()
    stats = pd.DataFrame(img_described.as_matrix()).T
    stats.columns = img_described.index
    stats['skew'] = img_series.skew()

    if img_series.size > 1:
        stats['kurtosis'] = img_series.kurtosis()
    else:
        stats['kurtosis'] = np.NaN

    return stats
