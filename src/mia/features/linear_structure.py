"""Linear structure features

This module provides an implementation of the orientated bins feature for
detecting breast ducts in mammograms.


Reference: Reyer Zwiggelaar, Tim C. Parr, and Christopher J. Taylor.
"Finding Orientated Line Patterns in Digital Mammographic Images." BMVC. 1996.
"""

import pandas as pd
import numpy as np
from skimage import measure, morphology, transform

from mia.features._orientated_bins import orientated_bins
from mia.features._nonmaximum_suppression import nonmaximum_suppression
from mia.utils import erode_mask, normalise_image


def detect_linear(img, msk, radius=20, nbins=8, threshold=5e-3):
    """Compute linear features from an image

    Uses orientated bins with nonmaximum suppression and binary thinning.

    :param img: image to find linear features in
    :param radius: radius of the bins to use
    :param nbins: number of bins to use
    :returns: tuple -- containing (line_image, regions)
    """

    img = transform.pyramid_reduce(img, 4)
    msk = transform.pyramid_reduce(msk, 4)

    line_strength, line_orientation = orientated_bins(img, radius, nbins=nbins)
    line_strength[line_strength < 0.04] = 0

    line_strength = nonmaximum_suppression(line_strength,
                                           line_orientation, nbins,
                                           kernel_size=3)
    msk = erode_mask(msk, kernel_size=10)
    line_strength[msk == 0] = 0

    line_image = normalise_image(line_strength)
    line_image = morphology.binary_dilation(line_image, morphology.disk(1))

    # find image regions
    line_image = morphology.remove_small_objects(line_image, 50,
                                                 connectivity=8)
    line_image_labelled = measure.label(line_image)
    regions = measure.regionprops(line_image_labelled)

    features = [[props.area] + list(props.bbox) for props in regions]
    return pd.DataFrame(features, columns=['area', 'min_row', 'min_col',
                                           'max_row', 'max_col']), line_image


def line_props(feature_set):
    """Contstruct a feature matrix from a list of blobs

    :param blobs: 3D list of blobs to compute statistics on.
    :returns: DataFrame - the feature matrix of statistics.
    """

    img_series = pd.Series(feature_set['area'])

    img_described = img_series.describe()
    stats = pd.DataFrame(img_described.as_matrix()).T
    stats.columns = img_described.index
    stats['skew'] = img_series.skew()

    if img_series.size > 1:
        stats['kurtosis'] = img_series.kurtosis()
    else:
        stats['kurtosis'] = np.NaN

    mean = stats['mean'][0]
    stats['upper_dist_count'] = img_series[img_series > mean].shape[0]

    return stats


def extract_line(props, image):
    """ Extract the area of an image belonging to a feature given a bounding box

    :param props: the properties of the region
    :param image: image to extract the region from
    :returns: ndarray -- section of the image within the bounding box
    """
    hs, ws = props['min_row'], props['min_col']
    he, we = props['max_row'], props['max_col']
    image_section = image[hs:he, ws:we]
    return image_section
