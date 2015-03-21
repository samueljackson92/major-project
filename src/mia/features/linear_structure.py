"""Linear structure features

This module provides an implementation of the orientated bins feature for
detecting breast ducts in mammograms.


Reference: Reyer Zwiggelaar, Tim C. Parr, and Christopher J. Taylor.
"Finding Orientated Line Patterns in Digital Mammographic Images." BMVC. 1996.
"""

import pandas as pd
from skimage import measure

from mia.features._orientated_bins import orientated_bins
from mia.features._nonmaximum_suppression import nonmaximum_suppression
from mia.utils import binary_image, skeletonize_image, erode_mask


def detect_linear(img, msk, radius=10, nbins=12, threshold=4e-2):
    """Compute linear features from an image

    Uses orientated bins with nonmaximum suppression and binary thinning.

    :param img: image to find linear features in
    :param radius: radius of the bins to use
    :param nbins: number of bins to use
    :returns: tuple -- containing (line_image, regions)
    """
    line_strength, line_orientation = orientated_bins(img, radius, nbins=nbins)

    msk = erode_mask(msk, kernel_size=10)
    line_strength = line_strength * msk

    line_strength_suppressed = nonmaximum_suppression(line_strength,
                                                      line_orientation, nbins)

    line_image = binary_image(line_strength_suppressed, threshold)
    line_image = skeletonize_image(line_image, 50, dilation_size=10)

    # find image regions
    line_image_labelled = measure.label(line_image)
    regions = measure.regionprops(line_image_labelled)

    features = [[props.area] + list(props.bbox) for props in regions]
    return pd.DataFrame(features, columns=['area', 'min_row', 'min_col',
                                           'max_row', 'max_col']), line_image


def extract_feature(props, image):
    """ Extract the area of an image belonging to a feature given a bounding box

    :param props: the properties of the region
    :param image: image to extract the region from
    :returns: ndarray -- section of the image within the bounding box
    """
    hs, ws = props['min_row'], props['min_col']
    he, we = props['max_row'], props['max_col']
    image_section = image[hs:he, ws:we]
    return image_section
