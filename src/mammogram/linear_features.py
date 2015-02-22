from skimage import measure

from mammogram.orientated_bins import orientated_bins
from mammogram.nonmaximum_suppression import nonmaximum_suppression
from mammogram.utils import *

def linear_features(img, radius, nbins, threshold):
    """Compute linear features from an image

    Uses orientated bins with nonmaximum suppression and binary thinning.

    :param img: image to find linear features in
    :param radius: radius of the bins to use
    :param nbins: number of bins to use
    :returns: tuple -- containing (line_image, regions)
    """
    line_strength, line_orientation = orientated_bins(img, radius, nbins=nbins)
    line_strength_suppressed = nonmaximum_suppression(line_strength, line_orientation, nbins)

    line_image = binary_image(line_strength_suppressed, threshold)
    line_image = skeletonize_image(line_image, 50, dilation_size=1)

    #find image regions
    line_image = measure.label(line_image)
    regions = measure.regionprops(line_image)

    return line_strength_suppressed, regions

def extract_feature(props, image):
    """ Extract the area of an image belonging to a feature given a bounding box

    :param props: the properties of the region
    :param image: image to extract the region from
    :returns: ndarray -- section of the image within the bounding box
    """
    hs, ws, he, we = props.bbox
    image_section = image[hs:he,ws:we]
    return image_section
