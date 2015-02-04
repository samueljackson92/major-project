import math
from skimage import morphology

def non_maximal_suppression(img, kernel):
    max_dst = morphology.dilation(img, kernel)
    img = img * (img == max_dst)
    return img


def to_polar_coordinates(x, y):
    """Convert the 2D pixel coordinates to polar coordinates

    :param x: x coordinate of the point
    :param y: y coordinate of the point
    :returns: tuple -- (r,phi) of the point as polar coordinates
    """
    theta = math.atan2(y, x)
    if theta < 0: theta = theta + 2 * math.pi
    return math.hypot(x, y), theta
