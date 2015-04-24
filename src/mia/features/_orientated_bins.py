"""Implementation of the Orientated Bins filter.

Used to compute the strength and orientation of linear features in an image.

Reference: Reyer Zwiggelaar, Tim C. Parr, and Christopher J. Taylor.
"Finding Orientated Line Patterns in Digital Mammographic Images." BMVC. 1996.
"""

import itertools
import math
import numpy as np

from scipy.ndimage import filters

from mia import utils

__all__ = ["orientated_bins"]


def orientated_bins(img, radius, nbins=8):
    """Filter an image using orientated bins

    :param img: the image to filter
    :param radius: the radius of the circular neighbourhood to use
    :param nbins: the number of bins to use (default 8)
    :returns: tuple -- containing the line strength and orientation images
              resulting from the filtering
    """
    orientated_bins, neighbourhood = compute_filters(radius, nbins=8)
    return filter_image(img, orientated_bins, neighbourhood)


def compute_filters(radius, nbins):
    """Create the filers for each bin and the surrounding neighbourhood

    :param radius: radius of the circle neighbourhood to use
    :param nbins: number of bins to create
    :returns: tuple -- containing a list of orientated bins and the
              neighbourhood filter
    """
    sectors = create_sectors(nbins, radius)
    orientated_bins = create_orientated_bins(sectors, radius)
    neighbourhood = create_neighbourhood_kernel(orientated_bins, radius)
    return orientated_bins, neighbourhood


def create_sectors(nbins, radius):
    """Compute all of the sectors for the required number of bins

    :param nbins: number of bins (sectors) to create
    :param radius: the radius of each of the bins (sectors)
    :returns: list -- list of sectors representing each bin
    """
    theta_offset = math.pi / nbins
    theta_step = math.pi / float(nbins/2)

    sectors = []
    centre_point = (radius/2, radius/2)
    window_coordinates = list(itertools.product(range(radius), repeat=2))

    for i in range(nbins):
        sector = np.zeros(shape=(radius, radius), dtype="int64")

        start_theta = theta_offset + i * theta_step
        end_theta = (start_theta + theta_step)

        start_theta = start_theta % (2*math.pi)
        end_theta = end_theta % (2*math.pi)

        sector = create_sector(window_coordinates, centre_point,
                               radius, start_theta, end_theta)

        sectors.append(sector)

    return sectors


def create_sector(window_coordinates, centre_point, radius,
                  start_theta, end_theta):
    """Compute a sector bins using the centre point and a start and end radius

    :param window_coordinates: the coordinates for each pixel in the window
    :param centre_point: the origin of the sector
    :param radius: the radius of the sector
    :param start_theta: the starting angle of the sector in rad
    :param end_theta: the end angle of the sector in rad
    :returns: ndarray -- binary array representing the values in the sector
    """

    sector = np.zeros(shape=(radius, radius), dtype="int64")

    for point in window_coordinates:
        if point == centre_point:
            continue

        x, y = point
        centre_x, centre_y = centre_point

        offset_x = x - centre_x
        offset_y = y - centre_y

        polar_point = utils.to_polar_coordinates(offset_x, offset_y)
        if in_sector_bounding_box(polar_point, radius,
                                  start_theta, end_theta):
            sector[x, y] = 1

    return sector


def create_orientated_bins(sectors, radius):
    """ Create the orientated bins from circle sectors

    Combines adjacent sectors into a single filter
    :param sectors: a list of pre-computed sectors
    :param radius: radius of each of the sectors
    :returns: list -- the combined orientated bins
    """
    nbins = len(sectors)/2
    orientated_bins = []
    sector_pairs = zip(sectors[:nbins], sectors[nbins:])

    for left, right in sector_pairs:
        orientated_bin = np.zeros(shape=(radius, radius), dtype="int64")
        orientated_bin[np.where(left)] = 1
        orientated_bin[np.where(right)] = 1
        orientated_bins.append(orientated_bin)

    return orientated_bins


def create_neighbourhood_kernel(orientated_bins, radius):
    """ Create the neighbourhood of the orientated bins

    :param orientated_bins: the orientated bins making up the neighbourhood
    :param radius: the radius of the neighbourhood
    :returns: ndarray -- the filer for the neighbourhood containing the bins
    """
    neighbourhood = np.zeros(shape=(radius, radius), dtype="int64")
    for obin in orientated_bins:
        neighbourhood[np.where(obin)] = 1
    return neighbourhood


def filter_image(img, orientated_bins, neighbourhood):
    """Compute the line strength and line orientation images

    This filters the image with each of the orientated bins to find the average
    intensity in each direction. The maximum value over all directions
    indicates the orientation. Line strength is computed by subtracting the
    average of the neighbourhood from the maximum orientation.

    :param img: the image to filter
    :param orientated_bins: list of the pre-computed sector filters
    :param neighbourhood: the pre-computed neighbourhood fitler
    :returns: tuple -- containing the strength and orientation images
    """
    average_images = np.array([apply_filter(img, kernel)
                               for kernel in orientated_bins])
    neighbourhood_image = apply_filter(img, neighbourhood)

    orientation_image = np.argmax(average_images, axis=0)
    strength_image = np.amax(average_images, axis=0)
    strength_image -= neighbourhood_image
    return strength_image, orientation_image


def apply_filter(img, kernel):
    """Apply the filter to every pixel in the image

    This uses scipy's generic_filter to computer the average intensity across
    the bins defined by the kernel parameter.

    :param img: image to apply the filter to.
    :param kernel: filter to apply to use as the footprint argument to
                   generic_filter
    :returns: ndarray -- copy of the img with the filter applied
    """
    def filter_func(x):
        """Function calculate the sum of the region filtered by the kernel"""
        # ignore any kernel that sums to zero.
        if (np.count_nonzero(x) == 0):
            return 0
        else:
            # vanilla python sum is a bit faster than numpy here
            return sum(x)

    result_image = np.zeros(shape=img.shape)
    total_pixels = np.count_nonzero(kernel)
    filters.generic_filter(img, filter_func, footprint=kernel,
                           output=result_image)
    result_image /= total_pixels
    return result_image


def in_sector_bounding_box(polar_point, radius, start_theta, end_theta):
    """Check if a polar coordinate lies within the segment of a circle

    :param polar_point: tuple representing the point to check (r,phi)
    :param radius: radius of the segement
    :param start_theta: start angle of the segement in rad
    :param end_theta: end angle of the segement in rad
    :returns: bool -- whether a point lies in the sector
    """
    point_radius, point_theta = polar_point

    if start_theta > end_theta:
        return (point_radius < radius and (point_theta >= start_theta or
                                           point_theta < end_theta))
    else:
        return (point_radius < radius and point_theta >= start_theta and
                point_theta < end_theta)
