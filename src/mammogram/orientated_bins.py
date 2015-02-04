import itertools
import math
import numpy as np

import skimage.io as io
from skimage.util import view_as_windows
from scipy.ndimage.filters import generic_filter


def orientated_bins(img, radius, nbins=8, feature_type='strength'):
    window_size = (radius, radius)
    orientated_bins, neighbourhood = _compute_sectors(radius, nbins=8)
    return _extract_features(img, window_size, orientated_bins, neighbourhood, feature_type)

def _compute_sectors(radius, nbins):
    nbins = nbins
    radius = radius

    theta_offset = math.pi / nbins
    theta_step = math.pi / float(nbins/2)
    orientated_bins = []

    sectors = []
    centre_point = (radius/2, radius/2)
    window_coordinates = list(itertools.product(range(radius), repeat=2))
    filter_shape = (radius, radius)

    for i in range(nbins):
        segment = np.zeros(shape=filter_shape, dtype="int64")

        start_theta = theta_offset + i * theta_step
        end_theta = (start_theta + theta_step)

        start_theta = start_theta % (2*math.pi)
        end_theta = end_theta % (2*math.pi)

        for point in window_coordinates:
            if point == centre_point:
                continue

            x,y = point
            centre_x, centre_y = centre_point

            offset_x = x - centre_x
            offset_y = y - centre_y

            polar_point = _to_polar_coordinates(offset_x, offset_y)
            if _in_segment_bounding_box(polar_point, radius,
                                        start_theta, end_theta):
                segment[x,y] = 1

        sectors.append(segment)

    sector_pairs = zip(sectors[:nbins/2], sectors[nbins/2:])
    for left, right in sector_pairs:
        orientated_bin = np.zeros(shape=filter_shape, dtype="int64")
        orientated_bin[np.where(left)] = 1
        orientated_bin[np.where(right)] = 1
        orientated_bins.append(orientated_bin)

    neighbourhood = np.zeros(shape=filter_shape, dtype="int64")
    for obin in orientated_bins:
        neighbourhood[np.where(obin)] = 1

    return orientated_bins, neighbourhood

def _extract_features(img, window_size, orientated_bins, neighbourhood, feature_type):
    response_image = np.zeros(shape=img.shape)
    obins = [np.where(obin.flatten()) for obin in orientated_bins]
    args = (obins, np.where(neighbourhood.flatten()))

    if feature_type == 'strength':
        generic_filter(img, line_strength_filter,
                       size=window_size,
                       output=response_image,
                       extra_arguments=args)
    elif feature_type == 'orientation':
        generic_filter(img, line_orientation_filter,
                       size=window_size,
                       output=response_image,
                       extra_arguments=args)

    return response_image

def _to_polar_coordinates(x, y):
    """Convert the 2D pixel coordinates to polar coordinates"""
    theta = math.atan2(y, x)
    if theta < 0: theta = theta + 2 * math.pi
    return math.hypot(x, y), theta

def _in_segment_bounding_box(polar_point, radius, start_theta, end_theta):
    """Check if a polar coordinates lies within the segment of a circle

    :param polar_point: tuple representing the point to check (r,phi)
    """

    point_radius, point_theta = polar_point

    if start_theta > end_theta:
        return (point_radius < radius
                and (point_theta >= start_theta
                or point_theta < end_theta))
    else:
        return (point_radius < radius
                and point_theta >= start_theta
                and point_theta < end_theta)

def line_strength_filter(window, obins, neighbourhood):
    if np.sum(window) == 0:
        return 0

    average_intensities = []
    for obin in obins:
        subset = window[obin]
        mean = np.sum(subset) / subset.size
        average_intensities.append(mean)

    average_intensities = np.array(average_intensities)

    neighbour_subset = window[neighbourhood]
    neighbourhood_average = np.sum(neighbour_subset) / neighbour_subset.size

    orientation = np.argmax(average_intensities)
    line_strength = average_intensities[orientation] - neighbourhood_average

    return line_strength

def line_orientation_filter(window, obins, neighbourhood):
    if np.sum(window) == 0:
        return 0

    average_intensities = []
    for obin in obins:
        subset = window[obin]
        mean = np.sum(subset) / subset.size
        average_intensities.append(mean)

    average_intensities = np.array(average_intensities)

    neighbour_subset = window[neighbourhood]
    neighbourhood_average = np.sum(neighbour_subset) / neighbour_subset.size

    orientation = np.argmax(average_intensities)

    return orientation
