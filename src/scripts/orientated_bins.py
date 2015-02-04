import itertools
import math
import numpy as np

import skimage.io as io
from skimage.util import view_as_windows
from scipy.ndimage.filters import generic_filter


class OrientatedBins(object):

    def __init__(self, radius, nbins=8):
        self._nbins = nbins
        self._radius = radius

        self._theta_offset = math.pi / self._nbins
        self._theta_step = math.pi / float(self._nbins/2)
        self._orientated_bins = []

    def compute_sectors(self, window_size):
        sectors = []
        filter_shape = (window_size, window_size)
        for i in range(self._nbins):
            segment = np.zeros(shape=filter_shape, dtype="int64")

            self._start_theta = self._theta_offset + i * self._theta_step
            self._end_theta = (self._start_theta + self._theta_step)

            self._start_theta = self._start_theta % (2*math.pi)
            self._end_theta = self._end_theta % (2*math.pi)

            for point in self._pixel_coordinates:
                if point == self._centre_point:
                    continue

                x,y = point
                centre_x, centre_y = self._centre_point

                offset_x = x - centre_x
                offset_y = y - centre_y

                polar_point = self._to_polar_coordinates(offset_x, offset_y)
                if self._in_segment_bounding_box(polar_point):
                    segment[x,y] = 1

            sectors.append(segment)

        sector_pairs = zip(sectors[:self._nbins/2], sectors[self._nbins/2:])
        for left, right in sector_pairs:
            orientated_bin = np.zeros(shape=filter_shape, dtype="int64")
            orientated_bin[np.where(left)] = 1
            orientated_bin[np.where(right)] = 1
            self._orientated_bins.append(orientated_bin)

        self._neighbourhood = np.zeros(shape=filter_shape, dtype="int64")
        for obin in self._orientated_bins:
            self._neighbourhood[np.where(obin)] = 1

    def extract_features(self, img, window_size, step_size, threshold=0):
        self._centre_point = (window_size/2, window_size/2)
        self._pixel_coordinates = list(itertools.product(range(window_size),
                                                         repeat=2))
        self.compute_sectors(window_size)

        line_strength_image = np.zeros(shape=img.shape)
        obins = [obin.flatten() for obin in self._orientated_bins]
        args = (obins, self._neighbourhood.flatten())

        generic_filter(img, filter_func,
                       size=(self._radius, self._radius),
                       output=line_strength_image,
                       extra_arguments=args)

        return line_strength_image

    def _to_polar_coordinates(self, x, y):
        """Convert the 2D pixel coordinates to polar coordinates"""
        theta = math.atan2(y, x)
        if theta < 0: theta = theta + 2 * math.pi
        return math.hypot(x, y), theta

    def _in_segment_bounding_box(self, polar_point):
        """Check if a polar coordinates lies within the segment of a circle

        :param polar_point: tuple representing the point to check (r,phi)
        """

        point_radius, point_theta = polar_point

        if self._start_theta > self._end_theta:
            return (point_radius < self._radius
                    and (point_theta >= self._start_theta
                    or point_theta < self._end_theta))
        else:
            return (point_radius < self._radius
                    and point_theta >= self._start_theta
                    and point_theta < self._end_theta)

def filter_func(window, obins, neighbourhood):
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
    # line_strength = average_intensities[orientation] - neighbourhood_average
    line_strength = orientation

    return line_strength

if __name__ == "__main__":
    import skimage.io as io
    from skimage.transform import pyramid_gaussian
    from scipy.ndimage.filters import gaussian_filter

    path="../../data/p214-010-60001-cl.png"
    mask="../../data/f214-010-60001-cl_mask.png"

    img = io.imread(path)
    msk = io.imread(mask, as_grey=True)

    pyramid = pyramid_gaussian(img, downscale=4)
    pyramid.next()
    img = pyramid.next()

    img = img * msk

    orientated_bins = OrientatedBins(8)
    line_strength = orientated_bins.extract_features(img, 8, 1)
