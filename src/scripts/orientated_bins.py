import skimage.io as io
from scipy.ndimage.filters import gaussian_filter
import itertools
import math
from math import hypot, atan2, pi
import numpy as np

# class OrientatedBins(object):
#
#     def __init__(self, radius, nbins=8):
#         pass

def to_polar_coordinates(x, y):
    theta = math.atan2(y, x)
    if theta < 0: theta = theta + 2 * math.pi
    return math.hypot(x, y), theta

def in_segment_bounding_box(polar_point, centre_radius, start_theta, end_theta):
    point_radius, point_theta = polar_point

    if start_theta > end_theta:
        return (point_radius < centre_radius
                and (point_theta >= start_theta
                or point_theta < end_theta))
    else:
        return (point_radius < centre_radius
                and point_theta >= start_theta
                and point_theta < end_theta)


linear_structure = np.identity(20)
img = gaussian_filter(linear_structure, 4)

centre_radius = 10
bins = 16
offset = math.pi/bins
theta_step = np.pi / float(bins/2)
window_size = linear_structure.shape[0]
centre = (window_size / 2, window_size / 2)
centre_x, centre_y = centre

segements = []
coordinates = list(itertools.product(range(window_size), repeat=2))
for i in range(bins):
    segment = np.zeros(shape=linear_structure.shape, dtype="int64")

    start_theta = offset + i * theta_step
    end_theta = (start_theta + theta_step)

    start_theta = start_theta % (2*math.pi)
    end_theta = end_theta % (2*math.pi)

    for point in coordinates:
        x,y = point

        #offset the location by the centre point
        offset_x = x - centre_x
        offset_y = y - centre_y

        polar_point = to_polar_coordinates(offset_x, offset_y)
        if in_segment_bounding_box(polar_point, centre_radius,
                                   start_theta, end_theta):
            segment[x,y] = 1

    segements.append(segment)
    io.imshow(segment)
    io.show()

relative_intensities = np.array([np.average(img[np.where(segement)]) for segement in segements])

import matplotlib.pyplot as plt
plt.bar(range(len(relative_intensities)), relative_intensities)
plt.show()

print relative_intensities
line_strength = np.max(relative_intensities) - np.average(relative_intensities)
print line_strength
