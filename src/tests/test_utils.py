import numpy as np
import skimage.filter as filters

def assert_lists_equal(a,b):
    return len(a) == len(b) and sorted(a) == sorted(b)

def generate_linear_structure(size, with_noise=False):
    """Generate a basic linear structure, optionally with noise"""
    linear_structure = np.zeros(shape=(size,size))
    linear_structure[:,size/2] = np.ones(size)

    if with_noise:
        linear_structure = np.identity(size)
        noise = np.random.rand(size, size) * 0.1
        linear_structure += noise
        linear_structure = filters.gaussian_filter(linear_structure, 1.5)

    return linear_structure


def generate_blob():
    """ Generate a blob by drawing from the normal distribution across two axes
    and binning it to the required size
    """
    mean = [0,0]
    cov = [[1,0],[0,1]] # diagonal covariance, points lie on x or y-axis
    x,y = np.random.multivariate_normal(mean,cov,50000).T
    h, xedges, yedges = np.histogram2d(x,y, bins=100)
    return h
