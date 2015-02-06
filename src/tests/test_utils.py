import numpy as np
import skimage.filter as filters

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
