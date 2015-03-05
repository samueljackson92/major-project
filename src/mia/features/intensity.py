import numpy as np
import scipy.stats
from skimage import exposure

from mia.features.blobs import extract_blob


def intensity_features(img):
    hist, bins = exposure.histogram(img)
    hist = np.array(hist)
    stats = np.array([hist.mean(), hist.std(),
                      scipy.stats.skew(hist),
                      scipy.stats.kurtosis(hist)])
    return stats


def blob_intensity_props(blob, img):
    img_section = extract_blob(blob, img)
    return intensity_features(img_section)
