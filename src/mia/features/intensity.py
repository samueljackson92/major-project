import numpy as np
import pandas as pd
import scipy.stats
from skimage import exposure

from mia.features.blobs import extract_blob


def detect_intensity(blobs, img):
    column_names = ['avg_intensity', 'std_intensity',
                    'skew_intensity', 'kurtosis_intensity']
    props = np.array([_find_intensity_props(blob, img)
                      for blob in blobs[['x', 'y', 'radius']].as_matrix()])
    return pd.DataFrame(props, columns=column_names)


def _intensity_stats(img):
    hist, bins = exposure.histogram(img)
    hist = np.array(hist)
    stats = np.array([hist.mean(), hist.std(),
                      scipy.stats.skew(hist),
                      scipy.stats.kurtosis(hist)])
    return stats


def _find_intensity_props(blob, img):
    img_section = extract_blob(blob, img)
    return _intensity_stats(img_section)


def intensity_features(blobs, img):
    column_names = ['avg_avg_intensity', 'avg_std_intensity',
                    'avg_skew_intensity', 'avg_kurt_intensity',
                    'std_avg_intensity', 'std_std_intensity',
                    'std_skew_intensity', 'std_kurt_intensity']

    return
