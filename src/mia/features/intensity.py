import pandas as pd
import numpy as np

from mia.features.blobs import extract_blob
from mia.features.linear_structure import extract_line


def detect_intensity(img, patches):

    def _extract_intensity(row):
        _, patch = row
        if 'area' in patch:
            img_patch = extract_line(patch, img)
        else:
            img_patch = extract_blob(patch, img)
        return intensity_props(img_patch)

    frames = map(_extract_intensity, patches.iterrows())
    features = pd.concat(frames)
    features.index = patches.index
    return pd.concat([patches, features], axis=1)


def intensity_props(img):
    # create dataframe of histogram features from the described series
    img_series = pd.Series(img.flatten())

    img_described = img_series.describe()
    stats = pd.DataFrame(img_described.as_matrix()).T
    stats.columns = img_described.index
    stats['skew'] = img_series.skew()

    if img_series.size > 1:
        stats['kurtosis'] = img_series.kurtosis()
    else:
        stats['kurtosis'] = np.NaN

    return stats
