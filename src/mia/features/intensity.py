import pandas as pd
from mia.features.blobs import extract_blob


def detect_intensity(img, blobs):

    intensity_features = []
    for blob in blobs:
        img_blob = extract_blob(blob, img)
        stats = intensity_props(img_blob)
        intensity_features.append(stats)

    return pd.concat(intensity_features)


def intensity_props(img):
    # create dataframe of histogram features from the described series
    img_series = pd.Series(img.flatten())
    img_described = img_series.describe()
    stats = pd.DataFrame(img_described.as_matrix()).T
    stats.columns = img_described.index

    stats['skew'] = img_series.skew()
    stats['kurtosis'] = img_series.kurtosis()
    return stats
