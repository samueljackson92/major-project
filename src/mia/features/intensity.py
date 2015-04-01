import pandas as pd
from mia.features.blobs import extract_blob


def detect_intensity(img, blobs):
    def _extract_intensity(blob):
        return pd.concat([pd.DataFrame([blob], columns=['x', 'y', 'radius']),
                          intensity_props(extract_blob(blob, img))], axis=1)

    frames = map(_extract_intensity, blobs)
    return pd.concat(frames)


def intensity_props(img):
    # create dataframe of histogram features from the described series
    img_series = pd.Series(img.flatten())
    img_described = img_series.describe()
    stats = pd.DataFrame(img_described.as_matrix()).T
    stats.columns = img_described.index

    stats['skew'] = img_series.skew()
    stats['kurtosis'] = img_series.kurtosis()
    return stats
