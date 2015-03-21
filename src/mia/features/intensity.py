import numpy as np
import pandas as pd
from mia.features.blobs import extract_blob


def detect_intensity(blobs, img):

    intensity_features = []
    for scale, (index, frame) in enumerate(blobs.groupby('radius')):
        rois = [extract_blob(blob, img).flatten()
                for blob in frame[['x', 'y', 'radius']].as_matrix()]
        rois = np.concatenate(rois)
        features = pd.Series(rois).describe()
        intensity_features.append(features)

    return pd.DataFrame(intensity_features)


def intensity_props(feature_set):
    column_names = ['avg_avg_intensity', 'avg_std_intensity',
                    'avg_skew_intensity', 'avg_kurt_intensity',
                    'std_avg_intensity', 'std_std_intensity',
                    'std_skew_intensity', 'std_kurt_intensity']

    # intensity statistics
    intensity_stats = feature_set[['avg_intensity', 'std_intensity',
                                   'skew_intensity', 'kurtosis_intensity']]

    avg_intensity_stats = intensity_stats.mean(axis=0).as_matrix()
    std_intensity_stats = intensity_stats.std(axis=0).as_matrix()

    props = np.concatenate([avg_intensity_stats, std_intensity_stats])
    return pd.DataFrame([props], columns=column_names)
