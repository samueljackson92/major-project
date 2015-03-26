""" Analysis module for examining the results of running feature detection on
a dataset of images.
"""

import functools
import logging
import re
import numpy as np
import pandas as pd

from sklearn import manifold, preprocessing
from mia.features.blobs import blob_props
from mia.features.intensity import intensity_props

logger = logging.getLogger(__name__)


def _handle_data_frame(func):
    @functools.wraps(func)
    def inner(feature_matrix, **kwargs):
        if isinstance(feature_matrix, pd.DataFrame):
            fit_output = func(feature_matrix.as_matrix(), **kwargs)
            return pd.DataFrame(fit_output, index=feature_matrix.index)
        else:
            return func(feature_matrix)
    return inner


@_handle_data_frame
def tSNE(feature_matrix, **kwargs):
    """Run the t-SNE algorithm on the feature space of a collection of images

    :param feature_matrix: matrix of features use with the t-SNE
    :returns: 2darray -- lower dimensional mapping of the t-SNE algorithm
    """
    feature_matrix = standard_scaler(feature_matrix)
    tSNE = manifold.TSNE(**kwargs)
    fit_output = tSNE.fit_transform(feature_matrix)
    return fit_output


@_handle_data_frame
def standard_scaler(feature_matrix):
    scalar = preprocessing.StandardScaler()
    feature_matrix = scalar.fit_transform(feature_matrix)
    return feature_matrix


@_handle_data_frame
def normalize_data_frame(feature_matrix):
    return preprocessing.normalize(feature_matrix)


def measure_closeness(data_frame, labels):
    groups = data_frame.groupby(labels)
    ds = [_cluster_measure(frame) for index, frame in groups]
    return pd.Series(ds, index=labels.unique())


def _cluster_measure(group):
    points = group[[0, 1]]
    centroid = points.sum() / group.size
    distances = ((centroid - points)**2).sum(axis=1)
    distances = distances.apply(np.sqrt)
    return distances.mean()


def create_hologic_meta_data(df, meta_data_file):
    data = [_split_hologic_img_name(img_name) for img_name in df.index.values]
    columns = ['patient_id', 'side', 'view']

    name_meta_data = pd.DataFrame(data, columns=columns)
    name_meta_data.index = name_meta_data.patient_id
    name_meta_data['img_name'] = df.index.values

    BIRADS_data = pd.DataFrame.from_csv(meta_data_file)
    meta_data = name_meta_data.join(BIRADS_data, how='left', rsuffix='_r')
    meta_data.drop('patient_id_r', axis=1, inplace=True)
    meta_data.index = df.index

    return meta_data


def _split_hologic_img_name(name):
    img_regex = re.compile(r'p(\d{3}-\d{3}-\d{5})-([a-z])([a-z])\.png')
    m = re.match(img_regex, name)
    return int(m.group(1).replace('-', '')), m.group(2), m.group(3)


def create_synthetic_meta_data(df, meta_data_file):
    indicies = [_split_sythentic_img_name(img_name)
                for img_name in df.index.values]
    raw_md = pd.DataFrame.from_csv(meta_data_file)
    md = raw_md.loc[indicies].copy()
    md['phantom_name'] = md.index
    md.index = df.index
    return md


def _split_sythentic_img_name(name):
    group_regex = re.compile(r'(test_Mix_DPerc\d+_c)_\d+\.dcm')
    img_regex = re.compile(r'(phMix\d+)_c_\d+\.dcm')

    group_match = re.match(group_regex, name)
    img_match = re.match(img_regex, name)
    if group_match:
        return group_match.group(1)
    elif img_match:
        return img_match.group(1)


def features_from_blobs(df):
    features = df.groupby(df.index).apply(blob_props)
    return features.reset_index(level=1, drop=True)


def features_from_intensity(df):
    features = df.groupby(df.index).apply(intensity_props)
    return features.reset_index(level=1, drop=True)


def remove_duplicate_index(df):
    index_name = df.index.name
    md = df.reset_index()
    md.drop_duplicates(index_name, inplace=True)
    md.set_index(index_name, inplace=True)
    return md
