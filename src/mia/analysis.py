""" Analysis module provides general functions used for examining the results
of running feature detection on a dataset of images.
"""

import functools
import logging
import re
import random
import numpy as np
import pandas as pd

from sklearn import manifold, preprocessing
from features.blobs import blob_props
from features.linear_structure import line_props

logger = logging.getLogger(__name__)


def _handle_data_frame(func):
    """Decorator to wrap the scikit-learn funcions so that they can handle
    being directly passed a pandas DataFrame.
    """
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
def isomap(feature_matrix, **kwargs):
    """Run the Isomap algorithm on the feature space of a collection of images

    :param feature_matrix: matrix of features use with the Isomap
    :returns: 2darray -- lower dimensional mapping of the Isomap algorithm
    """
    feature_matrix = standard_scaler(feature_matrix)
    isomap = manifold.Isomap(**kwargs)
    fit_output = isomap.fit_transform(feature_matrix)
    return fit_output


@_handle_data_frame
def lle(feature_matrix, **kwargs):
    """Run the Locally Linear Embedding algorithm on the feature space of a
    collection of images

    :param feature_matrix: matrix of features use with the Isomap
    :returns: 2darray -- lower dimensional mapping of the Isomap algorithm
    """
    feature_matrix = standard_scaler(feature_matrix)
    lle = manifold.LocallyLinearEmbedding(**kwargs)
    fit_output = lle.fit_transform(feature_matrix)
    return fit_output


@_handle_data_frame
def standard_scaler(feature_matrix):
    """Run the standard scaler algorithm. This removes the mean and scales to
    unit variance.

    :param feature_matrix: matrix of features to run standard scaler on.
    :returns: 2darray -- data transformed using the standard scaler
    """
    scalar = preprocessing.StandardScaler()
    feature_matrix = scalar.fit_transform(feature_matrix)
    return feature_matrix


@_handle_data_frame
def normalize_data_frame(feature_matrix):
    """Run the normalisation function. This scales all values to between 0 and 1

    :param feature_matrix: matrix of features to perform normalisation on.
    :returns: 2darray -- normalised data
    """
    return preprocessing.normalize(feature_matrix)


def measure_closeness(data_frame, labels):
    """This function computes the average distance between all data points in a
    group and the centroid of that group.

    The neighbours are defiend as being all points with the same
    class. The class for each point is defined by the labels parameter.

    :param data_frame: data frame points to measure the distance between.
    :param labels: labels for each data point. Each point with the same class
        are neighbours of points.
    :returns: Series -- containing the closeness measure for each group
    """
    groups = data_frame.groupby(labels)
    ds = [_cluster_measure(frame) for index, frame in groups]
    return pd.Series(ds, index=labels.unique())


def _cluster_measure(group):
    """Measure the distance between all points in a group from the centroid.

    :param group: the group of points all belonging to the same cluster.
    :returns: float -- mean value of the group.
    """
    points = group[[0, 1]]
    centroid = points.sum() / group.size
    distances = ((centroid - points)**2).sum(axis=1)
    distances = distances.apply(np.sqrt)
    return distances.mean()


def create_hologic_meta_data(df, meta_data_file):
    """Generate a data frame containing the meta data for the Hologic dataset.

    This uses the existing data frame of images for the index names.

    :param df: the data frame of features detected from a reduction.
    :param meta_data_file: name of the file containin the meta data for the
        dataset
    :returns: DataFrame -- containing the meta data for the Hologic data set
    """
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
    """Split the Hologic naming convention into several parts.

    This function will parse the image name to extract the patient id, side
    (left or right), and the view (CC or ML).

    :param name: file name to parse
    :returns: tuple -- containing the (id, view, side)
    """
    img_regex = re.compile(r'p(\d{3}-\d{3}-\d{5})-([a-z])([a-z])\.png')
    m = re.match(img_regex, name)
    return int(m.group(1).replace('-', '')), m.group(2), m.group(3)


def create_synthetic_meta_data(df, meta_data_file):
    """Create a meta data DataFrame for the synthetic data set.

    This uses the existing data frame of images for the index names.

    :param df: the data frame of features detected from a reduction.
    :param meta_data_file: name of the file containin the meta data for the
        dataset
    :returns: DataFrame -- containing the meta data for the synthetic data set
    """
    indicies = [_split_sythentic_img_name(img_name)
                for img_name in df.index.values]
    raw_md = pd.DataFrame.from_csv(meta_data_file)
    md = raw_md.loc[indicies].copy()
    md['phantom_name'] = md.index
    md.index = df.index
    return md


def _split_sythentic_img_name(name):
    """Split the synthetic naming convention into several parts.

    This function will parse the image name to extract the group of the
    synthetic.

    :param name: file name to parse
    :returns: string -- the group of the synthetic.
    """
    group_regex = re.compile(r'(test_Mix_DPerc\d+_c)_\d+\.dcm')
    img_regex = re.compile(r'(phMix\d+)_c_\d+\.dcm')

    group_match = re.match(group_regex, name)
    img_match = re.match(img_regex, name)
    if group_match:
        return group_match.group(1)
    elif img_match:
        return img_match.group(1)


def features_from_blobs(df):
    """ Create features from blobs detected over a image dataset

    :param df: DataFrame containing the raw detections.
    :returns: DataFrame with the descriptive statistics generated
        from the blobs.
    """
    features = df.groupby(df.index).apply(blob_props)
    return features.reset_index(level=1, drop=True)


def features_from_lines(df):
    """ Create features from lines detected over a image dataset

    :param df: DataFrame containing the raw detections.
    :returns: DataFrame with the descriptive statistics generated
        from the lines.
    """
    features = df.groupby(df.index).apply(line_props)
    return features.reset_index(level=1, drop=True)


def remove_duplicate_index(df):
    """Remove all entries in a data frame that have a duplicate index.

    :param df: DataFrame containing duplicate indicies.
    :returns: DataFrame with the duplicates removed
    """
    index_name = df.index.name
    md = df.reset_index()
    md.drop_duplicates(index_name, inplace=True)
    md.set_index(index_name, inplace=True)
    return md


def create_random_subset(data_frame, column_name):
    """Choose a random subset from a DataFrame.

    The column name determines which groups are sampled. Only one image from
    each group is returned.

    :param df: DataFrame of multiple images to sample from.
    :param column_name: column defining the group an image belongs to.
    :returns: DataFrame -- DataFrame with one image from each group.
    """
    def _select_random(x):
        return x.ix[random.sample(x.index, 1)]

    group = data_frame.groupby(column_name)
    random_subset = group.apply(_select_random)
    random_subset.drop(column_name, axis=1, inplace=True)
    random_subset.reset_index(drop=True, level=0, inplace=True)
    return random_subset


def group_by_scale_space(data_frame):
    """Group an image by scale space.

    This function takes a data frame of features detect from blobs and groups
    each one by the scale it was detected at.

    :param data_frame: containing features derived from each blob.
    :returns: DataFrame -- DataFrame with features grouped by scale.
    """
    radius_groups = data_frame.groupby([data_frame.radius, data_frame.index])
    blobs_by_scale = radius_groups.apply(lambda x: x.mean())

    scale_groups = blobs_by_scale.groupby(level=0)
    features = pd.DataFrame(index=blobs_by_scale.index.levels[1])

    for scale_num, (i, x) in enumerate(scale_groups):
        x = x.reset_index(level=0, drop=True)
        x = x.drop(['radius'], axis=1)
        features = features.join(x, rsuffix='_%d' % scale_num)

    features.fillna(features.mean(), inplace=True)
    return features


def sort_by_scale_space(data_frame, n_features):
    """Sort the features in a matrix into the correct order.

    This can be used to sort features so that each feature appears in scale
    order. (i.e. count_0, count_1, ... count_n)

    :param data_frame: containing features for each image.
    :returns: DataFrame -- DataFrame with features sorted by group and scale.
    """
    features = data_frame.copy()
    features = normalize_data_frame(features)
    features.columns = data_frame.columns

    cols = [features.columns[i::n_features] for i in range(n_features)]
    cols = [c for l in cols for c in l]
    features = features[cols]
    return features
