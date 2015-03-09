""" Analysis module for examining the results of running feature detection on
a dataset of images.
"""

import functools
import logging
import numpy as np
import pandas as pd

from sklearn import manifold, preprocessing

logger = logging.getLogger(__name__)


def _handle_data_frame(func):
    @functools.wraps(func)
    def inner(data_frame, columns):
        meta_columns = ['patient_id', 'view', 'side', 'class']
        df = _remove_meta_data(data_frame, meta_columns)
        df = df[columns]

        logger.info("Using columns: %s" % ', '.join(df.columns))
        fit_output = func(df.as_matrix())
        fit_output = pd.DataFrame(fit_output)

        fit_output = _copy_meta_data(fit_output, data_frame, meta_columns)
        return fit_output
    return inner


def _remove_meta_data(data_frame, meta_columns):
    meta_columns = filter(lambda x: x in data_frame.columns, meta_columns)
    return data_frame.drop(meta_columns, axis=1)


def _copy_meta_data(new_df, data_frame, meta_columns):
    meta_columns = filter(lambda x: x in data_frame.columns, meta_columns)
    new_df.index = data_frame.index
    image_info = data_frame[meta_columns]
    return pd.concat([new_df, image_info], axis=1)


@_handle_data_frame
def fit_tSNE(feature_matrix):
    """Run the t-SNE algorithm on the feature space of a collection of images

    :param feature_matrix: matrix of features use with the t-SNE
    :returns: 2darray -- lower dimensional mapping of the t-SNE algorithm
    """
    # feature_matrix = preprocessing.normalize(feature_matrix)
    scalar = preprocessing.StandardScaler()
    feature_matrix = scalar.fit_transform(feature_matrix)

    tSNE = manifold.TSNE(learning_rate=400, perplexity=45,
                         early_exaggeration=2.0, verbose=2)
    fit_output = tSNE.fit_transform(feature_matrix)
    return fit_output


def run_analysis(csv_file, filter_column=None, filter_value=None,
                 columns=None, output_file=None):
    """Run an analysis algorithm on the results of feature detection

    :param csv_file: file to load the dataset from
    :param output_file: file to output the results of the lower dimensional
                        mapping to
    """
    feature_matrix = pd.DataFrame.from_csv(csv_file)
    if filter_column is not None and filter_value is not None:
        condition = feature_matrix[filter_column] == filter_value
        feature_matrix = feature_matrix[condition]

    fit_output = fit_tSNE(feature_matrix, columns)

    if output_file is not None:
        fit_output.to_csv(output_file)
    else:
        logger.info(fit_output)


def measure_closeness(csv_file, column_name):
    import matplotlib.pyplot as plt
    df = pd.DataFrame.from_csv(csv_file)

    ds = [_cluster_measure(frame) for index, frame in df.groupby(column_name)]
    ds = pd.Series(ds, index=df[column_name].unique())

    print ds.describe()

    ds.plot('bar')
    plt.show()


def _cluster_measure(group):
    points = group[['0', '1']]
    centroid = points.sum() / group.size
    distances = ((centroid - points)**2).sum(axis=1)
    distances = distances.apply(np.sqrt)
    return distances.mean()
