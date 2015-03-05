""" Analysis module for examining the results of running feature detection on
a dataset of images.
"""

import functools
import logging
import numpy as np
import pandas as pd
import re

from sklearn import manifold, preprocessing

logger = logging.getLogger(__name__)


def _handle_data_frame(func):
    @functools.wraps(func)
    def inner(data_frame):
        info_columns = ['image_name', 'patient_id', 'view', 'side', 'class']

        df = data_frame.drop(info_columns, axis=1)
        fit_output = func(df.as_matrix())

        fit_output.index = data_frame.index
        image_info = data_frame[info_columns]
        fit_output = pd.concat([fit_output, image_info], axis=1)

        return fit_output
    return inner


@_handle_data_frame
def fit_tSNE(feature_matrix):
    """Run the t-SNE algorithm on the feature space of a collection of images

    :param feature_matrix: matrix of features use with the t-SNE
    :returns: 2darray -- lower dimensional mapping of the t-SNE algorithm
    """
    scalar = preprocessing.StandardScaler()
    feature_matrix = scalar.fit_transform(feature_matrix)

    tSNE = manifold.TSNE(learning_rate=400, perplexity=45,
                         early_exaggeration=2.0, verbose=2)
    fit_output = tSNE.fit_transform(feature_matrix)
    fit_output = pd.DataFrame(fit_output)

    return fit_output


def run_analysis(csv_file, output_file=None):
    """Run an analysis algorithm on the results of feature detection

    :param csv_file: file to load the dataset from
    :param output_file: file to output the results of the lower dimensional
                        mapping to
    """
    feature_matrix = pd.DataFrame.from_csv(csv_file)
    # feature_matrix = feature_matrix[['avg_radius', 'std_radius', 'class',
    #                                'min_radius', 'max_radius', 'blob_count']]
    fit_output = fit_tSNE(feature_matrix)

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

    ds.plot('hist')
    plt.show()


def _cluster_measure(group):
    points = group[['0', '1']]
    centroid = points.sum() / group.size
    distances = ((centroid - points)**2).sum(axis=1)
    distances = distances.apply(np.sqrt)
    return distances.mean()
