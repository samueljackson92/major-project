import functools
import logging
import pandas as pd

from sklearn import manifold, preprocessing

logger = logging.getLogger(__name__)


def handle_data_frame(func):
    @functools.wraps(func)
    def inner(data_frame):
        df = data_frame.drop('class', axis=1)

        fit_output = func(df.as_matrix())

        fit_output.index = data_frame.index
        fit_output['class'] = pd.Series(data_frame['class'],
                                        index=fit_output.index)
        return fit_output
    return inner


@handle_data_frame
def fit_tSNE(feature_matrix):
    scalar = preprocessing.StandardScaler()
    feature_matrix = scalar.fit_transform(feature_matrix)

    tSNE = manifold.TSNE(learning_rate=400, perplexity=45,
                         early_exaggeration=2.0, verbose=2)
    fit_output = tSNE.fit_transform(feature_matrix)
    fit_output = pd.DataFrame(fit_output)

    return fit_output


def run_analysis(csv_file, output_file=None):
    feature_matrix = pd.DataFrame.from_csv(csv_file)
    # feature_matrix = feature_matrix[['avg_radius', 'std_radius', 'class',
    #                                'min_radius', 'max_radius', 'blob_count']]
    fit_output = fit_tSNE(feature_matrix)

    if output_file is not None:
        fit_output.to_csv(output_file)
    else:
        logger.info(fit_output)
