""" Mammogram analysis from a reduced dataset

Usage:
  mia.py RESULTS --output-file=<output>
  mia.py (-h | --help)
  mia.py --version
Options:
  -h --help                   Show this screen.
  --version                   Show version.
  -o --output-file=<output>   File to output the results to.

This file produces a collection of commands for analysing the output of a
reduction of mammogram images.
"""
import functools
import logging
import pandas as pd
import matplotlib.pyplot as plt

from docopt import docopt
from sklearn import manifold, preprocessing
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MIA Analysis")


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

    tSNE = manifold.TSNE(learning_rate=400, perplexity=45, early_exaggeration=2.0, verbose=2)
    fit_output = tSNE.fit_transform(feature_matrix)
    fit_output = pd.DataFrame(fit_output)

    return fit_output


def plot_scatter_2d(data_frame):
    data_frame.plot(kind='scatter', x=0, y=1, c='class', cmap=plt.cm.Spectral)
    plt.show()


def plot_scattermatrix(data_frame):
    column_names = filter(lambda x: x != 'class', data_frame.columns.values)
    g = sns.pairplot(data_frame, hue="class", size=1.5, vars=column_names)
    g.add_legend()


def main():
    arguments = docopt(__doc__, version='0.5.0')
    results_file = arguments["RESULTS"]
    output_file = arguments['--output-file']
    feature_matrix = pd.DataFrame().from_csv(results_file)
    #feature_matrix = feature_matrix[['avg_radius', 'std_radius', 'class', 'min_radius', 'max_radius', 'blob_count']] 
    # plot_scattermatrix(feature_matrix)
    tsne_output = fit_tSNE(feature_matrix)
    plot_scatter_2d(tsne_output)

    tsne_output.to_csv(output_file)


if __name__ == "__main__":
    main()
