import click
import logging
import pandas as pd
import skimage.io as io

from mia.reduction import (run_reduction, reduction_feature_statistics,
                           process_image)
from mia.analysis import run_analysis, measure_closeness
from mia.plotting import (plot_scatter_2d, plot_scattermatrix,
                          plot_median_image_matrix, plot_blobs)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mia")

LOG_LEVELS = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING
}


@click.group()
@click.version_option(version='0.6.0', prog_name='mia')
@click.option('--log-level', default='info',
              type=click.Choice(['debug', 'info', 'warning']),
              help="Level of logging to report")
def cli(log_level):
    logger.setLevel(LOG_LEVELS[log_level])


@cli.command()
@click.argument('image-directory', type=click.Path())
@click.argument('masks-directory', type=click.Path())
@click.option('--output-file', '-o', default=None,
              help="Name of csv file to write the results of the reduction to")
@click.option('--BIRADS-file', '-b', default=None,
              help="Name of csv file containing BI-RADS classes from dataset")
@click.option('--num-processes', default=2,
              help="Num of processes to use for the reduction.")
def reduction(image_directory, masks_directory, output_file, birads_file,
              num_processes):
    run_reduction(image_directory, masks_directory, output_file, birads_file,
                  num_processes)


@cli.command()
@click.argument('csv-file', type=click.Path())
@click.argument('output-file', type=click.Path())
def feature_statistics(csv_file, output_file):
    reduction_feature_statistics(csv_file, output_file)


@cli.command()
@click.argument('image-file', type=click.Path())
@click.argument('mask-file', type=click.Path())
def detect_blobs(image_file, mask_file):
    data_frame = process_image(image_file, mask_file)
    img = io.imread(image_file, as_grey=True)
    plot_blobs(img, data_frame[['x', 'y', 'radius']].as_matrix())


@cli.group()
def analysis():
    pass


@analysis.command()
@click.argument('csv-file', type=click.Path())
@click.argument('columns', required=False, nargs=-1)
@click.option('--filter-column', '-c', default=None)
@click.option('--filter-value', '-v', default=None)
@click.option('--output-file', '-o', default=None,
              help="Name of output file to store results of analysis in")
def tSNE(csv_file, columns, filter_column, filter_value, output_file):
    run_analysis(csv_file, filter_column, filter_value,
                 list(columns), output_file)


@analysis.command()
@click.argument('csv-file', type=click.Path())
@click.argument('label')
def closeness(csv_file, label):
    measure_closeness(csv_file, label)


@cli.group()
def plotting():
    pass


@plotting.command()
@click.argument('csv-file', type=click.Path())
@click.option('--label-column', '-l', default=None,
              help="Name of column to use as the class labels")
@click.option('--annotate', is_flag=True,
              help="Annotate the images with image names")
def scatter_plot(csv_file, label_column, annotate):
    df = pd.DataFrame.from_csv(csv_file)
    plot_scatter_2d(df, label_column, annotate)


@plotting.command()
@click.argument('csv-file', type=click.Path())
@click.option('--label-column', '-l', default=None,
              help="Name of column to use as the class labels")
def scatter_matrix(csv_file, label_column):
    df = pd.DataFrame.from_csv(csv_file)
    plot_scattermatrix(df, label_column)


@plotting.command()
@click.argument('csv-file', type=click.Path())
@click.argument('img_path', type=click.Path())
@click.argument('output_file', type=click.Path())
@click.option('--label-column', '-l', default=None,
              help="Name of column to use as the class labels")
@click.option('--features-csv', '-f', default=None,
              help="Name of the file containg features")
def median_image_matrix(csv_file, img_path, output_file, label_column,
                        features_csv):
    df = pd.DataFrame.from_csv(csv_file)
    plot_median_image_matrix(df, img_path, label_column,
                             output_file=output_file,
                             raw_features_csv=features_csv)


if __name__ == '__main__':
    cli(obj={})
