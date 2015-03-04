import click
import logging
import pandas as pd

from mia.reduction import run_reduction
from mia.analysis import run_analysis
from mia.plotting import (plot_scatter_2d, plot_scattermatrix,
                                plot_median_image_matrix)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mia")

LOG_LEVELS = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING
}


def print_version(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    click.echo('Version 0.5.0')
    ctx.exit()


@click.group()
@click.option('--version', is_flag=True, callback=print_version,
              expose_value=False, is_eager=True,
              help="Version of the application.")
@click.option('--log-level', default='info',
              type=click.Choice(['debug', 'info', 'warning']),
              help="Level of logging to report")
def cli(log_level):
    logger.setLevel(LOG_LEVELS[log_level])


@cli.command()
@click.argument('image-directory')
@click.argument('masks-directory')
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
@click.argument('csv-file')
@click.option('--output-file', '-o', default=None,
              help="Name of output file to store results of analysis in")
def analysis(csv_file, output_file):
    run_analysis(csv_file, output_file)


@cli.group()
def plotting():
    pass


@plotting.command()
@click.argument('csv-file')
@click.option('--label-column', '-l', default=None,
              help="Name of column to use as the class labels")
def scatter_plot(csv_file, label_column):
    df = pd.DataFrame.from_csv(csv_file)
    plot_scatter_2d(df, label_column)


@plotting.command()
@click.argument('csv-file')
@click.option('--label-column', '-l', default=None,
              help="Name of column to use as the class labels")
def scatter_matrix(csv_file, label_column):
    df = pd.DataFrame.from_csv(csv_file)
    plot_scattermatrix(df, label_column)


@plotting.command()
@click.argument('csv-file')
@click.argument('img_path')
@click.option('--label-column', '-l', default=None,
              help="Name of column to use as the class labels")
@click.option('--output_file', '-o', default=None,
              help="Name of the file to save the image to")
def median_image_matrix(csv_file, img_path, label_column, output_file):
    df = pd.DataFrame.from_csv(csv_file)
    plot_median_image_matrix(df, img_path, label_column,
                             output_file=output_file)


if __name__ == '__main__':
    cli(obj={})
