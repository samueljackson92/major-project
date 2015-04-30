import click
import logging
import pandas as pd

from skimage import transform
import mia

from mia.reduction.reducers import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mia")

LOG_LEVELS = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING
}


@click.group()
@click.version_option(version='0.9.0', prog_name='mia')
@click.option('--log-level', default='info',
              type=click.Choice(['debug', 'info', 'warning']),
              help="Level of logging to report")
def cli(log_level):
    logger.setLevel(LOG_LEVELS[log_level])


###############################################################################
# MIA Reduction
###############################################################################


@cli.group()
def reduction():
    pass


@reduction.command()
@click.argument('image-directory', type=click.Path())
@click.argument('masks-directory', type=click.Path())
@click.argument('output-file', type=click.Path())
@click.option('--num-processes', default=2,
              help="Num of processes to use for the reduction.")
def blob(image_directory, masks_directory, output_file, num_processes):
    reduction = BlobFeaturesReduction(image_directory, masks_directory)
    features = reduction.reduce(num_processes=num_processes)
    features.to_csv(output_file)


@reduction.command()
@click.argument('image-directory', type=click.Path())
@click.argument('masks-directory', type=click.Path())
@click.argument('output-file', type=click.Path())
@click.option('--num-processes', default=2,
              help="Num of processes to use for the reduction.")
def line(image_directory, masks_directory, output_file, num_processes):
    reduction = LineFeaturesReduction(image_directory, masks_directory)
    features = reduction.reduce(num_processes=num_processes)
    features.to_csv(output_file)


@reduction.command()
@click.argument('image-directory', type=click.Path())
@click.argument('masks-directory', type=click.Path())
@click.argument('patch_file', type=click.Path())
@click.argument('output-file', type=click.Path())
@click.option('--num-processes', default=2,
              help="Num of processes to use for the reduction.")
def intensity_from_patch(image_directory, masks_directory, patch_file,
                         output_file, num_processes):
    patch_frame = pd.DataFrame.from_csv(patch_file)
    reduction = PatchIntensityFeaturesReduction(image_directory,
                                                masks_directory)
    features = reduction.reduce(patch_frame, num_processes=num_processes)
    features.to_csv(output_file)


@reduction.command()
@click.argument('image-directory', type=click.Path())
@click.argument('masks-directory', type=click.Path())
@click.argument('patch_file', type=click.Path())
@click.argument('output-file', type=click.Path())
@click.option('--num-processes', default=2,
              help="Num of processes to use for the reduction.")
def texture_from_patch(image_directory, masks_directory, patch_file,
                       output_file, num_processes):
    patch_frame = pd.DataFrame.from_csv(patch_file)
    reduction = PatchTextureFeaturesReduction(image_directory,
                                              masks_directory)
    features = reduction.reduce(patch_frame, num_processes=num_processes)
    features.to_csv(output_file)


@cli.command()
@click.argument('image-file', type=click.Path())
@click.argument('mask-file', required=False, type=click.Path())
def detect_blobs(image_file, mask_file):
    img, msk = mia.utils.preprocess_image(image_file, mask_file)
    blob_df = mia.features.blobs.detect_blobs(img, msk)
    mia.plotting.plot_blobs(img, blob_df[['x', 'y', 'radius']].as_matrix())


@cli.command()
@click.argument('image-file', type=click.Path())
@click.argument('mask-file', required=False, type=click.Path())
def detect_linear(image_file, mask_file):
    img, msk = mia.utils.preprocess_image(image_file, mask_file)
    _, line_image = mia.features.linear_structure.detect_linear(img, msk)
    img = transform.pyramid_reduce(img, 4)
    mia.plotting.plot_linear_structure(img, line_image)


###############################################################################
# MIA Analysis
###############################################################################

@cli.group()
def analysis():
    pass


@analysis.command()
@click.argument('csv-file', type=click.Path())
@click.argument('columns', nargs=-1)
@click.option('--output-file', '-o', default=None,
              help="Name of output file to store results of analysis in")
def tSNE(csv_file, columns, output_file):
    data_frame = pd.DataFrame.from_csv(csv_file)
    data_frame = data_frame[list(columns)]
    fit_output = mia.analysis.tSNE(data_frame)
    fit_output.to_csv(output_file)


###############################################################################
# MIA Plotting
###############################################################################


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
    mia.plotting.plot_scatter_2d(df, label_column, annotate)


@plotting.command()
@click.argument('csv-file', type=click.Path())
@click.option('--label-column', '-l', default=None,
              help="Name of column to use as the class labels")
def scatter_matrix(csv_file, label_column):
    df = pd.DataFrame.from_csv(csv_file)
    mia.plotting.plot_scattermatrix(df, label_column)


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
    mia.plotting.plot_median_image_matrix(df, img_path, label_column,
                                          output_file=output_file,
                                          raw_features_csv=features_csv)


if __name__ == '__main__':
    cli(obj={})
