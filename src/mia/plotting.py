import math
import logging
import os.path
import functools
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn as sns
from skimage import io, transform

from mia.utils import transform_2d

logger = logging.getLogger(__name__)


def plot_multiple_images(images):
    """Plot a list of images on horizontal subplots

    :param images: the list of images to plot
    """
    fig = plt.figure()

    num_images = len(images)
    for i, image in enumerate(images):
        fig.add_subplot(1, num_images, i+1)
        plt.imshow(image, cmap=cm.Greys_r)

    plt.show()


def plot_region_props(image, regions):
    """Plot the output of skimage.regionprops along with the image.

    Original code from
    http://scikit-image.org/docs/dev/auto_examples/plot_regionprops.html

    :param image: image to plot
    :param regions: the regions output from the regionprops function
    """
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=plt.cm.gray)

    for props in regions:
        y0, x0 = props.centroid
        orientation = props.orientation
        x1 = x0 + math.cos(orientation) * 0.5 * props.major_axis_length
        y1 = y0 - math.sin(orientation) * 0.5 * props.major_axis_length
        x2 = x0 - math.sin(orientation) * 0.5 * props.minor_axis_length
        y2 = y0 - math.cos(orientation) * 0.5 * props.minor_axis_length

        ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
        ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
        ax.plot(x0, y0, '.g', markersize=15)

        minr, minc, maxr, maxc = props.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        ax.plot(bx, by, '-b', linewidth=2.5)

    plt.show()


def plot_blobs(img, blobs):
    """Plot the output of blob detection on an image.

    :param img: the image to plot
    :param blobs: list of blobs found in the image
    """

    fig, ax = plt.subplots(1, 1)
    ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    ax.set_axis_off()

    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
        ax.add_patch(c)

    plt.show()


def plot_image_orthotope(image_orthotope, titles=None):
    """ Plot an image orthotope

    :param image_orthotope: the orthotope of images to plot
    :param titles: titles for each of the images in orthotope
    """

    fig, ax = plt.subplots(*image_orthotope.shape[:2])

    if titles is not None:
        title_iter = titles.__iter__()

    for i in range(image_orthotope.shape[0]):
        for j in range(image_orthotope.shape[1]):
            if titles is not None:
                ax[i][j].set_title(title_iter.next())

            ax[i][j].imshow(image_orthotope[i][j], cmap=plt.cm.gray)
            ax[i][j].axis('off')

    plt.show()


def plot_scatter_2d(data_frame, label_name=None, annotate=False):
    """ Create a scatter plot from a pandas data frame

    :param data_frame: data frame containing the lower dimensional mapping
    :param label_name: name of the column containing the class label for the
                       image
    """
    label_name = 1 if label_name is None else label_name

    ax = data_frame.plot(kind='scatter', x=0, y=1, c=label_name,
                         cmap=plt.cm.Spectral, s=50)

    if annotate:
        def annotate_df(row):
            ax.text(row.values[0], row.values[1], row.name, fontsize=10)
        data_frame.apply(annotate_df, axis=1)

    plt.show()


def plot_scattermatrix(data_frame, label_name=None):
    """ Create a scatter plot matrix from a pandas data frame

    :param data_frame: data frame containing the lower dimensional mapping
    :param label_name: name of the column containing the class label for the
                       image
    """
    column_names = filter(lambda x: x != label_name, data_frame.columns.values)
    sns.pairplot(data_frame, hue=label_name, size=1.5, vars=column_names)
    plt.show()


def plot_median_image_matrix(data_frame, img_path, label_name=None,
                             raw_features_csv=None, output_file=None):
    """Plot images from a dataset according to their position defined by the
    lower dimensional mapping

    This rebins the points in the mapping using a 2D histogram then takes
    the median point in each bin. The image corresponding to this point is
    then plotted for that bin.

    :param data_frame: data frame defining the lower dimensional mapping
    :param img_path: path to find the images in
    :param label_name: name of the column in the data frame containing the
                       class labels
    :param output_file: file name to output the resulting image to
    """
    blobs = _load_blobs(raw_features_csv) if raw_features_csv else None

    grid = _bin_data_frame_2d(data_frame)
    axes_iter = _prepare_figure(len(grid))
    transform_2d(_prepare_median_image, grid, img_path, blobs, axes_iter)

    logger.info("Saving image")
    plt.savefig(output_file, bbox_inches='tight', dpi=3000)


def _load_blobs(raw_features_csv):
    features = pd.DataFrame.from_csv(raw_features_csv)
    return features[['x', 'y', 'radius', 'image_name']]


def _prepare_figure(size):
    fig, axs = plt.subplots(size, size,
                            gridspec_kw={"wspace": 0, "hspace": 0})
    axs = np.array(axs).flatten()
    return iter(axs)


def _prepare_median_image(img_name, path, blobs_df, axs_iter):
    """ Function to load an image if one is present in the square"""
    scale_factor = 16

    ax = axs_iter.next()
    ax.set_axis_off()
    ax.set_aspect('auto')

    blobs = _select_blobs(blobs_df, img_name, scale_factor)

    if img_name == '':
        _add_blank_image_to_axis(ax, scale_factor)
    else:
        logger.info("Loading image %s" % img_name)

        path = os.path.join(path, img_name)
        img = _load_image(path, scale_factor)
        _add_image_to_axis(img, ax, img_name)
        _add_blobs_to_axis(ax, blobs)


def _select_blobs(blobs_df, img_name, scale_factor):
    b = blobs_df[blobs_df['image_name'] == img_name]
    b = b[['x', 'y', 'radius']].as_matrix()
    b /= scale_factor
    return b


def _load_image(path, scale_factor):
    img = io.imread(path, as_grey=True)
    img = transform.resize(img, (img.shape[0]/scale_factor,
                                 img.shape[1]/scale_factor))
    return img


def _add_image_to_axis(img, ax, img_name):
    ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    ax.text(20, 0, img_name, style='italic', fontsize=3, color='white',
            transform=ax.transData)


def _add_blank_image_to_axis(ax, scale_factor):
    img = np.ones((3328/scale_factor, 2560/scale_factor))
    ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)


def _add_blobs_to_axis(ax, blobs):
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
        ax.add_patch(c)


def _bin_data_frame_2d(data_frame):
    hist, xedges, yedges = np.histogram2d(data_frame['0'], data_frame['1'])
    grid = []
    for x_bounds in zip(xedges, xedges[1:]):
        row = []
        for y_bounds in zip(yedges, yedges[1:]):
            entries = _find_points_in_bounds(data_frame, x_bounds, y_bounds)
            name = _find_median_image_name(entries)
            row.append(name)
        grid.append(row)

    return np.rot90(np.array(grid))


def _find_median_image_name(data_frame):
    name = ''
    num_rows = data_frame.shape[0]

    if num_rows > 0:
        sorted_df = data_frame.sort(['0', '1'])
        med_df = sorted_df.iloc[[num_rows/2]]
        name = med_df.index.values[0]

    return name


def _find_points_in_bounds(data_frame, x_bounds, y_bounds):
    xlower, xupper = x_bounds
    ylower, yupper = y_bounds
    xbounds = (data_frame['0'] >= xlower) & (data_frame['0'] < xupper)
    ybounds = (data_frame['1'] >= ylower) & (data_frame['1'] < yupper)
    return data_frame[xbounds & ybounds]
