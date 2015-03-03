import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import seaborn as sns
from skimage import io, transform
from mammogram.utils import normalise_image


def plot_multiple_images(images):
    """Plot a list of images on horizontal subplots

    :param images: -- the list of images to plot
    """
    fig = plt.figure()

    num_images = len(images)
    for i, image in enumerate(images):
        fig.add_subplot(1, num_images, i+1)
        plt.imshow(image, cmap=cm.Greys_r)

    plt.show()


def plot_region_props(image, regions):
    """Plot the output of skimage.regionprops along with the image.

    Original code from http://scikit-image.org/docs/dev/auto_examples/plot_regionprops.html

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

    Original code from
    """

    fig, ax = plt.subplots(1, 1)
    # ax.set_title(title)
    ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
        ax.add_patch(c)

    plt.show()


def plot_image_orthotope(image_orthotope, titles=None):
    """ Plot an image orthotope """

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


def plot_scatter_2d(data_frame, label_name=None):
    """ Create a scatter plot from a pandas data frame """
    label_name = 1 if label_name is None else label_name
    data_frame.plot(kind='scatter', x=0, y=1, c=label_name, cmap=plt.cm.Spectral)
    plt.show()


def plot_scattermatrix(data_frame, label_name=None):
    """ Create a scatter plot matrix from a pandas data frame """
    column_names = filter(lambda x: x != 'class', data_frame.columns.values)
    sns.pairplot(data_frame, hue=label_name, size=1.5, vars=column_names)
    plt.show()


def bin_data_frame(data_frame):
    hist, xedges, yedges = np.histogram2d(data_frame['0'], data_frame['1'])
    grid = []
    for xlower, xupper in zip(xedges, xedges[1:]):
        row = []
        for ylower, yupper in zip(yedges, yedges[1:]):
            xbounds = (data_frame['0'] >= xlower) & (data_frame['0'] < xupper)
            ybounds = (data_frame['1'] >= ylower) & (data_frame['1'] < yupper)
            entires = data_frame[xbounds & ybounds]

            name = ''
            num_rows = entires.shape[0]
            if num_rows > 0:
                sorted_df = entires.sort(['0', '1'])
                med_df = sorted_df.iloc[[num_rows/2]]
                name = med_df.index.values[0]

            row.append(name)
        grid.append(row)

    return np.array(grid)


def transform_grid(f, grid):
    out_grid = []
    for row in grid:
        out_row = []
        for value in row:
            out_value = f(value)
            out_row.append(out_value)
        out_grid.append(out_row)
    return np.array(out_grid)


def filter_func(x):
    import os.path
    path = '/Volumes/Seagate/MammoData/pngs'

    if x == '':
        img = np.empty((3328, 2560))
    else:
        img = io.imread(os.path.join(path, x), as_grey=True)
        img = transform.resize(img, (3328, 2560))
        print img.shape

    return img


def plot_median_image_matrix(data_frame, label_name=None):
    grid = bin_data_frame(data_frame)
    images = transform_grid(filter_func, grid)

    rows = []
    for row in images:
        rows.append(np.hstack(row))

    io.imsave('/Users/samuel/Desktop/test_grid.jpg', np.vstack(rows))
