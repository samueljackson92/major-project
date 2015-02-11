import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from skimage import draw

def plot_multiple_images(images):
    """Plot a list of images on horizontal subplots

    :param images -- the list of images to plot
    """
    fig = plt.figure()

    num_images = len(images)
    for i, image in enumerate(images):
        axis = fig.add_subplot(1, num_images, i+1)
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
