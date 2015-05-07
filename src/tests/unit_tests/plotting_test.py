import nose.tools
import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import mia
from mia.plotting import *
from ..test_utils import get_file_path


class PlottingTests(unittest.TestCase):

    @classmethod
    def setupClass(cls):
        blobs_path = get_file_path("blob_detection.csv")
        cls._df = pd.DataFrame.from_csv(blobs_path)

        # run plots on a seperate thread
        plt.ion()

    def tearDown(cls):
        # close plots after test is run
        plt.close()

    def test_plot_multiple_images(self):
        img1 = np.random.rand(10, 10)
        img2 = np.random.rand(10, 10)

        plot_multiple_images([img1, img2])

    def test_plot_region_props(self):
        img = np.random.rand(10, 10)
        region = [2, 2, 5, 5]

        plot_region_props(img, [region])

    def test_plot_linear_structure(self):
        img = np.random.rand(10, 10)
        line_image = np.zeros(img.shape)
        line_image[5, 5] = 1

        plot_linear_structure(img, line_image)

    def test_plot_blobs(self):
        img = np.random.rand(10, 10)
        blob = [5, 5, 3]

        plot_blobs(img, [blob])

    def test_plot_image_orthotope(self):
        img1 = np.random.rand(10, 10)
        img2 = np.random.rand(10, 10)
        img3 = np.random.rand(10, 10)
        img4 = np.random.rand(10, 10)

        cube = np.array([[img1, img2], [img3, img4]])
        plot_image_orthotope(cube)

    def test_plot_image_orthotope_with_titles(self):
        img1 = np.random.rand(10, 10)
        img2 = np.random.rand(10, 10)
        img3 = np.random.rand(10, 10)
        img4 = np.random.rand(10, 10)
        titles = ['x', 'y', 'w', 'u']

        cube = np.array([[img1, img2], [img3, img4]])
        plot_image_orthotope(cube, titles)

    def test_plot_risk_classes(self):
        self._df['class'] = np.random.randint(4, self._df.shape[0])
        plot_risk_classes(self._df, ['class'])

    def test_plot_risk_classes_single(self):
        self._df['class'] = np.random.randint(4, self._df.shape[0])
        plot_risk_classes_single(self._df, ['class'])

    def test_plot_scatter_2d(self):
        data = np.random.rand(10, 2)
        df = pd.DataFrame(data, columns=['x', 'y'])
        df['class'] = np.random.randint(4, self._df.shape[0])
        plot_scatter_2d(df, ['x', 'y'], df['class'])

    def test_plot_scatter_2d_with_annotate(self):
        data = np.random.rand(10, 2)
        df = pd.DataFrame(data, columns=['x', 'y'])
        df['class'] = np.random.randint(4, self._df.shape[0])
        plot_scatter_2d(df, ['x', 'y'], df['class'], annotate=True)

    def test_plot_scatter_2d_incorrect_dimensions(self):
        data = np.random.rand(10, 3)
        df = pd.DataFrame(data, columns=['x', 'y', 'z'])
        df['class'] = np.random.randint(4, self._df.shape[0])
        try:
            plot_scatter_2d(df, ['x', 'y', 'z'], df['class'])
        except ValueError, e:
            nose.tools.assert_equal(e.message,
                                    "Number of columns must be exactly 2")

    def test_plot_scatter_3d(self):
        data = np.random.rand(10, 3)
        df = pd.DataFrame(data, columns=['x', 'y', 'z'])
        df['class'] = np.random.randint(4, self._df.shape[0])
        plot_scatter_3d(df, ['x', 'y', 'z'], df['class'])

    def test_plot_scatter_3d_incorrect_dimensions(self):
        data = np.random.rand(10, 4)
        df = pd.DataFrame(data, columns=['x', 'y', 'z', 'w'])
        df['class'] = np.random.randint(4, self._df.shape[0])
        try:
            plot_scatter_3d(df, ['x', 'y', 'z', 'w'], df['class'])
        except ValueError, e:
            nose.tools.assert_equal(e.message,
                                    "Number of columns must be exactly 3")

    def test_plot_mapping_2d(self):
        data = np.random.rand(10, 2)
        df = pd.DataFrame(data, columns=['x', 'y'])
        labels = pd.Series(np.random.randint(4, size=self._df.shape[0]))

        index_a = pd.Series(np.arange(5))
        index_b = pd.Series(np.arange(5, 10))

        plot_mapping_2d(df, index_a, index_b, labels)

    def test_plot_mapping_3d(self):
        data = np.random.rand(10, 3)
        df = pd.DataFrame(data, columns=['x', 'y', 'z'])
        labels = pd.Series(np.random.randint(4, size=self._df.shape[0]))

        index_a = pd.Series(np.arange(5))
        index_b = pd.Series(np.arange(5, 10))

        plot_mapping_3d(df, index_a, index_b, labels)
