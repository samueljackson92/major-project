import unittest
import nose.tools
import numpy as np

from mia.utils import (preprocess_image, cluster_image, clusters_from_labels,
                       sort_clusters_by_density)
from ..test_utils import get_file_path


class UtilsRegressionTest(unittest.TestCase):

    def test_load_real_image_and_mask(self):
        img_path = get_file_path("mias/mdb154.png")
        msk_path = get_file_path("mias/masks/mdb154_mask.png")
        img, msk = preprocess_image(img_path, msk_path)

        nose.tools.assert_equal(img.shape, (1024, 980))
        nose.tools.assert_equal(msk.shape, img.shape)

    def test_cluster_image(self):
        img_path = get_file_path("mias/mdb154.png")
        msk_path = get_file_path("mias/masks/mdb154_mask.png")
        img, msk = preprocess_image(img_path, msk_path)

        labels = cluster_image(img)

        nose.tools.assert_equal(labels.shape, img.shape)
        nose.tools.assert_equal(labels.dtype, 'int32')
        nose.tools.assert_equal(labels.max(), 4)
        nose.tools.assert_equal(labels.min(), 0)

    def test_clusters_from_labels(self):
        img_path = get_file_path("mias/mdb154.png")
        msk_path = get_file_path("mias/masks/mdb154_mask.png")
        img, msk = preprocess_image(img_path, msk_path)

        labels = cluster_image(img)
        clusters = clusters_from_labels(img, labels)

        expected_shape = (5, img.shape[0], img.shape[1])
        nose.tools.assert_equal(clusters.shape, expected_shape)

    def test_sort_clusters_by_density(self):
        img_path = get_file_path("mias/mdb154.png")
        msk_path = get_file_path("mias/masks/mdb154_mask.png")
        img, msk = preprocess_image(img_path, msk_path)

        labels = cluster_image(img)
        clusters = clusters_from_labels(img, labels)
        srtd_clusters = sort_clusters_by_density(clusters)

        expected_shape = (5, img.shape[0], img.shape[1])
        nose.tools.assert_equal(srtd_clusters.shape, expected_shape)
        nose.tools.assert_equal(np.count_nonzero(srtd_clusters[0]), 0)
        nose.tools.assert_almost_equal(srtd_clusters[1].mean(),
                                       0.01, delta=0.015)
        nose.tools.assert_almost_equal(srtd_clusters[2].mean(),
                                       0.03, delta=0.015)
        nose.tools.assert_almost_equal(srtd_clusters[3].mean(),
                                       0.06, delta=0.015)
        nose.tools.assert_almost_equal(srtd_clusters[4].mean(),
                                       0.06, delta=0.015)
