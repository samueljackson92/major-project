""" Module implements a coranking matrix for checking the quality of a
lower dimensional mapping produce by manifold learning algorithms.
"""

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


def trustworthiness(high_data, low_data, k):
    n, m = high_data.shape
    w = 2.0 / (n*k*(2*n - 3*k - 1))

    high_distance = pairwise_distances(high_data)
    low_distance = pairwise_distances(low_data)

    high_ranking = high_distance.argsort(axis=1).argsort(axis=1)

    high_dist_indicies = high_distance.argsort(axis=1)
    low_dist_indicies = low_distance.argsort(axis=1)

    high_k_neighbours = high_dist_indicies[:, 1:k+1]
    low_k_neighbours = low_dist_indicies[:, 1:k+1]

    neighbour_differences = [np.setdiff1d(ln, hn) for ln, hn in
                             zip(low_k_neighbours, high_k_neighbours)]

    rank_differences = np.array([sum(rank[diffs] - k) for rank, diffs in
                                zip(high_ranking, neighbour_differences)])

    return 1 - w*rank_differences.sum()


def continuity(high_data, low_data, k):
    return trustworthiness(low_data, high_data, k)
