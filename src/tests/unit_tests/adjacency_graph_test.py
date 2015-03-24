import unittest
import nose.tools
from ..test_utils import *

from mia.features._adjacency_graph import Graph

class GraphTests(unittest.TestCase):

    def test_create(self):
        g = Graph()

        for i in range(1,10):
            g.add_node(i,i)

            for j in range(1,10):
                if i % j == 0:
                    g.add_adjacent(i,j)

        assert_lists_equal(g.get_adjacents(1), [1])
        assert_lists_equal(g.get_adjacents(7), [1, 7])
        assert_lists_equal(g.get_adjacents(8), [1, 2, 4, 8])

        for i in range(1, 10):
            nose.tools.assert_equal(g.get_node(i), i)
