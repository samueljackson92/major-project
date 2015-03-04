class Node:
    def __init__(self, data):
        self._adjacents = []
        self.set_data(data)

    def add_adjacent(self, node):
        self._adjacents.append(node)

    def get_adjacents(self):
        return self._adjacents

    def get_data(self):
        return self._data

    def set_data(self, data):
        self._data = data


class Graph:
    def __init__(self):
        self._nodes = {}

    def add_node(self, key, val):
        self._nodes[key] = Node(val)

    def add_adjacent(self, key, node_key):
        self._nodes[key].add_adjacent(node_key)

    def get_node(self, key):
        return self._nodes[key].get_data()

    def get_adjacents(self, key):
        return self._nodes[key].get_adjacents()

    def iterate(self):
        for key, node in self._nodes.iteritems():
            yield key, node.get_adjacents()
