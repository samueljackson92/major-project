class Node:
    def __init__(self, data):
        """Create a new node

        :param data: the data to store with this Node.
        """
        self._adjacents = []
        self.set_data(data)

    def add_adjacent(self, node):
        """Add a connection to another node

        :param node: the other node to connect with.
        """
        self._adjacents.append(node)

    def get_adjacents(self):
        """Add a list of nodes which are neighbours with this node.

        :returns: list -- the list of nodes which are neighbours.
        """
        return self._adjacents

    def get_data(self):
        """Get the data stored on this node

        :returns: Object -- the data stored with this node.
        """
        return self._data

    def set_data(self, data):
        """Set the data stored on this node

        :param data: -- the data to store in this node.
        """
        self._data = data


class Graph:
    def __init__(self):
        self._nodes = {}

    def add_node(self, key, val):
        """Add a node to the graph.

        :param key: key used to reference the node.
        :param val: value to store with the node.
        """
        self._nodes[key] = Node(val)

    def add_adjacent(self, key, node_key):
        """Add a connection between two nodes in the graph.

        :param key: key used to reference the node.
        :param node_key: value to reference the other node.
        """
        self._nodes[key].add_adjacent(node_key)

    def get_node(self, key):
        """Get the data stored in a node.

        :param key: key used to reference the node.
        :returns: the data stored in the node.
        """
        return self._nodes[key].get_data()

    def get_adjacents(self, key):
        """Get neighbours of a node.

        :param key: key used to reference the node.
        :returns: list -- neighbours of the node.
        """
        return self._nodes[key].get_adjacents()

    def iterate(self):
        """Iterate over all nodes in the graph with a list of their neighbours.

        :returns: iterator -- tuple of (key, neighbours of the node).
        """
        for key, node in self._nodes.iteritems():
            yield key, node.get_adjacents()
