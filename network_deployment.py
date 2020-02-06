import numpy as np
import math
import networkx as nx
import scipy.spatial
import random


def generate_square(num_nodes, radius, side_len):
    """Generate a network into a square shaped area using random placement and unit disk graph communication model.

    :param num_nodes: number of sensors
    :param radius: communication radius of the sensors
    :param side_len: length of the sides of the square
    :return: A list of generated node positions and the network connectivity graph
    """

    # try 1000 times, to generate a random graph with these parameters
    for X in range(0, 1000):
        # make graph, add nodes and random positions
        connectivity_graph = nx.Graph()
        connectivity_graph.add_nodes_from(range(0, num_nodes))
        positions = dict()
        positions_ = list()
        for i in range(0, num_nodes):
            positions[i] = [np.random.rand()*side_len, np.random.rand()*side_len]
            positions_.append(positions[i])

        # calculate pairwise distances
        y = scipy.spatial.distance.pdist(np.asarray(positions_), 'euclidean')
        dist_matrix = scipy.spatial.distance.squareform(y)

        # add edges based on radius
        for i in range(0, num_nodes-1):
            for j in range(i, num_nodes):
                if dist_matrix[i, j] <= radius and i != j:
                    connectivity_graph.add_edge(i, j)
        # if the generated network is not connected, generate a new one
        if nx.is_connected(connectivity_graph):
            return positions, connectivity_graph

    assert False, 'Error: Unable to create connected graph using the provided parameters!\n'


def generate_square_quasi(num_nodes, radius, side_len, p=0.5):
    """Generate a network into a square shaped area using random placement and QUD graph communication model.

    :param num_nodes: number of sensors
    :param radius: communication radius of the sensors
    :param side_len: length of the sides of the square
    :param p: parameter for the quasi communication model
    :return: A list of generated node positions and the network connectivity graph
    """
    radius2 = radius
    radius1 = radius * p
    # try 1000 times, to generate a random graph with these parameters
    for X in range(0, 1000):
        # make graph, add nodes and random positions
        connectivity_graph = nx.Graph()
        connectivity_graph.add_nodes_from(range(0, num_nodes))
        positions = dict()
        positions_ = list()
        for i in range(0, num_nodes):
            positions[i] = [np.random.rand()*side_len, np.random.rand()*side_len]
            positions_.append(positions[i])

        # calculate pairwise distances
        y = scipy.spatial.distance.pdist(np.asarray(positions_), 'euclidean')
        dist_matrix = scipy.spatial.distance.squareform(y)

        # add edges based on radius and param p
        for i in range(0, num_nodes-1):
            for j in range(i, num_nodes):
                if dist_matrix[i, j] <= radius1 and i != j:
                    connectivity_graph.add_edge(i, j)
                if radius1 < dist_matrix[i, j] <= radius2:
                    asd = np.random.rand()
                    alpha = radius1 / radius2
                    prob = (alpha/(1 - alpha)) * ((radius2/dist_matrix[i, j]) - 1)

                    if asd < prob:
                        connectivity_graph.add_edge(i, j)

        # if the generated network is not connected, generate a new one
        if nx.is_connected(connectivity_graph):
            return positions, connectivity_graph

    assert False, 'Error: Unable to create connected graph using the provided parameters!\n'


def generate_grid(num_nodes, radius, side_len, noise=0.75):
    """Generate a network into a square shaped area using perturbed grid placement and UDG communication model.

    :param num_nodes: number of sensors
    :param radius: communication radius of the sensors
    :param side_len: length of the sides of the square
    :param noise: noise used for perturbation
    :return: A list of generated node positions and the network connectivity graph
    """
    assert (math.sqrt(num_nodes).is_integer()), 'the number of sensors must be a square number'

    # try 1000 times, to generate a random graph with these parameters
    for X in range(0, 1000):
        h = int(math.sqrt(num_nodes))
        node_dist = side_len / h
        max_noise = node_dist * noise

        connectivity_graph = nx.Graph()
        connectivity_graph.add_nodes_from(range(0, num_nodes))
        positions_ = list()
        positions = dict()

        # deploy network on grid
        for i in range(0, h):
            for j in range(0, h):
                rx = random.uniform(-1*max_noise, max_noise)
                ry = random.uniform(-1*max_noise, max_noise)
                positions[i + j * h] = [i * node_dist + rx, j * node_dist + ry]
                positions_.append(positions[i + j * h])

        y = scipy.spatial.distance.pdist(np.asarray(positions_), 'euclidean')
        dist_matrix = scipy.spatial.distance.squareform(y)

        # add edges based on radius
        for i in range(0, num_nodes-1):
            for j in range(i, num_nodes):
                if dist_matrix[i, j] <= radius and i != j:
                    connectivity_graph.add_edge(i, j)

        if nx.is_connected(connectivity_graph):
            return positions, connectivity_graph

    assert False, 'Error: the graph is not connected with these parameters!\n'


def generate_grid_quasi(num_nodes, radius, side_len, p=0.5, noise=0.75):
    """Generate a network into a square shaped area using perturbed grid placement and QUDG communication model.

    :param num_nodes: number of sensors
    :param radius: communication radius of the sensors
    :param side_len: length of the sides of the square
    :param p: parameter for the quasi communication model
    :param noise: noise used for perturbation
    :return: A list of generated node positions and the network connectivity graph
    """
    assert (math.sqrt(num_nodes).is_integer()), 'Error: the number of sensors must be a square number\n'

    radius2 = radius
    radius1 = radius * p

    # try 1000 times, to generate a random graph with these parameters
    for X in range(0, 1000):
        h = int(math.sqrt(num_nodes))
        node_dist = side_len / h
        max_noise = node_dist*noise

        connectivity_graph = nx.Graph()
        connectivity_graph.add_nodes_from(range(0, num_nodes))
        positions_ = list()
        positions = dict()

        # deploy network on grid
        for i in range(0, h):
            for j in range(0, h):
                rx = random.uniform(-1*max_noise, max_noise)  # max_noise*np.random.rand() - max_noise
                ry = random.uniform(-1*max_noise, max_noise)  # max_noise*np.random.rand() - max_noise
                positions[i + j * h] = [i * node_dist + rx, j * node_dist + ry]
                positions_.append(positions[i + j * h])

        y = scipy.spatial.distance.pdist(np.asarray(positions_), 'euclidean')
        dist_matrix = scipy.spatial.distance.squareform(y)

        # add edges based on radius
        for i in range(0, num_nodes-1):
            for j in range(i, num_nodes):
                if dist_matrix[i, j] <= radius1 and i != j:
                    connectivity_graph.add_edge(i, j)
                if radius1 < dist_matrix[i, j] <= radius2:
                    asd = np.random.rand()
                    alpha = radius1 / radius2
                    prob = (alpha/(1 - alpha)) * ((radius2/dist_matrix[i, j]) - 1)
                    if asd < prob:
                        connectivity_graph.add_edge(i, j)

        if nx.is_connected(connectivity_graph):
            return positions, connectivity_graph

    assert False, 'Error: Unable to create connected graph using the provided parameters!\n'
